"""Server HTTP Pocket TTS Indonesian (port 6001).

GET  /                  dashboard web
GET  /info              info + daftar suara
POST /normalize         pratinjau teks yang dieja
GET  /voices            daftar suara di voices/
POST /tts               JSON {"text","voice","eos_threshold"} -> audio/wav
GET  /tts?text=...&voice=klon   -> audio/wav
POST /voices/{nama}     unggah wav/mp3 sebagai suara baru (multipart: file)
"""
import io
import os
import re
import subprocess
import sys
import threading

import scipy.io.wavfile
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize  # noqa: E402
from pocket_tts import TTSModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(ROOT, "voices")
CONFIG = ("hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml"
          "@17257664e384561c957b02ac92edd1a24807f0e5")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "klon")
DEFAULT_EOS = float(os.environ.get("EOS_THRESHOLD", "-6.0"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "600"))

app = FastAPI(title="Pocket TTS Indonesian")
_lock = threading.Lock()          # model dipakai satu permintaan per waktu
_models = {}                      # eos_threshold -> TTSModel
_states = {}                      # (eos, voice, mtime) -> voice state


def _model(eos: float) -> TTSModel:
    if eos not in _models:
        _models[eos] = TTSModel.load_model(config=CONFIG, eos_threshold=eos)
    return _models[eos]


def _voice_path(nama: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", nama):
        raise HTTPException(400, "nama suara hanya boleh huruf/angka/_/-")
    p = os.path.join(VOICE_DIR, nama + ".wav")
    if not os.path.isfile(p):
        raise HTTPException(404, f"suara '{nama}' tidak ada")
    return p


def _synth(text: str, voice: str, eos: float) -> bytes:
    text = normalize(text)
    if not text:
        raise HTTPException(400, "teks kosong")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"teks terlalu panjang (maks {MAX_CHARS} karakter)")
    path = _voice_path(voice)
    with _lock:
        m = _model(eos)
        key = (eos, voice, os.path.getmtime(path))
        if key not in _states:
            _states[key] = m.get_state_for_audio_prompt(path)
        audio = m.generate_audio(_states[key], text)
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, m.sample_rate, audio.numpy())
    return buf.getvalue()


def _voices():
    return sorted(f[:-4] for f in os.listdir(VOICE_DIR) if f.endswith(".wav"))


class Req(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    eos_threshold: float = DEFAULT_EOS


@app.on_event("startup")
def warmup():
    with _lock:
        _model(DEFAULT_EOS)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


class NormReq(BaseModel):
    text: str


@app.post("/normalize")
def norm(r: NormReq):
    return {"text": normalize(r.text)}


@app.get("/voices/{nama}/audio")
def voice_audio(nama: str):
    return FileResponse(_voice_path(nama), media_type="audio/wav")


@app.delete("/voices/{nama}")
def delete_voice(nama: str):
    os.remove(_voice_path(nama))
    return {"ok": True}


@app.get("/info")
def info():
    return {"service": "pocket-tts-indonesian", "default_voice": DEFAULT_VOICE,
            "voices": _voices(), "max_chars": MAX_CHARS}


@app.get("/voices")
def voices():
    return {"voices": _voices()}


@app.post("/tts")
def tts_post(r: Req):
    return Response(_synth(r.text, r.voice, r.eos_threshold), media_type="audio/wav")


@app.get("/tts")
def tts_get(text: str, voice: str = DEFAULT_VOICE,
            eos_threshold: float = Query(DEFAULT_EOS)):
    return Response(_synth(text, voice, eos_threshold), media_type="audio/wav")


@app.post("/voices/{nama}")
async def upload_voice(nama: str, file: UploadFile = File(...)):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", nama):
        raise HTTPException(400, "nama suara hanya boleh huruf/angka/_/-")
    tmp = os.path.join(VOICE_DIR, f".upload_{nama}")
    with open(tmp, "wb") as f:
        f.write(await file.read())
    out = os.path.join(VOICE_DIR, nama + ".wav")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-ar", "48000", "-ac", "1", "-f", "wav", out],
                       capture_output=True, text=True)
    os.remove(tmp)
    if r.returncode != 0:
        raise HTTPException(400, "file audio tidak valid")
    return JSONResponse({"ok": True, "voice": nama})
