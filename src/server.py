"""Server HTTP Pocket TTS Indonesian.

GET  /                  dashboard web
GET  /info              info + daftar suara
POST /normalize         pratinjau teks yang dieja
GET  /voices            daftar suara di voices/
POST /tts               JSON {"text","voice","eos_threshold"} -> audio/wav
GET  /tts?text=...&voice=klon   -> audio/wav
POST /script            (Beta) JSON {"text","style"} -> naskah segmen dari AI
POST /tts/script        (Beta) JSON {"segmen":[{"teks","jeda_ms"}],"voice"} -> audio/wav
POST /tts/beta          (Beta) satu langkah: {"text","style","voice"} -> audio/wav
POST /voices/{nama}     unggah wav/mp3 sebagai suara baru (multipart: file)
DELETE /voices/{nama}   hapus suara
"""
import io
import os
import re
import subprocess
import sys
import threading
from typing import List, Optional

import numpy as np
import scipy.io.wavfile
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
import beta  # noqa: E402
from normalize import normalize  # noqa: E402
from pocket_tts import TTSModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(ROOT, "voices")
CONFIG = ("hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml"
          "@17257664e384561c957b02ac92edd1a24807f0e5")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "klon")
DEFAULT_EOS = float(os.environ.get("EOS_THRESHOLD", "-6.0"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "600"))
MAX_CHARS_BETA = int(os.environ.get("MAX_CHARS_BETA", "1500"))

app = FastAPI(title="Pocket TTS Indonesian")
_lock = threading.Lock()          # satu sintesis per waktu
_model_obj = None
_states = {}                      # (voice, mtime) -> voice state


def _model() -> TTSModel:
    global _model_obj
    if _model_obj is None:
        _model_obj = TTSModel.load_model(config=CONFIG, eos_threshold=DEFAULT_EOS)
    return _model_obj


def _voice_path(nama: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", nama):
        raise HTTPException(400, "nama suara hanya boleh huruf/angka/_/-")
    p = os.path.join(VOICE_DIR, nama + ".wav")
    if not os.path.isfile(p):
        raise HTTPException(404, f"suara '{nama}' tidak ada")
    return p


def _generate(text: str, voice: str, eos: float) -> np.ndarray:
    """Satu sintesis (pegang kunci). Mengembalikan float32 mono."""
    path = _voice_path(voice)
    with _lock:
        m = _model()
        m.eos_threshold = eos
        key = (voice, os.path.getmtime(path))
        if key not in _states:
            _states[key] = m.get_state_for_audio_prompt(path)
        return m.generate_audio(_states[key], text).numpy().astype(np.float32).reshape(-1)


def _wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, _model().sample_rate, audio)
    return buf.getvalue()


def _synth(text: str, voice: str, eos: float) -> bytes:
    text = normalize(text)
    if not text:
        raise HTTPException(400, "teks kosong")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"teks terlalu panjang (maks {MAX_CHARS} karakter)")
    return _wav(_generate(text, voice, eos))


def _synth_segmen(segmen, voice: str, eos: float):
    if not segmen:
        raise HTTPException(400, "naskah kosong")
    if len(segmen) > 60:
        raise HTTPException(413, "terlalu banyak segmen (maks 60)")
    sr = _model().sample_rate
    potongan, ulang = [], 0
    for s in segmen:
        teks = beta.bersihkan(s.teks)
        if not re.search(r"[A-Za-z]", teks):
            continue
        jeda = s.jeda_ms if s.jeda_ms else beta.jeda_default(teks)
        jeda = max(80, min(900, int(jeda)))
        a = None
        for e in dict.fromkeys([eos, -5.0, -4.0]):  # ulangi bila senyap/terpotong/melantur
            a = _generate(teks, voice, e)
            if beta.layak(a, sr, teks):
                break
            ulang += 1
        potongan.append((beta.pangkas(a, sr), jeda))
    if not potongan:
        raise HTTPException(400, "naskah tidak berisi teks yang bisa diucapkan")
    return _wav(beta.sambung(potongan, sr)), len(potongan), ulang


def _voices():
    return sorted(f[:-4] for f in os.listdir(VOICE_DIR) if f.endswith(".wav"))


class Req(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    eos_threshold: float = DEFAULT_EOS


class NormReq(BaseModel):
    text: str


class Seg(BaseModel):
    teks: str
    jeda_ms: Optional[int] = None


class ScriptReq(BaseModel):
    text: str
    style: str = "percakapan"


class SegReq(BaseModel):
    segmen: List[Seg]
    voice: str = DEFAULT_VOICE
    eos_threshold: float = DEFAULT_EOS


class BetaReq(BaseModel):
    text: str
    style: str = "percakapan"
    voice: str = DEFAULT_VOICE
    eos_threshold: float = DEFAULT_EOS


@app.on_event("startup")
def warmup():
    with _lock:
        _model()


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/info")
def info():
    return {"service": "pocket-tts-indonesian", "default_voice": DEFAULT_VOICE,
            "voices": _voices(), "max_chars": MAX_CHARS,
            "max_chars_beta": MAX_CHARS_BETA, "ai_aktif": bool(beta.LR_KEY),
            "ai_model": beta.LR_MODEL if beta.LR_KEY else None,
            "gaya": list(beta.GAYA)}


@app.post("/normalize")
def norm(r: NormReq):
    return {"text": normalize(r.text)}


@app.get("/voices")
def voices():
    return {"voices": _voices()}


@app.get("/voices/{nama}/audio")
def voice_audio(nama: str):
    return FileResponse(_voice_path(nama), media_type="audio/wav")


@app.delete("/voices/{nama}")
def delete_voice(nama: str):
    os.remove(_voice_path(nama))
    return {"ok": True}


@app.post("/tts")
def tts_post(r: Req):
    return Response(_synth(r.text, r.voice, r.eos_threshold), media_type="audio/wav")


@app.get("/tts")
def tts_get(text: str, voice: str = DEFAULT_VOICE,
            eos_threshold: float = Query(DEFAULT_EOS)):
    return Response(_synth(text, voice, eos_threshold), media_type="audio/wav")


@app.post("/script")
def script(r: ScriptReq):
    if not r.text.strip():
        raise HTTPException(400, "teks kosong")
    if len(r.text) > MAX_CHARS_BETA:
        raise HTTPException(413, f"teks terlalu panjang (maks {MAX_CHARS_BETA} karakter)")
    return beta.rencana(r.text, r.style)


@app.post("/tts/script")
def tts_script(r: SegReq):
    audio, n, ulang = _synth_segmen(r.segmen, r.voice, r.eos_threshold)
    return Response(audio, media_type="audio/wav",
                    headers={"X-Segments": str(n), "X-Retries": str(ulang)})


@app.post("/tts/beta")
def tts_beta(r: BetaReq):
    p = script(ScriptReq(text=r.text, style=r.style))
    segmen = [Seg(**s) for s in p["segmen"]]
    audio, n, ulang = _synth_segmen(segmen, r.voice, r.eos_threshold)
    return Response(audio, media_type="audio/wav",
                    headers={"X-Segments": str(n), "X-Retries": str(ulang),
                             "X-Script-Source": p["sumber"]})


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
