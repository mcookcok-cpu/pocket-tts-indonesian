"""Server HTTP Pocket TTS Indonesian.

GET  /                  dashboard web
GET  /info              info + daftar suara
POST /normalize         pratinjau teks yang dieja
GET  /voices            daftar suara di voices/
POST /tts               JSON {"text","voice","eos_threshold"} -> audio/wav
GET  /tts?text=...&voice=klon   -> audio/wav
POST /script            (Beta) JSON {"text","style"} -> naskah utuh dari AI
POST /tts/script        (Beta) JSON {"segmen":[{"teks"}],"voice"} -> audio/wav (satu rekaman utuh)
POST /tts/beta          (Beta) satu langkah: {"text","style","voice"} -> audio/wav
POST /tts/script-eksp   (Eksperimental) seperti /tts/script, tapi EOS lebih konservatif, validasi durasi lebih ketat, singkatan tidak memecah kalimat
POST /voices/{nama}     unggah wav/mp3 sebagai suara baru (multipart: file)
DELETE /voices/{nama}   hapus suara
"""
import io
import os
import re
import subprocess
import sys
import threading
import zlib
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
MAX_CHARS = int(os.environ.get("MAX_CHARS", "10000"))
MAX_CHARS_BETA = int(os.environ.get("MAX_CHARS_BETA", "10000"))

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


def _generate(text: str, voice: str, eos: float, jeda: Optional[int] = None) -> np.ndarray:
    """Satu sintesis (pegang kunci). Mengembalikan float32 mono."""
    path = _voice_path(voice)
    with _lock:
        m = _model()
        m.eos_threshold = eos
        key = (voice, os.path.getmtime(path))
        if key not in _states:
            _states[key] = m.get_state_for_audio_prompt(path)
        return m.generate_audio(_states[key], text, frames_after_eos=jeda).numpy().astype(np.float32).reshape(-1)


def _ubah_tempo(a: np.ndarray, sr: int, tempo: float) -> np.ndarray:
    """Percepat/perlambat audio tanpa mengubah tinggi nada (ffmpeg atempo)."""
    if abs(tempo - 1.0) < 0.03:
        return a
    try:
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, sr, a)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "wav", "-i", "pipe:0",
             "-filter:a", f"atempo={tempo:.3f}", "-f", "wav", "-c:a", "pcm_f32le", "pipe:1"],
            input=buf.getvalue(), capture_output=True, timeout=30)
        if r.returncode != 0:
            return a
        _, out = scipy.io.wavfile.read(io.BytesIO(r.stdout))
        out = out.astype(np.float32).reshape(-1)
        return out if out.size else a
    except Exception:
        return a


def _wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    scipy.io.wavfile.write(buf, _model().sample_rate, pcm)
    return buf.getvalue()


def _synth(text: str, voice: str, eos: float) -> bytes:
    text = normalize(text)
    if not text:
        raise HTTPException(400, "teks kosong")
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"teks terlalu panjang (maks {MAX_CHARS} karakter)")
    return _wav(_generate(text, voice, eos))


def _synth_segmen(segmen, voice: str, eos: float, style: str = "percakapan"):
    """Beta: naskah dibacakan per kalimat utuh, bukan dipotong per frasa.

    Kalimat adalah satuan yang dilatihkan ke model (pocket_tts juga membagi di batas
    kalimat). Jeda setelah titik/tanda tanya dihasilkan model sendiri lewat
    frames_after_eos, jadi lebih panjang dari jeda koma (napas alami di dalam kalimat).
    Tidak ada pemangkasan antar kalimat dan tidak ada senyap sisipan. Tiap kalimat
    divalidasi dan diulang sendiri bila senyap, terpotong, atau melantur.
    """
    if not segmen:
        raise HTTPException(400, "naskah kosong")
    if len(segmen) > 1000:
        raise HTTPException(413, "terlalu banyak baris naskah (maks 1000)")
    sr = _model().sample_rate
    # (teks, tempo) per baris naskah; baris yang berakhir koma disambung ke baris berikutnya
    item = []
    for sg in segmen:
        t = beta.bersihkan(sg.teks)
        if not re.search(r"[A-Za-z]", t):
            continue
        if item and item[-1][0].endswith(","):
            item[-1] = (item[-1][0] + " " + t, item[-1][1])
        else:
            tp = 1.0 if sg.tempo is None else max(0.7, min(1.3, float(sg.tempo)))
            item.append((t, tp))
    if not item:
        raise HTTPException(400, "naskah tidak berisi teks yang bisa diucapkan")
    if sum(len(t) for t, _ in item) > MAX_CHARS_BETA * 2:
        raise HTTPException(413, f"naskah terlalu panjang (maks {MAX_CHARS_BETA * 2} karakter)")
    kalimat = []
    for t, tp in item:
        for k in re.split(r"(?<=[.?])\s+", t):
            if re.search(r"[A-Za-z]", k):
                kalimat.append((k, tp))
    dasar = beta.jeda_titik(style)
    potongan, ulang = [], 0
    for i, (k, tp) in enumerate(kalimat):
        # kalimat terakhir memakai jeda bawaan: ekor rekaman dipangkas juga
        jf = None if i == len(kalimat) - 1 else dasar + (1 if k.endswith("?") else 0)
        a = None
        for e in dict.fromkeys([eos, -5.0, -4.0]):
            a = _generate(k, voice, e, jf)
            if beta.layak(a, sr, k):
                break
            ulang += 1
        potongan.append(_ubah_tempo(a, sr, tp))
    return _wav(beta.rapikan(np.concatenate(potongan), sr)), len(kalimat), ulang


# ---------- Eksperimental (mode Beta yang diperketat; Beta asli tidak diubah) ----------
EKSP_EOS = -5.0                   # ambang EOS paling awal; -6.0 terlalu mudah memotong kalimat
EKSP_RATE_MIN = 0.05              # detik per huruf; Beta memakai 0.04 sehingga potongan ~30% lolos
EKSP_RATE_NORMAL = 0.065
_SINGKATAN = ("dr", "prof", "ir", "drs", "dra", "no", "hj", "kh", "bpk", "sdr", "jl",
              "tn", "ny", "yth", "vs", "mr", "mrs", "mgr", "kol", "jend", "kapt", "lt")
_RE_SINGKATAN = re.compile(r"\b(?:" + "|".join(_SINGKATAN) + r")\.", re.I)


def _pecah_kalimat(t: str):
    """Pecah di titik/tanda tanya, kecuali titik milik singkatan (Dr., No., Prof., dst.)."""
    t2 = _RE_SINGKATAN.sub(lambda m: m.group(0)[:-1] + "\x00", t)
    return [k.replace("\x00", ".") for k in re.split(r"(?<=[.?])\s+", t2)]


def _ujung_wajar(a: np.ndarray, sr: int, batas: float = 0.6) -> bool:
    """Deteksi kalimat/frasa yang berhenti mendadak: ucapan wajar mengecil di ujungnya (energi
    40 ms terakhir sekitar 0.15-0.4x median), sedangkan yang terpotong masih keras (0.7x ke atas)."""
    n = int(0.01 * sr)
    m = a.size // n
    if m < 8:
        return True
    fr = np.sqrt((a[:m * n].reshape(m, n) ** 2).mean(1))
    p = float(fr.max())
    if p <= 0:
        return False
    aktif = np.where(fr > 0.02 * float(np.max(np.abs(a))))[0]
    if aktif.size < 8:
        return False
    med = float(np.median(fr[aktif]))
    e = int(aktif[-1])
    return float(np.mean(fr[max(0, e - 3):e + 1])) / med <= batas


def _haluskan_ujung(a: np.ndarray, sr: int, ms: float = 40.0, batas: float = 0.4) -> np.ndarray:
    """Kalau ucapan masih keras tepat sebelum berhenti (EOS model memotong), turunkan volume
    40 ms terakhirnya dengan lengkung halus agar tidak terdengar seperti dipotong gunting."""
    n = int(0.01 * sr)
    m = a.size // n
    if m < 8:
        return a
    fr = np.sqrt((a[:m * n].reshape(m, n) ** 2).mean(1))
    aktif = np.where(fr > 0.02 * float(np.max(np.abs(a))))[0]
    if aktif.size < 8:
        return a
    med = float(np.median(fr[aktif]))
    e = int(aktif[-1])
    if float(np.mean(fr[max(0, e - 3):e + 1])) / med <= batas:
        return a
    akhir = min(a.size, (e + 1) * n)
    f = min(int(ms / 1000 * sr), akhir)
    a = a.copy()
    t = np.linspace(0, 1, f, dtype=np.float32)
    a[akhir - f:akhir] *= 0.5 * (1 + np.cos(np.pi * t))
    return a


def _atur_jeda(a: np.ndarray, sr: int, jeda_ms: float, aman_ms: float = 40.0) -> np.ndarray:
    """Buang senyap pembuka dan ujung hasil model, lalu sisakan jeda persis `jeda_ms` (+ pengaman 80 ms).
    Model dibiarkan menghasilkan beberapa frame setelah EOS supaya kata terakhir selesai dulu
    (dengan 2-3 frame, ~0.1-0.3 detik ujung kalimat hilang: "rumah" terdengar "rum")."""
    n = int(0.01 * sr)
    m = a.size // n
    if m < 4:
        return a
    fr = np.sqrt((a[:m * n].reshape(m, n) ** 2).mean(1))
    aktif = np.where(fr > 0.01 * float(np.max(np.abs(a))))[0]
    if aktif.size == 0:
        return a
    awal = max(0, int(aktif[0]) * n - int(0.04 * sr))   # senyap pembuka bawaan model dibuang
    target = (int(aktif[-1]) + 1) * n + int((aman_ms + jeda_ms) / 1000 * sr)
    if target <= a.size:
        return a[awal:target].copy()
    return np.concatenate([a[awal:], np.zeros(target - a.size, dtype=np.float32)])


def _wilayah_ucap(a: np.ndarray, sr: int):
    """(awal, akhir, daftar_jeda_dalam) dalam detik dari bagian yang benar-benar berbunyi."""
    n = int(0.01 * sr)
    m = a.size // n
    if m < 4:
        return None
    fr = np.sqrt((a[:m * n].reshape(m, n) ** 2).mean(1))
    ak = np.where(fr > 0.015 * float(np.max(np.abs(a))))[0]
    if ak.size < 4:
        return None
    jeda = (np.diff(ak) - 1) * 0.01
    return ak[0] * 0.01, (ak[-1] + 1) * 0.01, jeda[jeda > 0]


def _layak_eksp(a: np.ndarray, sr: int, teks: str) -> bool:
    """Laju bicara wajar (hanya bagian berbunyi, tanpa jeda ujung), dan tidak ada senyap janggal
    di tengah (>0.38 dtk) yang menandakan model tersendat."""
    if a.size == 0 or float(np.max(np.abs(a))) < 0.02:
        return False
    w = _wilayah_ucap(a, sr)
    if w is None:
        return False
    awal, akhir, jeda = w
    dur = akhir - awal
    huruf = max(1, beta._huruf(teks))
    if jeda.size and float(jeda.max()) > 0.38:
        return False
    return 0.048 * huruf <= dur <= 0.11 * huruf + 0.6


def _jeda_eksp(dasar: int, k: str, huruf: int, akhir_baris: bool, skala: float = 1.0) -> int:
    """Jeda setelah kalimat (1 frame ~ 80 ms). Tanya diberi ruang lebih, kalimat panjang butuh
    napas lebih panjang, kalimat pendek lebih lugas, ujung baris naskah seperti ganti paragraf.
    Variasi 0-1 frame ditentukan dari isi kalimat (tetap sama tiap kali, tapi tidak seragam)."""
    j = float(dasar)
    if k.endswith("?"):
        j += 1.5
    if huruf >= 70:
        j += 1
    elif huruf < 20:
        j -= 1
    if akhir_baris:
        j += 1
    j += (zlib.crc32(k.encode()) % 3) * 0.5
    return max(2, int(round(j * max(0.5, min(2.5, skala)))))


def _jeda_koma(huruf_frasa: int, skala: float = 1.0) -> int:
    """Jeda di koma: napas pendek, sedikit lebih lama setelah frasa panjang."""
    j = 2 + (1 if huruf_frasa >= 35 else 0)
    return max(1, int(round(j * max(0.5, min(2.5, skala)))))


def _pecah_frasa(k: str):
    """Pecah kalimat panjang di koma menjadi frasa (min 16 huruf per frasa). Koma tetap menempel
    di ujung frasa supaya model memberi intonasi menggantung, bukan intonasi akhir kalimat."""
    if "," not in k or beta._huruf(k) < 45:
        return [k]
    frasa, cur = [], ""
    for p in re.split(r"(?<=,)\s+", k):
        cur = (cur + " " + p).strip()
        if beta._huruf(cur) >= 16:
            frasa.append(cur)
            cur = ""
    if cur:
        if frasa:
            frasa[-1] = frasa[-1] + " " + cur
        else:
            frasa.append(cur)
    return frasa


def _rms_aktif(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    p = float(np.max(np.abs(a)))
    if p <= 0:
        return 0.0
    x = a[np.abs(a) > 0.1 * p]
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _samakan_keras(daftar):
    """Samakan kerasnya tiap kalimat ke median (dibatasi 0.7x-1.4x) supaya volume tidak naik-turun."""
    r = [_rms_aktif(a) for a in daftar]
    ok = [v for v in r if v > 0]
    if len(ok) < 2:
        return daftar
    target = float(np.median(ok))
    return [(a * float(np.clip(target / v, 0.7, 1.4))).astype(np.float32) if v > 0 else a
            for a, v in zip(daftar, r)]


def _halus_tepi(a: np.ndarray, sr: int, ms: float = 6.0) -> np.ndarray:
    """Fade sangat singkat di awal/akhir kalimat agar sambungan tidak berbunyi klik."""
    f = min(int(ms / 1000 * sr), a.size // 2)
    if f <= 0:
        return a
    a = a.copy()
    r = np.linspace(0, 1, f, dtype=np.float32)
    a[:f] *= r
    a[-f:] *= r[::-1]
    return a


def _synth_segmen_eksp(segmen, voice: str, eos: float, style: str = "percakapan",
                       jeda_skala: float = 1.0, pecah_koma: bool = False):
    if not segmen:
        raise HTTPException(400, "naskah kosong")
    if len(segmen) > 1000:
        raise HTTPException(413, "terlalu banyak baris naskah (maks 1000)")
    sr = _model().sample_rate
    item = []
    for sg in segmen:
        t = beta.bersihkan(sg.teks)
        if not re.search(r"[A-Za-z]", t):
            continue
        if item and item[-1][0].endswith(","):
            item[-1] = (item[-1][0] + " " + t, item[-1][1])
        else:
            tp = 1.0 if sg.tempo is None else max(0.7, min(1.3, float(sg.tempo)))
            item.append((t, tp))
    if not item:
        raise HTTPException(400, "naskah tidak berisi teks yang bisa diucapkan")
    if sum(len(t) for t, _ in item) > MAX_CHARS_BETA * 2:
        raise HTTPException(413, f"naskah terlalu panjang (maks {MAX_CHARS_BETA * 2} karakter)")
    kalimat = []
    for t, tp in item:
        bagian = [k for k in _pecah_kalimat(t) if re.search(r"[A-Za-z]", k)]
        for j, k in enumerate(bagian):
            kalimat.append((k, tp, j == len(bagian) - 1))
    dasar = beta.jeda_titik(style)
    e0 = max(eos, EKSP_EOS)
    hasil, stat = [], {"ulang": 0, "gabung": 0}

    def buat(teks: str, jf):
        """Satu potongan: ulangi dengan ambang EOS lebih longgar bila terlalu cepat/lambat,
        senyap, atau berhenti mendadak. Mengembalikan (audio, lolos_validasi)."""
        huruf = max(1, beta._huruf(teks))
        terbaik, selisih = None, None
        for e in (e0, -4.5, e0, -4.0, -4.5):
            # beri model cukup frame setelah EOS agar kata terakhir tuntas; jeda diatur sesudahnya
            frames_gen = 6 if jf is None else min(11, max(6, int(jf / 80) + 4))
            a = _atur_jeda(_generate(teks, voice, e, frames_gen), sr, 0 if jf is None else jf)   # jf = jeda dalam ms
            if _layak_eksp(a, sr, teks) and _ujung_wajar(a, sr):
                return a, True
            stat["ulang"] += 1
            if a.size:  # cadangan: percobaan yang lajunya paling mendekati ucapan normal
                d = abs(a.size / sr / huruf - EKSP_RATE_NORMAL)
                if selisih is None or d < selisih:
                    terbaik, selisih = a, d
        return terbaik, False

    for i, (k, tp, akhir_baris) in enumerate(kalimat):
        huruf_k = max(1, beta._huruf(k))
        terakhir = i == len(kalimat) - 1
        jf_kalimat = None if terakhir else _jeda_eksp(dasar, k, huruf_k, akhir_baris, jeda_skala) * 80 + 160
        frasa = _pecah_frasa(k) if pecah_koma else [k]
        potong = []
        if len(frasa) > 1:
            for m, fr in enumerate(frasa):
                jf = jf_kalimat if m == len(frasa) - 1 else _jeda_koma(beta._huruf(fr), jeda_skala) * 80
                a, ok = buat(fr, jf)
                potong.append(a)
                if not ok:  # frasa ini terpotong: batalkan pemecahan, bacakan kalimat utuh
                    stat["gabung"] += 1
                    potong = []
                    break
        if not potong:
            a, _ = buat(k, jf_kalimat)
            potong = [a]
        hasil.extend(_ubah_tempo(a, sr, tp) for a in potong)
    potongan = [_halus_tepi(_haluskan_ujung(a, sr), sr) for a in _samakan_keras(hasil)]
    return (_wav(beta.rapikan(np.concatenate(potongan), sr)), len(kalimat), stat["ulang"],
            len(hasil), stat["gabung"])


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
    tempo: Optional[float] = None   # 1.0 normal, <1 lebih lambat, >1 lebih cepat


class ScriptReq(BaseModel):
    text: str
    style: str = "percakapan"


class SegReq(BaseModel):
    segmen: List[Seg]
    style: str = "percakapan"
    voice: str = DEFAULT_VOICE
    eos_threshold: float = DEFAULT_EOS


class SegEkspReq(SegReq):
    jeda_skala: float = 1.0      # pengali semua jeda tanda baca (0.5 - 2.5)
    pecah_koma: bool = False     # kalimat panjang dipecah per frasa di koma agar jeda koma terdengar


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
    audio, n, ulang = _synth_segmen(r.segmen, r.voice, r.eos_threshold, r.style)
    return Response(audio, media_type="audio/wav",
                    headers={"X-Segments": str(n), "X-Retries": str(ulang)})


@app.post("/tts/script-eksp")
def tts_script_eksp(r: SegEkspReq):
    audio, n, ulang, nf, gb = _synth_segmen_eksp(r.segmen, r.voice, r.eos_threshold, r.style,
                                                 r.jeda_skala, r.pecah_koma)
    return Response(audio, media_type="audio/wav",
                    headers={"X-Segments": str(n), "X-Phrases": str(nf), "X-Fallback": str(gb), "X-Retries": str(ulang),
                             "X-Mode": "eksperimental"})


@app.post("/tts/beta")
def tts_beta(r: BetaReq):
    p = script(ScriptReq(text=r.text, style=r.style))
    segmen = [Seg(**s) for s in p["segmen"]]
    audio, n, ulang = _synth_segmen(segmen, r.voice, r.eos_threshold, r.style)
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
