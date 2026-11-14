# pocket-tts-indonesian

TTS bahasa Indonesia + voice cloning, memakai model
[anak10thn/pocket-tts-indonesian](https://huggingface.co/anak10thn/pocket-tts-indonesian)
(basis kyutai/pocket-tts, dilatih 502 jam data LEMAS). Jalan di CPU.

## Pakai
```bash
uv run python src/tts.py "Halo, ada 15 pesanan baru." -v voices/suara1.wav -o output/halo.wav
```
- Model: varian 6-layer (438 MB), revisi dipin di `src/tts.py`. Unduhan pertama otomatis dari HF.
- `--eos-threshold` default -6.0 (terbaik di eval). Jika hasil kosong, coba -5.0.
- `voices/suara1-4.wav`: prompt suara CC0 (Common Voice), 48 kHz. Bisa ganti dengan rekaman sendiri;
  rekaman 48 kHz yang tenang menghasilkan kloning lebih jelas.

## Catatan model (dari README HF)
- Angka harus dieja -> `src/normalize.py` melakukannya otomatis (juga `%` dan `!`).
- Teks panjang: kadang satu item daftar terlewat; pecah jadi kalimat terpisah.
- Kualitas dibatasi data 16 kHz (UTMOS 2.68), setara telepon/YouTube.
- Lisensi CC-BY-4.0.

## Server dan dashboard
`pm2 start ecosystem.config.js` -> http://HOST:6001/ (port 6000 diblokir Chrome/Brave: ERR_UNSAFE_PORT).

## Mode Beta AI
Model ini belajar intonasi dari tanda baca, jadi AI bekerja di sisi naskah (`src/beta.py`):
LLM (LRouter, kunci dibaca dari .env proyek Mao saat pm2 start) mengeja angka/singkatan dan
mengatur koma/titik/tanda tanya menjadi SATU naskah utuh. Naskah dibacakan per kalimat utuh
(bukan per frasa; cara lama itu membuat suara putus-putus), tanpa pemangkasan antar kalimat dan tanpa
senyap sisipan. Jeda setelah titik/tanda tanya dihasilkan model lewat `frames_after_eos` (`JEDA_TITIK`
di `src/beta.py`, per gaya), jadi selalu lebih panjang dari jeda koma yang napas alami di dalam kalimat.
Tiap kalimat divalidasi dan diulang sendiri bila senyap/terpotong/melantur (eos -5.0 lalu -4.0).
Jika AI gagal, memakai aturan dasar (tanpa AI).
Endpoint: `POST /script`, `POST /tts/script`, `POST /tts/beta`.

### Gaya bicara (menjiwai)
Pilihan gaya: percakapan, berita, ekspresif, dongeng, desa, puitis, dramatis. Gaya mengubah dua hal:
1. Naskah: LLM mengatur koma/titik/tanda tanya sesuai irama gaya (kata asli tidak diubah).
2. Panjang jeda setelah titik/tanda tanya (`JEDA_TITIK` di `src/beta.py`).
Suhu sampling model sudah diuji (0,3 sampai 0,9) dan tidak mengubah variasi nada berarti, jadi tidak dipakai.
Pengaruh terbesar pada nada tetap dari rekaman suara acuan: untuk membaca cerita lebih hidup,
unggah rekaman 10 sampai 20 detik orang yang membaca dengan ekspresif.

### Gaya Penutur desa (tempo mengikuti cerita)
AI memberi tiap kalimat label tempo (sangat_lambat 0,74 / lambat 0,86 / normal / cepat 1,14 / sangat_cepat 1,27)
sesuai suasana cerita: hening dan haru pelan, aksi dan ketegangan cepat. Tiap kalimat disintesis lalu
diperlambat atau dipercepat tanpa mengubah tinggi nada (ffmpeg `atempo`, `_ubah_tempo` di `src/server.py`);
jeda setelah kalimat ikut melambat atau mempercepat. Field `tempo` (0,7 sampai 1,3) ada di tiap segmen
`/script` dan `/tts/script`; di dashboard, tempo dipertahankan saat naskah diedit selama jumlah baris tetap.
Mode tanpa AI memakai tempo normal.
