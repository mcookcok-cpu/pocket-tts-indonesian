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
LLM (LRouter, kunci dibaca dari .env proyek Mao saat pm2 start) mengeja angka/singkatan,
memecah jadi segmen pendek, mengatur koma/titik/tanda tanya dan jeda per segmen.
Tiap segmen disintesis, divalidasi (senyap/terpotong/melantur -> diulang dengan eos -5.0 lalu -4.0),
dipangkas, disambung dengan jeda, dan dinormalkan volumenya. Jika AI gagal, memakai aturan dasar.
Endpoint: `POST /script`, `POST /tts/script`, `POST /tts/beta`.
