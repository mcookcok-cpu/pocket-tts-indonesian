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
