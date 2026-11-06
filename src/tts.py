#!/usr/bin/env python3
"""CLI Pocket TTS Indonesian.

Contoh:
  uv run python src/tts.py "Halo, ada 15 orang di sini." -v voices/x.wav -o output/halo.wav
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize  # noqa: E402

# Revisi dipin sesuai README model
CONFIG = ("hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml"
          "@17257664e384561c957b02ac92edd1a24807f0e5")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("teks")
    p.add_argument("-v", "--voice", help="wav prompt suara (kosong = suara bawaan Inggris)")
    p.add_argument("-o", "--output", default="output/out.wav")
    p.add_argument("--eos-threshold", type=float, default=-6.0,
                   help="-6.0 terbaik di eval README; coba -5.0 jika hasil kosong")
    args = p.parse_args()

    import scipy.io.wavfile
    from pocket_tts import TTSModel

    teks = normalize(args.teks)
    print(f"Teks: {teks}")
    model = TTSModel.load_model(config=CONFIG, eos_threshold=args.eos_threshold)
    voice = args.voice or "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    state = model.get_state_for_audio_prompt(voice)
    audio = model.generate_audio(state, teks)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scipy.io.wavfile.write(args.output, model.sample_rate, audio.numpy())
    print(f"Selesai: {args.output}")


if __name__ == "__main__":
    main()
