"""Normalisasi teks Indonesia untuk Pocket TTS.

Model dilatih dengan transkrip yang menulis angka sebagai kata, jadi digit
harus dieja. Tanda '!' tidak ada di kosakata -> diganti '.'.
"""
import re

_SATUAN = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
           "delapan", "sembilan", "sepuluh", "sebelas"]


def terbilang(n: int) -> str:
    if n < 0:
        return "minus " + terbilang(-n)
    if n < 12:
        return _SATUAN[n]
    if n < 20:
        return _SATUAN[n - 10] + " belas"
    if n < 100:
        s, r = divmod(n, 10)
        return _SATUAN[s] + " puluh" + (" " + _SATUAN[r] if r else "")
    if n < 200:
        return "seratus" + (" " + terbilang(n - 100) if n > 100 else "")
    if n < 1000:
        s, r = divmod(n, 100)
        return _SATUAN[s] + " ratus" + (" " + terbilang(r) if r else "")
    if n < 2000:
        return "seribu" + (" " + terbilang(n - 1000) if n > 1000 else "")
    if n < 1_000_000:
        s, r = divmod(n, 1000)
        return terbilang(s) + " ribu" + (" " + terbilang(r) if r else "")
    if n < 1_000_000_000:
        s, r = divmod(n, 1_000_000)
        return terbilang(s) + " juta" + (" " + terbilang(r) if r else "")
    if n < 1_000_000_000_000:
        s, r = divmod(n, 1_000_000_000)
        return terbilang(s) + " miliar" + (" " + terbilang(r) if r else "")
    s, r = divmod(n, 1_000_000_000_000)
    return terbilang(s) + " triliun" + (" " + terbilang(r) if r else "")


def _angka(m: re.Match) -> str:
    teks = m.group(0)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", teks):  # 1.500.000
        return terbilang(int(teks.replace(".", "")))
    if "," in teks:  # 3,14
        bulat, desimal = teks.split(",", 1)
        return terbilang(int(bulat)) + " koma " + " ".join(_SATUAN[int(d)] for d in desimal)
    return terbilang(int(teks))


def normalize(teks: str) -> str:
    teks = re.sub(r"(?<=\d)%", " persen", teks)
    teks = re.sub(r"\d{1,3}(?:\.\d{3})+|\d+,\d+|\d+", _angka, teks)
    teks = teks.replace("!", ".")
    return re.sub(r"\s+", " ", teks).strip()


if __name__ == "__main__":
    import sys
    print(normalize(" ".join(sys.argv[1:])))
