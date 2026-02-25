# Beban Capsa 🃏

> *"Jangan biarkan tongkrongan asyikmu terganggu karena kamu jadi BEBAN"*

Aplikasi deteksi kartu capsa berbasis **YOLOv12** yang menganalisis kombinasi kartu di tanganmu secara otomatis.

**Author:** Uqi Shafi

---
[Streamlit](https://bebancapsa.streamlit.app/)
## Fitur

- **Deteksi kartu otomatis** dari foto menggunakan YOLOv12
- **Urutan kartu** dari terkuat ke terlemah
- **Deteksi kombinasi:** Royal Flush, Straight Flush, Four of a Kind (Bomb), Full House, Flush, Straight, Tris, Pair
- **Koreksi manual:** tambah kartu yang terlewat atau hapus kartu yang salah terdeteksi
- **Sample image** untuk mencoba aplikasi tanpa perlu foto sendiri

---

## Cara Menggunakan Aplikasi

```
poetry run streamlit run src/bebancapsa/main.py
```

**Upload foto:**
1. Klik area upload atau drag & drop foto kartu
2. Atur slider **Confidence** — turunkan jika kartu tidak terdeteksi
3. Klik tombol **Deteksi Kartu**

**Sample image:**
- Klik **"Gunakan foto ini"** di bawah salah satu sample untuk langsung mencoba tanpa upload

**Koreksi deteksi:**
- Klik tombol **−** di bawah kartu untuk menghapus kartu yang salah terdeteksi
- Klik **+ Tambah Kartu** lalu pilih kartu yang terlewat

**Membaca hasil:**
- Kartu ditampilkan urut dari terkuat ke terlemah
- Kombinasi ditampilkan lengkap dengan keterangan
- **BOMB** ditandai warna merah — simpan untuk situasi kritis

---

## Aturan Capsa

### Urutan Nilai (tertinggi ke terendah)

```
2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3
```

### Urutan Simbol (pada nilai yang sama)

```
♠ Spade > ♥ Heart > ♣ Club > ♦ Diamond
```

> Kartu tertinggi: **2 Spade** | Kartu terendah: **3 Diamond**

### Urutan Kombinasi

| Kombinasi | Keterangan |
|---|---|
| Royal Flush | A K Q J 10 satu simbol — **BOMB** |
| Straight Flush | 5 kartu berurutan satu simbol — **BOMB** |
| Four of a Kind | 4 kartu nilai sama — **BOMB** |
| Full House | 3 kartu + 2 kartu nilai sama |
| Flush | 5 kartu simbol sama |
| Straight | 5 kartu berurutan |
| Three of a Kind | 3 kartu nilai sama |
| Pair | 2 kartu nilai sama |
| High Card | Kartu tunggal |

**BOMB** mengalahkan kombinasi apapun, termasuk sesama Bomb yang lebih rendah.

---

