# Beban Capsa 🃏

> *"Jangan biarkan tongkrongan asyikmu terganggu karena kamu jadi BEBAN"*

Aplikasi deteksi kartu capsa berbasis **YOLOv12** yang menganalisis kombinasi kartu di tanganmu secara otomatis.

**Author:** Uqi Shafi

---

## Fitur

- **Deteksi kartu otomatis** dari foto menggunakan YOLOv12
- **Urutan kartu** dari terkuat ke terlemah (2 tertinggi, 3 terendah)
- **Deteksi kombinasi:** Royal Flush, Straight Flush, Four of a Kind (Bomb), Full House, Flush, Straight, Tris, Pair
- **Koreksi manual:** tambah kartu yang terlewat atau hapus kartu yang salah terdeteksi
- **Sample image** untuk mencoba aplikasi tanpa perlu foto sendiri
- Deploy di Streamlit Community Cloud

---

## Struktur Folder

```
BebanCapsa/
├── models/
│   ├── .gitkeep
│   └── best_capsa.pt        # model YOLOv12 (tidak di-commit ke GitHub)
├── sample/
│   ├── hand1.jpg            # sample foto kartu untuk demo
│   ├── hand2.jpg
│   └── hand3.jpg
├── src/capsa/
│   ├── __init__.py
│   └── main.py              # aplikasi Streamlit
├── training_code/
│   └── Train_YOLOv12_Capsa.ipynb
├── pyproject.toml
├── poetry.lock
├── .gitignore
└── README.md
```

---

## Cara Menjalankan Secara Lokal

### Prasyarat

- Python 3.12
- Poetry 2.x
- NVIDIA GPU (opsional, untuk inferensi lebih cepat)

### Langkah-langkah

**1. Clone repository**

```bash
git clone https://github.com/uqishafi-cloud/BebanCapsa.git
cd BebanCapsa
```

**2. Install dependencies via Poetry**

```bash
poetry env use python3.12
poetry install
```

**3. Install PyTorch GPU** (skip jika tidak punya GPU NVIDIA)

```bash
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**4. Download model**

Unduh file `best_capsa.pt` dari link berikut dan taruh di folder `models/`:

```
[Link Google Drive model]
```

Atau jalankan script download otomatis:

```bash
poetry run python download_model.py
```

**5. Jalankan aplikasi**

```bash
poetry run streamlit run src/capsa/main.py
```

Buka browser di `http://localhost:8501`

---

## Cara Menggunakan Aplikasi

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

## Dataset & Training

- Dataset: kartu remi standar 52 kartu, dianotasi via Roboflow
- Format label: YOLOv12 (YOLO bounding box)
- Model base: `yolov12s.pt`
- Training environment: NVIDIA RTX 4060 (lokal) / Google Colab T4

Notebook training tersedia di `training_code/Train_YOLOv12_Capsa.ipynb`

---

## Dependencies

```
streamlit >= 1.39.0
ultralytics >= 8.3.0
opencv-python-headless >= 4.8.0
pillow >= 9.0.0
numpy >= 1.24.0, < 2.0.0
pandas >= 1.4.0, < 3.0.0
pyyaml >= 6.0
torch (GPU: cu124)
```

---

## Deploy ke Streamlit Community Cloud

1. Push semua file ke GitHub (tanpa `models/*.pt`)
2. Upload model ke Google Drive, set sharing ke **Anyone with the link**
3. Buka [share.streamlit.io](https://share.streamlit.io)
4. Klik **New app** → hubungkan repo GitHub
5. Set **Main file path:** `src/capsa/main.py`
6. Klik **Deploy**

---

## Lisensi

MIT License — bebas digunakan untuk keperluan edukasi.
