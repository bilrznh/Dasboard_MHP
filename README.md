# Amazon Reviews — Sentiment Dashboard

Dashboard analisis sentimen VADER untuk dataset Amazon Fine Food Reviews.

---

## 📁 Struktur Proyek

```
amazon_sentiment_dashboard/
├── app.py               # File utama Streamlit
├── requirements.txt     # Daftar library yang dibutuhkan
└── README.md            # Panduan ini
```

---

## ⚙️ Cara Setup & Menjalankan

### 1. Buat Virtual Environment (disarankan)

```bash
python -m venv venv
```

Aktifkan virtual environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

---

## 📊 Dataset

Download dataset dari Kaggle:
**Amazon Fine Food Reviews**
→ https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

File yang dibutuhkan: `Reviews.csv`

Upload file tersebut langsung melalui tombol upload di dashboard.

---

## 🔍 Fitur Dashboard

- KPI Cards: total review, rata-rata rating, distribusi sentimen
- Distribusi label sentimen (bar + pie chart)
- Distribusi rating (1–5 bintang)
- Histogram & boxplot VADER compound score
- Heatmap crosstab Rating × Sentimen
- Tren sentimen per tahun
- Distribusi panjang review & helpfulness ratio
- Top 20 kata per sentimen
- Ringkasan statistik & insight utama

---

## 🛠️ Troubleshooting

**Error `ModuleNotFoundError`:**
Pastikan virtual environment sudah aktif dan `pip install -r requirements.txt` sudah dijalankan.

**Aplikasi lambat saat upload file besar:**
Dataset lengkap (~568k baris) membutuhkan waktu beberapa menit untuk proses VADER scoring. Hasil akan di-cache sehingga tidak perlu diproses ulang.
