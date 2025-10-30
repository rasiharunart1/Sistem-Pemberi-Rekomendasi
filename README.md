# Advanced Phone Recommendation System

Sistem rekomendasi ponsel pintar dengan filtering brand berbasis TF-IDF dan agregasi statistik.

## Fitur Utama

1. **TF-IDF + String Matching Brand Filtering**  
   - Deteksi brand dari input pengguna menggunakan TF-IDF dan pencocokan string.
   - Menampilkan hanya ponsel dari brand yang diminta, lebih akurat dari pencarian biasa.

2. **Statistical Aggregation (Median/Mean/Mode)**  
   - Hitung agregasi statistik dari fitur utama (kamera, baterai, layar, audio, selfie) untuk setiap brand.
   - Bisa memilih metode median, mean, atau mode sebagai profil virtual.

3. **Virtual Profile Generation**
   - Sistem membuat profil ponsel virtual dari statistik grup, lalu mencari ponsel real paling mirip dengan profil ini.

4. **Cosine Similarity Recommendation**
   - Rekomendasi didasarkan pada kemiripan fitur (cosine similarity) dengan profil acuan (real atau virtual).

5. **Dynamic Recommendation Control**
   - Pengguna bisa mengatur jumlah rekomendasi yang ingin ditampilkan.

6. **Multi-Tab Gradio Interface**
   - Tab rekomendasi berdasarkan brand, mencari ponsel serupa, pencarian nama/model, dan tab informasi.

## Cara Pakai

1. Jalankan script `advanced_phone_recommender.py`.
2. Gradio akan otomatis terbuka di browser.
3. Pilih tab "Brand-Based Recommendations" untuk merekomendasikan ponsel berdasarkan brand/statistik grup.
4. Masukkan nama brand (misal: `apple`, `samsung`, `xiaomi`, dsb.) atau nama model spesifik (misal: `iPhone 14 Pro Max`).
5. Pilih metode statistik (median, mean, mode) untuk virtual profile.
6. Atur jumlah rekomendasi, lalu tekan **Generate Recommendations**.
7. Hasil rekomendasi dan tabel detail akan muncul di sisi kanan.

## Penjelasan Algoritma (Singkat)

- **Brand Filtering:**  
  Kombinasi TF-IDF vectorization dan pencocokan string memastikan hanya ponsel dari brand yang diminta ditampilkan.

- **Statistical Aggregation:**  
  Fitur utama dianalisis secara statistik (median, rata-rata, atau modus) untuk membentuk "profil virtual" brand.

- **Cosine Similarity:**  
  Kemiripan fitur ponsel dengan profil acuan dihitung secara matematis, lalu diurutkan untuk rekomendasi.

- **StandardScaler:**  
  Semua fitur dinormalisasi (Z-score) agar perbandingan adil.

## Formula Matematika

**Z-Score Normalization:**
```
z = (x - μ) / σ
```
**Cosine Similarity:**
```
similarity = (A · B) / (||A|| * ||B||)
```

## Dataset

- Dataset diunduh otomatis dari Kaggle `prasertk/mobile-phone-rating`.

## Dependencies

- gradio
- kagglehub
- pandas, numpy, scikit-learn, plotly, seaborn, scipy, matplotlib

## Author

- **rasiharunart1**
- **Tanggal Update:** 2025-10-30

---

Sistem ini didesain untuk riset, analisis, dan kebutuhan rekomendasi ponsel berbasis fitur utama.  
Jika ada pertanyaan, silakan hubungi developer via GitHub.
