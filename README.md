# 📘 Judul Proyek
*(Isi judul proyek Anda di sini)*

## 👤 Informasi
- **Nama:** ALFINDO OKTAVIAN RAMADHAN  
- **Repo:**   
- **Video:** https://drive.google.com/file/d/105VQAco8SYl-759nEuwwcWuo9Vyru4pm/view?usp=sharing 

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi dan menentukan model terbaik  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
1.	Penentuan tingkat pendapatan individu secara manual berdasarkan faktor demografis dan pekerjaan sering bersifat subjektif dan tidak konsisten, sehingga diperlukan model klasifikasi yang mampu memprediksi tingkat pendapatan secara akurat dan objektif.
2.	Dataset Adult UCI memiliki kombinasi fitur numerik dan kategorikal dengan karakteristik yang beragam serta adanya ketidakseimbangan kelas, sehingga membutuhkan proses preprocessing yang tepat agar model dapat mempelajari pola data secara optimal.
3.	Belum diketahui faktor-faktor apa saja yang paling berpengaruh dalam menentukan apakah pendapatan individu berada di atas atau di bawah batas tertentu (>$50K atau ≤$50K), sehingga diperlukan analisis feature importance untuk memahami faktor penentu tingkat pendapatan.
4.	Diperlukan perbandingan performa antara model baseline, model machine learning lanjutan, dan model deep learning untuk menentukan pendekatan terbaik dalam melakukan klasifikasi pendapatan individu.
 

**Goals:**  
1.	Mengembangkan tiga model klasifikasi yang terdiri dari model baseline, model machine learning lanjutan, dan model deep learning untuk memprediksi tingkat pendapatan individu menggunakan dataset Adult UCI.
2.	Mencapai akurasi klasifikasi minimal ≥ 80% pada model terbaik sebagai indikator keberhasilan prediksi tingkat pendapatan.
3.	Melakukan preprocessing data secara tepat, termasuk encoding fitur kategorikal, penanganan data tidak seimbang, serta pembagian data train dan validation agar model dapat belajar secara optimal.
4.	Mengevaluasi dan membandingkan performa ketiga model menggunakan metrik evaluasi seperti accuracy, precision, recall, dan F1-score untuk menentukan model yang paling efektif.
5.	Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap tingkat pendapatan individu melalui analisis feature importance pada model machine learning lanjutan.


---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** https://archive.ics.uci.edu/dataset/2/adult  
- **Jumlah Data:** 48.842 baris
- **Tipe:** Tabular (numerik dan categorical)

### Fitur Utama
| Fitur           | Deskripsi                                                                 |
|-----------------|--------------------------------------------------------------------------|
| age             | Umur individu (numerik kontinu)                                          |
| workclass       | Kelas pekerjaan (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked) |
| fnlwgt          | Final weight, bobot individu di populasi (numerik kontinu)               |
| education       | Tingkat pendidikan (Bachelors, HS-grad, Some-college, Assoc-acdm, Assoc-voc, Masters, etc.) |
| education-num   | Representasi numerik dari tingkat pendidikan (numerik kontinu)           |
| marital-status  | Status pernikahan (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, etc.) |
| occupation      | Pekerjaan individu (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, etc.) |
| relationship    | Hubungan keluarga (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried) |
| race            | Ras individu (White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other) |
| sex             | Jenis kelamin (Male, Female)                                            |
| capital-gain    | Keuntungan modal (numerik kontinu)                                       |
| capital-loss    | Kerugian modal (numerik kontinu)                                         |
| hours-per-week  | Jumlah jam kerja per minggu (numerik kontinu)                             |
| native-country  | Negara asal individu (United-States, Mexico, etc.)                        |
| income          | Target / label (<=50K atau >50K, klasifikasi biner)                     |


---

# 4. 🔧 Data Preparation
- Cleaning (missing/duplicate/outliers)  
- Transformasi (encoding/scaling)  
- Splitting (train/val/test)  

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** [...]  
- **Model 2 – Advanced ML:** [...]  
- **Model 3 – Deep Learning:** [...]  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)

### Hasil Singkat
| Model | Score | Catatan |
|-------|--------|---------|
| Baseline | [...] | |
| Advanced | [...] | |
| Deep Learning | [...] | |

---

# 7. 🏁 Kesimpulan
- Model terbaik: [...]  
- Alasan: [...]  
- Insight penting: [...]  

---

# 8. 🔮 Future Work
- [ ] Tambah data  
- [ ] Tuning model  
- [ ] Coba arsitektur DL lain  
- [ ] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment:
