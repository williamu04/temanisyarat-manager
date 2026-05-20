# Analisis Ekstraksi Landmark dan Rekomendasi Perekaman

## Temuan Utama

Ekstraksi landmark dari 1600 video (8 signer, 20 gesture) menunjukkan tingkat kegagalan yang tinggi pada deteksi face:

| Landmark | Dimensi | Rata-rata NaN |
|----------|---------|:------------:|
| Pose     | 27      | **0.1%**     |
| Face     | 99      | **84.5%**    |
| Hands    | 126     | **29.8%**    |

Face gagal terdeteksi di sebagian besar frame, bahkan dengan confidence threshold minimal (0.05) dan preprocessing CLAHE.

### Pola Kegagalan Face

| Signer   | Face NaN | Hands NaN |
|----------|:--------:|:---------:|
| farras   | 44.2%    | 14.4%     |
| willi    | 64.4%    | 7.7%      |
| ian      | 74.8%    | 8.3%      |
| hani     | 98.3%    | 49.6%     |
| mutia    | 97.3%    | 67.8%     |
| fredi    | 97.4%    | 9.9%      |
| saidah   | 99.8%    | 60.6%     |
| ivan     | 100.0%   | 20.2%     |

Pola ini menunjukkan bahwa akar masalah bukan pada pipeline ekstraksi atau threshold model, melainkan pada **kualitas perekaman**: signer tidak menghadap kamera, pencahayaan buruk, atau wajah tidak terlihat.

### Dampak pada Akurasi Model

Pipeline model sebelumnya menggunakan 252 dimensi input (pose + face + hands). Dengan 84.5% face dan 29.8% hands NaN, model hanya belajar dari ~27 dimensi data bersih. ~~Hasil akhir: akurasi ~10-16% pada test set (setara random untuk 20 kelas).~~

## Keputusan

1. **Drop face dari pipeline** — face tidak diekstrak lagi di `extractor.py` dan tidak digunakan di model. Input dimensi turun dari 252 ke 153 (pose 27 + hands 126).
2. **Pipeline model sudah diperbaiki** — arsitektur GRU 1 layer (96 hidden), label smoothing, ReduceLROnPlateau, gradient clipping.

## Rekomendasi Perekaman ke Depan

### Posisi Kamera
- **Kamera setara mata** — tidak dari atas/bawah agar wajah terekam frontal
- **Frame upper body** — kepala + kedua tangan + dada harus selalu dalam frame
- Resolusi 720p sudah cukup baik — pertahankan

### Pencahayaan
- **Cahaya depan (frontal)** — hindari backlight (jendela di belakang signer)
- Gunakan **diffuse light** — hindari bayangan keras di wajah
- Standar: mean brightness ~150-200 dengan std dev >40 (cek dengan histogram)

### Instruksi untuk Signer
- **Hadap kamera** — jangan melihat ke tangan saat memberi gesture
- **Tangan di frame** — jangan gesture terlalu rendah atau ke samping
- **Pakaian kontras** dengan background — hindari warna kulit/skin tone di background untuk deteksi hands

### Sesi Perekaman
- 1 signer per sesi — tidak boros biaya transport/harian
- Rekam per gesture dalam 1 take kontinu — lebih mudah daripada per video pendek
- Validasi cepat: minta signer review 2-3 video untuk memastikan landmark terdeteksi

### QC Checklist (Sebelum Simpan)
- [ ] Pose terdeteksi di semua frame (<1% NaN)
- [ ] Wajah terlihat jelas di >80% frame
- [ ] Kedua tangan terdeteksi di >90% frame
- [ ] Background tidak berantakan (deteksi tangan lebih baik dengan BG polos)
