import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_metrics(img_ref, img_warped):
    """Fungsi untuk menghitung metrik evaluasi (MSE & PSNR)"""
    mse = np.mean((img_ref.astype(float) - img_warped.astype(float))**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    return mse, psnr

def main():
    print("=== PIPELINE TRANSFORMASI GEOMETRIK ===")
    
    # 1. LOAD CITRA
    # Pastikan file gambar_lurus.jpeg dan gambar_miring.jpeg ada di folder yang sama
    img_ref = cv2.imread('gambar_lurus.jpeg', cv2.IMREAD_GRAYSCALE)
    img_miring = cv2.imread('gambar_miring.jpeg', cv2.IMREAD_GRAYSCALE)
    
    if img_ref is None or img_miring is None:
        print("Error: Gambar tidak ditemukan. Cek kembali nama filenya.")
        return

    # Resize agar ukurannya seragam
    TARGET_SIZE = (600, 800) # Lebar 600, Tinggi 800
    img_ref = cv2.resize(img_ref, TARGET_SIZE)
    img_miring = cv2.resize(img_miring, TARGET_SIZE)
    h, w = img_ref.shape

    # ==========================================
    # BAGIAN A: IMPLEMENTASI TRANSFORMASI DASAR
    # ==========================================
    print("A. Menjalankan Transformasi Dasar...")
    
    # a1. Translasi (Geser sumbu X=50, Y=30)
    M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
    img_trans = cv2.warpAffine(img_miring, M_trans, (w, h))
    
    # a2. Rotasi (Putar 45 derajat)
    M_rot = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
    img_rot = cv2.warpAffine(img_miring, M_rot, (w, h))
    
    # a3. Scaling (Perbesar 1.5x)
    img_scaled = cv2.resize(img_miring, None, fx=1.5, fy=1.5)
    
    # a4. Affine (Estimasi dengan 3 Titik bebas)
    pts1_aff = np.float32([[50,50], [200,50], [50,200]])
    pts2_aff = np.float32([[10,100], [200,50], [100,250]])
    M_aff = cv2.getAffineTransform(pts1_aff, pts2_aff)
    img_affine = cv2.warpAffine(img_miring, M_aff, (w, h))

    # ==========================================
    # BAGIAN B: TRANSFORMASI PERSPEKTIF & INTERPOLASI
    # ==========================================
    print("B. Mengevaluasi Transformasi Perspektif & Interpolasi...")
    
    # KOORDINAT MANUAL (HARDCODED)
    # Ganti angka-angka ini jika hasil tarikannya kurang pas dengan foto Anda
    # Urutan: [Kiri-Atas, Kanan-Atas, Kanan-Bawah, Kiri-Bawah]
    pts_miring = np.float32([
        [50, 100],   # Kiri Atas
        [550, 80],   # Kanan Atas
        [550, 750],  # Kanan Bawah
        [30, 720]    # Kiri Bawah
    ])
    
    # Koordinat tujuan (layar penuh lurus/sejajar)
    pts_lurus = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Hitung matriks transformasi perspektif
    M_persp = cv2.getPerspectiveTransform(pts_miring, pts_lurus)

    # Evaluasi metode interpolasi
    methods = [
        ('Nearest', cv2.INTER_NEAREST),
        ('Bilinear', cv2.INTER_LINEAR),
        ('Bicubic', cv2.INTER_CUBIC)
    ]

    results = []
    for name, flag in methods:
        start_time = time.time()
        # Aplikasikan transformasi perspektif beserta jenis interpolasinya
        img_warped = cv2.warpPerspective(img_miring, M_persp, (w, h), flags=flag)
        calc_time = (time.time() - start_time) * 1000 # dalam ms
        
        mse, psnr = calculate_metrics(img_ref, img_warped)
        results.append((name, img_warped, mse, psnr, calc_time))
        print(f"   [{name}] MSE: {mse:.2f} | PSNR: {psnr:.2f} dB | Waktu: {calc_time:.2f} ms")

    # ==========================================
    # VISUALISASI HASIL AKHIR
    # ==========================================
    # 1. Plot Transformasi Dasar
    fig1, axes1 = plt.subplots(1, 4, figsize=(15, 4))
    axes1[0].imshow(img_trans, cmap='gray'); axes1[0].set_title('Translasi')
    axes1[1].imshow(img_rot, cmap='gray'); axes1[1].set_title('Rotasi')
    axes1[2].imshow(img_scaled, cmap='gray'); axes1[2].set_title('Scaling')
    axes1[3].imshow(img_affine, cmap='gray'); axes1[3].set_title('Affine (3 Titik)')
    for ax in axes1: ax.axis('off')
    plt.suptitle('Transformasi Geometrik Dasar', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 2. Plot Perbandingan Perspektif & Interpolasi (5 Kolom)
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 5))
    
    # Gambar Lurus (Referensi)
    axes2[0].imshow(img_ref, cmap='gray')
    axes2[0].set_title('Target:\nDokumen Lurus (Referensi)')
    axes2[0].axis('off')

    # Gambar Miring (Input)
    axes2[1].imshow(img_miring, cmap='gray')
    axes2[1].set_title('Input:\nDokumen Miring')
    axes2[1].axis('off')
    
    # Hasil masing-masing interpolasi
    for i, (name, img_reg, mse, psnr, t) in enumerate(results):
        axes2[i+2].imshow(img_reg, cmap='gray')
        axes2[i+2].set_title(f'Perspektif ({name})\nMSE: {mse:.1f} | PSNR: {psnr:.1f}dB\nWaktu: {t:.2f} ms')
        axes2[i+2].axis('off')
        
    plt.suptitle('Transformasi Perspektif & Evaluasi Interpolasi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()