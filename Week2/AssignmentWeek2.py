import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- FUNGSI UTILITAS ---

def get_image_size(image):
    """Menghitung ukuran memori array gambar dalam Kilobytes (KB)"""
    return image.nbytes / 1024

def uniform_quantization(image, levels):
    """Kuantisasi Uniform: Membagi rentang warna secara merata"""
    factor = 256 / levels
    quantized = (image // factor) * factor
    return quantized.astype(np.uint8)

def nonuniform_quantization(image, k):
    """Kuantisasi Non-Uniform menggunakan K-Means Clustering"""
    # Reshape ke array 2D pixel
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Kriteria penghentian algoritma (maks iterasi 10 atau epsilon 1.0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Penerapan K-Means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Konversi kembali ke uint8 dan reshape ke dimensi asli
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    quantized_image = res.reshape(image.shape)
    return quantized_image

# --- MAIN PROCESSING ---

# Ganti dengan path gambar Anda
image_paths = ['terang.png', 'normal.png', 'bayang.jpg'] 
titles = ['Normal', 'Redup (Dim)', 'Terang (Bright)']

print(f"{'Metrik':<20} | {'Waktu (ms)':<10} | {'Size Awal (KB)':<15} | {'Size Akhir (KB)':<15} | {'Rasio Kompresi':<15}")
print("-" * 90)

for idx, img_path in enumerate(image_paths):
    # 1. Load Image
    original_bgr = cv2.imread(img_path)
    if original_bgr is None:
        print(f"Error: Gambar {img_path} tidak ditemukan.")
        continue
    
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Konversi Ruang Warna
    start_time = time.time()
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    conversion_time = (time.time() - start_time) * 1000
    
    # 3. Kuantisasi (Contoh pada citra RGB)
    # Uniform (16 Level)
    start_q_uni = time.time()
    rgb_uniform = uniform_quantization(original_rgb, 16)
    time_uni = (time.time() - start_q_uni) * 1000
    
    # Non-Uniform (K=8 Colors)
    start_q_non = time.time()
    rgb_kmeans = nonuniform_quantization(original_rgb, 8)
    time_non = (time.time() - start_q_non) * 1000
    
    # 4. Hitung Metrik
    size_orig = get_image_size(original_rgb)
    # Simulasi ukuran file terkompresi (perkiraan nbytes raw) - dalam praktik kompresi file bergantung format (JPG/PNG)
    # Di sini kita membandingkan footprint memori mentah jika bit-depth dikurangi
    size_quant = get_image_size(rgb_uniform) # Ukuran array sama, tapi entropi berkurang
    
    print(f"Img: {titles[idx]}")
    print(f"{'  Konversi HSV':<20} | {conversion_time:.4f}     | {'-':<15} | {'-':<15} | {'-':<15}")
    print(f"{'  Kuantisasi Uni':<20} | {time_uni:.4f}     | {size_orig:.2f}          | {size_quant:.2f}* | {'1:1 (Raw)'}")
    print(f"{'  Kuantisasi K-Means':<20} | {time_non:.4f}     | {size_orig:.2f}          | {size_quant:.2f}* | {'1:1 (Raw)'}")

    # 5. Visualisasi & Histogram
    plt.figure(figsize=(15, 8))
    
    # Baris 1: Citra Asli & Hasil Konversi
    plt.subplot(2, 4, 1); plt.imshow(original_rgb); plt.title(f"Asli ({titles[idx]})")
    plt.subplot(2, 4, 2); plt.imshow(gray, cmap='gray'); plt.title("Grayscale")
    plt.subplot(2, 4, 3); plt.imshow(hsv); plt.title("HSV Visualization")
    plt.subplot(2, 4, 4); plt.imshow(lab); plt.title("LAB Visualization")

    # Baris 2: Hasil Kuantisasi & Histogram
    plt.subplot(2, 4, 5); plt.imshow(rgb_uniform); plt.title("Uniform Quant (16 lvl)")
    plt.subplot(2, 4, 6); plt.hist(original_rgb.ravel(), 256, [0, 256], color='r', alpha=0.5, label='Asli'); 
    plt.hist(rgb_uniform.ravel(), 256, [0, 256], color='b', alpha=0.5, label='Uni'); plt.legend(); plt.title("Hist: Asli vs Uniform")
    
    plt.subplot(2, 4, 7); plt.imshow(rgb_kmeans); plt.title("K-Means Quant (K=8)")
    plt.subplot(2, 4, 8); plt.imshow(cv2.Canny(rgb_kmeans, 100, 200), cmap='gray'); plt.title("Deteksi Tepi (Canny) pada K-Means")

    plt.tight_layout()
    plt.show()

print("*Catatan: Ukuran memori raw NumPy array tetap sama (uint8), namun entropi informasi berkurang drastis.")