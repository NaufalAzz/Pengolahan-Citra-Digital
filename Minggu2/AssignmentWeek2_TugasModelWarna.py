import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd # Opsional, hanya untuk tampilan tabel rapi di terminal

# --- FUNGSI ---
def get_image_size_kb(image):
    return image.nbytes / 1024

def uniform_quantization(image, levels):
    factor = 256 / levels
    quantized = (image // factor) * factor
    return quantized.astype(np.uint8)

def nonuniform_quantization(image, k):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(image.shape)

# --- SETUP DATA ---
filenames = ['normal.png', 'terang.png', 'bayang.png']
results = []

print("=== MEMPROSES 3 CITRA (MOHON TUNGGU...) ===")

# --- 1. LOOPING UNTUK MENGHITUNG METRIK (SEMUA GAMBAR) ---
for f in filenames:
    img_bgr = cv2.imread(f)
    if img_bgr is None:
        print(f"ERROR: File {f} tidak ditemukan!")
        continue
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Hitung Waktu Konversi HSV
    start = time.time()
    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    t_hsv = (time.time() - start) * 1000
    
    # Hitung Waktu & Size Uniform
    start = time.time()
    img_uni = uniform_quantization(img_rgb, 16)
    t_uni = (time.time() - start) * 1000
    size_uni = get_image_size_kb(img_uni)
    
    # Hitung Waktu K-Means (Non-Uniform)
    start = time.time()
    img_kmeans = nonuniform_quantization(img_rgb, 8)
    t_kmeans = (time.time() - start) * 1000
    
    results.append({
        "File": f,
        "Waktu HSV (ms)": round(t_hsv, 2),
        "Waktu Uniform (ms)": round(t_uni, 2),
        "Waktu K-Means (ms)": round(t_kmeans, 2),
        "Size (KB)": round(size_uni, 2)
    })

# Tampilkan Tabel Data ke Terminal
print("\n=== DATA HASIL EKSPERIMEN ===")
df = pd.DataFrame(results)
print(df.to_string(index=False))
print("="*60)


# --- 2. VISUALISASI DAN MATRIKS (FOKUS: NORMAL.PNG) ---
# Kita ambil normal.png sebagai sampel representatif untuk laporan
target_file = 'normal.png'
img_bgr = cv2.imread(target_file)

if img_bgr is not None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    img_uni = uniform_quantization(img_rgb, 16)
    img_kmeans = nonuniform_quantization(img_rgb, 8)
    
    # TAMPILKAN MATRIKS
    print(f"\n=== SAMPEL MATRIKS PIKSEL ({target_file}) ===")
    print(">> Koordinat [100:103, 100:103]")
    print("\n1. Matriks RGB Asli:")
    print(img_rgb[100:103, 100:103])
    print("\n2. Matriks Uniform Quantization:")
    print(img_uni[100:103, 100:103])
    
    # PLOT GAMBAR
    plt.figure(figsize=(15, 8))
    
    # Baris 1: Konversi Ruang Warna
    plt.subplot(2, 4, 1); plt.imshow(img_rgb); plt.title(f"Asli ({target_file})")
    plt.subplot(2, 4, 2); plt.imshow(img_hsv); plt.title("HSV Visualization")
    plt.subplot(2, 4, 3); plt.imshow(img_lab); plt.title("LAB Visualization")
    plt.subplot(2, 4, 4); plt.hist(img_rgb.ravel(), 256, [0,256], color='r', alpha=0.5); plt.title("Histogram Asli")
    
    # Baris 2: Kuantisasi
    plt.subplot(2, 4, 5); plt.imshow(img_uni); plt.title("Uniform Quant (16 Lv)")
    plt.subplot(2, 4, 6); plt.imshow(img_kmeans); plt.title("K-Means Quant (K=8)")
    
    # Deteksi Tepi (Analisis Segmentasi)
    edges_uni = cv2.Canny(img_uni, 100, 200)
    edges_kmeans = cv2.Canny(img_kmeans, 100, 200)
    
    plt.subplot(2, 4, 7); plt.imshow(edges_uni, cmap='gray'); plt.title("Tepi: Uniform (Berisik)")
    plt.subplot(2, 4, 8); plt.imshow(edges_kmeans, cmap='gray'); plt.title("Tepi: K-Means (Bersih)")
    
    plt.tight_layout()
    plt.show()

else:
    print(f"File {target_file} tidak ditemukan untuk visualisasi.")
