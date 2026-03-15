import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

# --- 1. FUNGSI PEMBUATAN NOISE ---
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image.astype(float) + noise, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.05):
    noisy = image.copy()
    total_pixels = image.size
    
    num_salt = int(total_pixels * prob / 2)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[tuple(salt_coords)] = 255
    
    num_pepper = int(total_pixels * prob / 2)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[tuple(pepper_coords)] = 0
    return noisy

def add_speckle_noise(image, variance=0.1):
    noise = np.random.randn(*image.shape) * variance
    noisy = image + image * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# --- 2. PERSIAPAN CITRA ---
image_path = 'GambarCitra.png'
# Membaca gambar langsung dalam mode Grayscale (0)
img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Cek apakah gambar berhasil dimuat
if img_original is None:
    print(f"[PERINGATAN] File '{image_path}' tidak ditemukan di folder ini.")
    print("Menggunakan citra dummy sebagai pengganti...\n")
    img_original = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img_original, (50, 50), (200, 200), 128, -1)
    cv2.circle(img_original, (128, 128), 50, 255, -1)
else:
    print(f"[SUKSES] Berhasil memuat '{image_path}' dengan ukuran {img_original.shape}\n")

# Buat variasi noise
img_gaussian = add_gaussian_noise(img_original)
img_sp = add_salt_and_pepper_noise(img_original)
img_speckle = add_speckle_noise(img_original)

noised_images = {
    'Gaussian Noise': img_gaussian,
    'Salt & Pepper': img_sp,
    'Speckle Noise': img_speckle
}

# --- 3. DEFINISI FILTER ---
filters = {
    'Mean 3x3': lambda x: cv2.blur(x, (3, 3)),
    'Mean 5x5': lambda x: cv2.blur(x, (5, 5)),
    'Gaussian (sigma=1)': lambda x: cv2.GaussianBlur(x, (5, 5), 1),
    'Gaussian (sigma=2)': lambda x: cv2.GaussianBlur(x, (5, 5), 2),
    'Median 3x3': lambda x: cv2.medianBlur(x, 3),
    'Median 5x5': lambda x: cv2.medianBlur(x, 5),
    'Min 3x3': lambda x: cv2.erode(x, np.ones((3, 3), np.uint8))
}

# --- 4. EVALUASI DAN VISUALISASI ---
results = []

for noise_name, noisy_img in noised_images.items():
    print(f"Evaluasi untuk: {noise_name}")
    print("-" * 75)
    print(f"{'Filter':<20} | {'Waktu (ms)':<10} | {'MSE':<8} | {'PSNR (dB)':<10} | {'SSIM':<6}")
    print("-" * 75)
    
    # Plotting setup
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"Restorasi - {noise_name}", fontsize=16)
    axes = axes.ravel()
    
    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title("Noisy Image")
    axes[0].axis('off')
    
    for idx, (filter_name, filter_func) in enumerate(filters.items()):
        # Waktu komputasi
        start_time = time.time()
        restored_img = filter_func(noisy_img)
        calc_time = (time.time() - start_time) * 1000 # dalam ms
        
        # Metrik performa
        mse_val = mean_squared_error(img_original, restored_img)
        psnr_val = peak_signal_noise_ratio(img_original, restored_img, data_range=255)
        # Menggunakan data_range=255 agar lebih stabil untuk citra 8-bit
        ssim_val = ssim(img_original, restored_img, data_range=255) 
        
        print(f"{filter_name:<20} | {calc_time:<10.2f} | {mse_val:<8.2f} | {psnr_val:<10.2f} | {ssim_val:<6.3f}")
        
        # Simpan ke dict untuk analisis lanjut jika perlu
        results.append({
            'Noise': noise_name, 'Filter': filter_name,
            'Time': calc_time, 'MSE': mse_val, 'PSNR': psnr_val, 'SSIM': ssim_val
        })
        
        # Visualisasi
        axes[idx+1].imshow(restored_img, cmap='gray')
        axes[idx+1].set_title(f"{filter_name}\nPSNR: {psnr_val:.1f} | SSIM: {ssim_val:.2f}")
        axes[idx+1].axis('off')
        
    plt.tight_layout()
    plt.show()
    print("\n")