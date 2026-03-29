import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

# 1. FUNGSI MEMUAT CITRA
def load_image(filename):
    """Memuat citra masukan dari file dalam mode grayscale"""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return None
    
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Gagal memuat file '{filename}'. Mungkin format file tidak didukung.")
        return None
        
    print(f"Berhasil memuat citra '{filename}' dengan ukuran {img.shape}")
    return img

# 2. FUNGSI DEGRADASI
def get_motion_blur_psf(length, angle, size=31):
    """Menghasilkan PSF untuk linear motion blur"""
    psf = np.zeros((size, size))
    center = size // 2
    angle_rad = np.deg2rad(angle)
    
    x_start = int(center - (length/2) * np.cos(angle_rad))
    y_start = int(center - (length/2) * np.sin(angle_rad))
    x_end = int(center + (length/2) * np.cos(angle_rad))
    y_end = int(center + (length/2) * np.sin(angle_rad))
    
    cv2.line(psf, (x_start, y_start), (x_end, y_end), 1, 1)
    return psf / np.sum(psf)

def apply_motion_blur(image, psf):
    """Menerapkan konvolusi PSF untuk menghasilkan motion blur"""
    blurred = cv2.filter2D(image.astype(float), -1, psf)
    return np.clip(blurred, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, sigma=20):
    """Menambahkan Gaussian noise aditif"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(float) + noise, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_pepper_noise(image, prob=0.05):
    """Menambahkan Impulse (Salt & Pepper) noise"""
    noisy = image.copy()
    total_pixels = image.size
    
    num_salt = int(total_pixels * prob / 2)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255
    
    num_pepper = int(total_pixels * prob / 2)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    return noisy

# 3. FUNGSI RESTORASI
def inverse_filter(degraded, psf, epsilon=1e-3):
    """Menerapkan Inverse Filter dengan regulasi"""
    pad_size = psf.shape[0] // 2
    padded = cv2.copyMakeBorder(degraded, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    G = np.fft.fft2(padded.astype(float))
    
    psf_padded = np.zeros_like(padded, dtype=float)
    psf_center_y, psf_center_x = psf.shape[0]//2, psf.shape[1]//2
    pad_center_y, pad_center_x = padded.shape[0]//2, padded.shape[1]//2
    y_start = pad_center_y - psf_center_y
    x_start = pad_center_x - psf_center_x
    psf_padded[y_start:y_start+psf.shape[0], x_start:x_start+psf.shape[1]] = psf
    psf_padded = np.fft.ifftshift(psf_padded)
    
    H = np.fft.fft2(psf_padded)
    H_reg = H + epsilon
    F_hat = G / H_reg
    
    restored_padded = np.abs(np.fft.ifft2(F_hat))
    restored = restored_padded[pad_size:-pad_size, pad_size:-pad_size]
    return np.clip(restored, 0, 255).astype(np.uint8)

def wiener_filter(degraded, psf, K=0.01):
    """Menerapkan Wiener Filter (minimum mean square error)"""
    pad_size = psf.shape[0] // 2
    padded = cv2.copyMakeBorder(degraded, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    G = np.fft.fft2(padded.astype(float))
    
    psf_padded = np.zeros_like(padded, dtype=float)
    psf_center_y, psf_center_x = psf.shape[0]//2, psf.shape[1]//2
    pad_center_y, pad_center_x = padded.shape[0]//2, padded.shape[1]//2
    y_start = pad_center_y - psf_center_y
    x_start = pad_center_x - psf_center_x
    psf_padded[y_start:y_start+psf.shape[0], x_start:x_start+psf.shape[1]] = psf
    psf_padded = np.fft.ifftshift(psf_padded)
    
    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)
    H_abs_sq = np.abs(H) ** 2
    W = H_conj / (H_abs_sq + K)
    F_hat = G * W
    
    restored_padded = np.abs(np.fft.ifft2(F_hat))
    restored = restored_padded[pad_size:-pad_size, pad_size:-pad_size]
    return np.clip(restored, 0, 255).astype(np.uint8)

def richardson_lucy(image, psf, iterations=15):
    """Menerapkan Iterative Richardson-Lucy Deconvolution"""
    f = image.copy().astype(np.float32)
    psf_flipped = np.flip(psf)
    
    for _ in range(iterations):
        conv = cv2.filter2D(f, -1, psf)
        conv = np.where(conv == 0, 1e-8, conv)
        ratio = image.astype(np.float32) / conv
        correction = cv2.filter2D(ratio, -1, psf_flipped)
        f = f * correction
        f = np.clip(f, 0, 255)
        
    return f.astype(np.uint8)

# 4. FUNGSI BANTUAN (METRIK & SPEKTRUM)
def calculate_metrics(original, restored):
    """Menghitung metrik evaluasi MSE, PSNR, dan SSIM"""
    mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    # Simplified SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(original.astype(float), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(restored.astype(float), (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(original.astype(float) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(restored.astype(float) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original.astype(float) * restored.astype(float), (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(ssim_map)
    
    return mse, psnr, ssim

def get_magnitude_spectrum(img):
    """Menghitung log magnitude spectrum dari citra (Domain Frekuensi)"""
    f = np.fft.fft2(img.astype(float))
    fshift = np.fft.fftshift(f)
    return np.log(1 + np.abs(fshift))

# 5. PIPELINE EKSEKUSI UTAMA
def main():
    filename = "citraasli.jpg" 
    
    # 1. Memuat citra asli
    img_asli = load_image(filename)
    if img_asli is None:
        return 
    
    # 2. Persiapkan PSF
    print("Mempersiapkan parameter degradasi (30 derajat, 15 piksel blur)...")
    psf = get_motion_blur_psf(length=15, angle=30)
    
    # 3. Buat variasi degradasi
    print("Menghasilkan variasi degradasi...")
    deg_mb = apply_motion_blur(img_asli, psf)
    deg_mb_gauss = add_gaussian_noise(deg_mb, sigma=20)
    deg_mb_sp = add_salt_pepper_noise(deg_mb, prob=0.05)
    
    # Pilih skenario untuk dievaluasi
    scen_name_plot = "Motion Blur + Gaussian"
    img_deg_plot = deg_mb_gauss
    
    print(f"\nSedang memproses restorasi citra untuk skenario '{scen_name_plot}', mohon tunggu...")
    
    # --- Proses Restorasi & Kalkulasi Metrik ---
    # 1. Inverse Filter
    start = time.time()
    res_inv = inverse_filter(img_deg_plot, psf, epsilon=1e-2)
    t_inv = time.time() - start
    mse_i, psnr_i, ssim_i = calculate_metrics(img_asli, res_inv)
    
    # 2. Wiener Filter
    start = time.time()
    res_wien = wiener_filter(img_deg_plot, psf, K=0.05) 
    t_wien = time.time() - start
    mse_w, psnr_w, ssim_w = calculate_metrics(img_asli, res_wien)
    
    # 3. Lucy-Richardson
    start = time.time()
    res_lr = richardson_lucy(img_deg_plot, psf, iterations=15)
    t_lr = time.time() - start
    mse_l, psnr_l, ssim_l = calculate_metrics(img_asli, res_lr)

    # --- OUTPUT TERMINAL SEBAGAI TABEL ---
    print("\n" + "="*80)
    print(f"{'TABEL PERBANDINGAN METRIK (SKENARIO: '+scen_name_plot+')':^80}")
    print("="*80)
    print(f"{'Metode Restorasi':<20} | {'PSNR (dB)':<12} | {'MSE':<12} | {'SSIM':<10} | {'Waktu (detik)':<15}")
    print("-" * 80)
    print(f"{'Inverse Filter':<20} | {psnr_i:<12.2f} | {mse_i:<12.2f} | {ssim_i:<10.3f} | {t_inv:<15.3f}")
    print(f"{'Wiener Filter':<20} | {psnr_w:<12.2f} | {mse_w:<12.2f} | {ssim_w:<10.3f} | {t_wien:<15.3f}")
    print(f"{'Lucy-Richardson':<20} | {psnr_l:<12.2f} | {mse_l:<12.2f} | {ssim_l:<10.3f} | {t_lr:<15.3f}")
    print("="*80)
    print("\nData berhasil dicetak! Membuka jendela grafik...")

    # --- VISUALISASI MATPLOTLIB ---
    images_dict = {
        "1. Citra Asli": img_asli,
        f"2. {scen_name_plot}": img_deg_plot,
        "3. Hasil Inverse Filter": res_inv,
        "4. Hasil Wiener Filter": res_wien,
        "5. Hasil Lucy-Richardson": res_lr
    }

    # FIGUR 1: SPASIAL
    plt.figure("Domain Spasial (Citra)", figsize=(15, 8))
    for i, (title, img) in enumerate(images_dict.items()):
        plt.subplot(2, 3, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    
    # FIGUR 2: SPEKTRUM
    plt.figure("Spektrum Frekuensi", figsize=(15, 8))
    for i, (title, img) in enumerate(images_dict.items()):
        spectrum = get_magnitude_spectrum(img)
        plt.subplot(2, 3, i+1)
        plt.imshow(spectrum, cmap='gray')
        plt.title(f"Spektrum: {title}")
        plt.axis('off')
    plt.tight_layout()

    # FIGUR 3: PROFIL INTENSITAS
    center_row = img_asli.shape[0] // 2
    plt.figure("Profil Intensitas (Analisis Ringing & Edge)", figsize=(12, 6))
    plt.plot(img_asli[center_row, :], label='Citra Asli', color='black', linewidth=2, linestyle='--')
    plt.plot(img_deg_plot[center_row, :], label='Terdegradasi', color='red', alpha=0.5)
    plt.plot(res_wien[center_row, :], label='Wiener Filter', color='green', alpha=0.8)
    plt.plot(res_lr[center_row, :], label='Lucy-Richardson', color='blue', alpha=0.8)
    
    plt.title(f"Profil Intensitas Horizontal (Baris {center_row})")
    plt.xlabel("Indeks Kolom (Piksel)")
    plt.ylabel("Intensitas Piksel (0-255)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Tampilkan semua window
    plt.show()

if __name__ == "__main__":
    main()