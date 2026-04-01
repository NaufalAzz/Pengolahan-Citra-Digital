import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import pywt
import time

# FUNGSI UTILITAS
def calculate_psnr(img1, img2):
    """Menghitung Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def analyze_fourier(image):
    """Menghitung komponen FFT dari citra"""
    f = fft2(image)
    fshift = fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    log_magnitude = np.log(1 + magnitude)
    return fshift, magnitude, log_magnitude, phase

# 1. TRANSFORMASI FOURIER & REKONSTRUKSI
def task_1_fourier_reconstruction(img_name, img):
    print(f"\n--- Memproses {img_name} : Transformasi & Rekonstruksi FFT ---")
    fshift, magnitude, log_magnitude, phase = analyze_fourier(img)
    
    # Rekonstruksi dari Fase saja (Magnitude diset 1)
    complex_phase_only = 1 * np.exp(1j * phase)
    img_phase_only = np.abs(ifft2(ifftshift(complex_phase_only)))
    
    # Normalisasi agar bisa divisualisasikan
    img_phase_only = cv2.normalize(img_phase_only, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Rekonstruksi dari Magnitudo saja (Fase diset 0)
    complex_mag_only = magnitude * np.exp(1j * 0)
    img_mag_only = np.abs(ifft2(ifftshift(complex_mag_only)))
    img_mag_only = np.log(1 + img_mag_only) # Log scale karena range nilainya sangat besar
    img_mag_only = cv2.normalize(img_mag_only, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img, cmap='gray'); ax[0].set_title('Citra Asli')
    ax[1].imshow(log_magnitude, cmap='gray'); ax[1].set_title('Spektrum Magnitudo (Log)')
    ax[2].imshow(img_phase_only, cmap='gray'); ax[2].set_title('Rekonstruksi (Fase Saja)')
    ax[3].imshow(img_mag_only, cmap='gray'); ax[3].set_title('Rekonstruksi (Magnitudo Saja)')
    for a in ax: a.axis('off')
    plt.suptitle(f'Analisis Fourier: {img_name}')
    plt.tight_layout()
    plt.show()

# 2. FILTERING DOMAIN FREKUENSI
def create_gaussian_lowpass(shape, d0):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(-crow, rows - crow)
    v = np.arange(-ccol, cols - ccol)
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    return np.exp(-(D**2) / (2 * (d0**2)))

def apply_frequency_filter(img, filter_mask):
    start_time = time.time()
    fshift, _, _, _ = analyze_fourier(img)
    fshift_filtered = fshift * filter_mask
    img_back = np.abs(ifft2(ifftshift(fshift_filtered)))
    exec_time = time.time() - start_time
    return np.clip(img_back, 0, 255).astype(np.uint8), exec_time

def task_2_filtering(img_natural, img_noise):
    print("\n--- Filtering Domain Frekuensi ---")
    
    # 2.A: Gaussian Lowpass pada Citra Natural
    cutoff = 50
    h_gaussian = create_gaussian_lowpass(img_natural.shape, cutoff)
    filtered_gaussian, time_freq = apply_frequency_filter(img_natural, h_gaussian)
    
    # Bandingkan dengan Gaussian Blur Spasial (OpenCV)
    start_time_spatial = time.time()
    spatial_gaussian = cv2.GaussianBlur(img_natural, (11, 11), 0)
    time_spatial = time.time() - start_time_spatial
    
    psnr_freq = calculate_psnr(img_natural, filtered_gaussian)
    
    print(f"Waktu komputasi Frekuensi (Gaussian LP): {time_freq:.4f} detik")
    print(f"Waktu komputasi Spasial (cv2.GaussianBlur): {time_spatial:.4f} detik")
    print(f"PSNR (Citra Asli vs Filter Frekuensi): {psnr_freq:.2f} dB")

    # Plot Hasil Lowpass
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_natural, cmap='gray'); ax[0].set_title('Natural Asli')
    ax[1].imshow(h_gaussian, cmap='gray'); ax[1].set_title(f'Mask Gaussian LP (D0={cutoff})')
    ax[2].imshow(filtered_gaussian, cmap='gray'); ax[2].set_title('Hasil Filter Frekuensi')
    for a in ax: a.axis('off')
    plt.show()

    # 2.B: Notch / Bandreject Filter untuk Citra Noise
    # Membuat manual notch filter sederhana (misal, menghilangkan frekuensi tinggi tertentu)
    rows, cols = img_noise.shape
    crow, ccol = rows // 2, cols // 2
    notch_filter = np.ones((rows, cols), np.float32)
    
    # Parameter ini (radius & ketebalan) perlu disesuaikan dengan pola spektrum CitraNoise Anda
    radius = 60
    thickness = 10
    u = np.arange(-crow, rows - crow)
    v = np.arange(-ccol, cols - ccol)
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    
    # Set 0 pada area noise (berbentuk cincin / bandreject sederhana)
    notch_filter[(D > radius - thickness) & (D < radius + thickness)] = 0
    
    filtered_noise, _ = apply_frequency_filter(img_noise, notch_filter)
    fshift_noise, _, log_mag_noise, _ = analyze_fourier(img_noise)
    
    # Plot Hasil Notch
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img_noise, cmap='gray'); ax[0].set_title('Citra Bernoise')
    ax[1].imshow(log_mag_noise, cmap='gray'); ax[1].set_title('Spektrum Bernoise')
    ax[2].imshow(notch_filter, cmap='gray'); ax[2].set_title('Bandreject/Notch Filter')
    ax[3].imshow(filtered_noise, cmap='gray'); ax[3].set_title('Hasil Denoising')
    for a in ax: a.axis('off')
    plt.show()

# 3. TRANSFORMASI WAVELET (DWT 2-Level)
def task_3_wavelet(img):
    print("\n--- Dekomposisi Wavelet 2-Level ---")
    wavelet_type = 'db4' # Bisa diganti 'haar'
    
    # Dekomposisi 2 Level
    coeffs = pywt.wavedec2(img, wavelet=wavelet_type, level=2)
    LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
    
    # Visualisasi Koefisien Level 2
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0,0].imshow(LL2, cmap='gray'); ax[0,0].set_title('LL2 (Aproksimasi)')
    ax[0,1].imshow(LH2, cmap='gray'); ax[0,1].set_title('LH2 (Detail Horizontal)')
    ax[1,0].imshow(HL2, cmap='gray'); ax[1,0].set_title('HL2 (Detail Vertikal)')
    ax[1,1].imshow(HH2, cmap='gray'); ax[1,1].set_title('HH2 (Detail Diagonal)')
    for a in ax.flat: a.axis('off')
    plt.suptitle('Koefisien Wavelet Level 2')
    plt.tight_layout()
    plt.show()

    # Rekonstruksi parsial: Membuang detail diagonal Level 1 dan 2 (Denoising sederhana)
    coeffs_filtered = [LL2, (LH2, HL2, np.zeros_like(HH2)), (LH1, HL1, np.zeros_like(HH1))]
    img_reconstructed = pywt.waverec2(coeffs_filtered, wavelet=wavelet_type)
    
    # Pastikan ukuran sama dengan aslinya (kadang ada perbedaan 1 pixel karena padding)
    img_reconstructed = img_reconstructed[:img.shape[0], :img.shape[1]]
    
    psnr_wavelet = calculate_psnr(img, img_reconstructed)
    print(f"PSNR Rekonstruksi Wavelet (tanpa HH1 & HH2): {psnr_wavelet:.2f} dB")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img, cmap='gray'); ax[0].set_title('Citra Asli')
    ax[1].imshow(img_reconstructed, cmap='gray'); ax[1].set_title('Rekonstruksi Parsial Wavelet')
    for a in ax: a.axis('off')
    plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
    # 1. Load Citra (Pastikan nama file dan foldernya benar)
    img_natural = cv2.imread('Citranatural.png', cv2.IMREAD_GRAYSCALE)
    img_noise = cv2.imread('CitraNoise.png', cv2.IMREAD_GRAYSCALE)
    
    if img_natural is None or img_noise is None:
        print("ERROR: Gambar tidak ditemukan! Pastikan 'Citranatural.png' dan 'CitraNoise.png' ada di direktori yang sama dengan script ini.")
    else:
        # Resize sedikit jika ukuran terlalu besar agar proses cepat (Opsional)
        img_natural = cv2.resize(img_natural, (512, 512))
        img_noise = cv2.resize(img_noise, (512, 512))

        # Eksekusi Tugas
        task_1_fourier_reconstruction("Citra Natural", img_natural)
        task_1_fourier_reconstruction("Citra Noise", img_noise)
        
        task_2_filtering(img_natural, img_noise)
        
        task_3_wavelet(img_natural)