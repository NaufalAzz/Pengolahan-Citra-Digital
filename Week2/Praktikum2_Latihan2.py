import cv2
import numpy as np
import matplotlib.pyplot as plt

def simulate_image_aliasing(image, downsampling_factors):
    results = []
    height, width = image.shape[:2]
    
    for factor in downsampling_factors:
        # 1. Aliasing (Downsampling langsung tanpa filter)
        aliased_small = image[::factor, ::factor]
        
        # 2. Anti-Aliasing (Gaussian Blur sebelum downsampling)
        sigma = factor / 2.0 
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        aa_small = blurred[::factor, ::factor]
        
        # Resize kembali ke ukuran asli untuk visualisasi perbandingan
        aliased_recon = cv2.resize(aliased_small, (width, height), interpolation=cv2.INTER_NEAREST)
        aa_recon = cv2.resize(aa_small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        results.append({
            'factor': factor,
            'aliased': aliased_recon,
            'anti_aliased': aa_recon
        })
        
    return results

# --- Program Utama ---
filename = 'hollow.jpg'
img = cv2.imread(filename)

if img is None:
    print(f"Error: File '{filename}' tidak ditemukan.")
else:
    # Konversi ke RGB untuk Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Faktor downsampling yang diuji
    factors = [2, 4, 8]
    
    # Jalankan simulasi
    simulation_results = simulate_image_aliasing(img_rgb, factors)
    
    # Visualisasi Hasil
    rows = len(factors)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    
    # Jika hanya 1 baris, axes perlu diubah jadi list 2D agar loop tetap jalan
    if rows == 1: axes = [axes]

    for i, res in enumerate(simulation_results):
        factor = res['factor']
        
        # Kolom 1: Original
        axes[i][0].imshow(img_rgb)
        axes[i][0].set_title("Original")
        axes[i][0].axis('off')
        
        # Kolom 2: Aliasing (Tanpa Filter)
        axes[i][1].imshow(res['aliased'])
        axes[i][1].set_title(f"Aliasing (Factor {factor}x)")
        axes[i][1].axis('off')
        
        # Kolom 3: Anti-Aliasing (Dengan Filter)
        axes[i][2].imshow(res['anti_aliased'])
        axes[i][2].set_title(f"Anti-Aliasing (Factor {factor}x)")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()