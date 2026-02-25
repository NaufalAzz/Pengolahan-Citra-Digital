import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_my_image(image_path, sample_img=None):
    """Analyze your own image"""
    
    img = cv2.imread(image_path)
    
    if img is None:
        print("Gambar tidak ditemukan!")
        return None
    
    print("\n=== ANALISIS CITRA PRIBADI ===")
    
    # 1. Dimensi dan Resolusi
    height, width, channels = img.shape
    resolution = width * height
    print(f"Dimensi          : {width} x {height}")
    print(f"Jumlah Channel   : {channels}")
    print(f"Resolusi         : {resolution:,} pixels")
    
    # 2. Aspect Ratio
    aspect_ratio = width / height
    print(f"Aspect Ratio     : {aspect_ratio:.2f}")
    
    # 3. Konversi ke Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    color_memory = img.size * img.dtype.itemsize
    gray_memory = gray.size * gray.dtype.itemsize
    
    print(f"Ukuran Memori RGB  : {color_memory/1024:.2f} KB")
    print(f"Ukuran Memori Gray : {gray_memory/1024:.2f} KB")
    
    # 4. Statistik
    print("\n--- Statistik RGB ---")
    print(f"Min  : {img.min()}")
    print(f"Max  : {img.max()}")
    print(f"Mean : {img.mean():.2f}")
    print(f"Std  : {img.std():.2f}")
    
    print("\n--- Statistik Grayscale ---")
    print(f"Min  : {gray.min()}")
    print(f"Max  : {gray.max()}")
    print(f"Mean : {gray.mean():.2f}")
    print(f"Std  : {gray.std():.2f}")
    
    # 5. Histogram semua channel
    plt.figure(figsize=(12,5))
    
    # Histogram grayscale
    plt.subplot(1,2,1)
    plt.hist(gray.ravel(), 256, [0,256], color='gray')
    plt.title("Histogram Grayscale")
    
    # Histogram RGB
    plt.subplot(1,2,2)
    colors = ('b','g','r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist, color=color)
    plt.title("Histogram RGB")
    plt.legend(['Blue','Green','Red'])
    
    plt.tight_layout()
    plt.show()
    
    # 6. Bandingkan dengan citra sample
    if sample_img is not None:
        print("\n=== PERBANDINGAN DENGAN CITRA SAMPLE ===")
        print(f"Resolusi Citra Pribadi : {resolution:,}")
        print(f"Resolusi Citra Sample  : {sample_img.shape[1] * sample_img.shape[0]:,}")
    
    return {
        "width": width,
        "height": height,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "mean": img.mean(),
        "std": img.std()
    }
sample_img = cv2.imread("Laptop.jpeg")

my_results = analyze_my_image("Mouse.jpeg", sample_img)