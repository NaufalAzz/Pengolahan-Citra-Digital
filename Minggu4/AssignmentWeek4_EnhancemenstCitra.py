import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Mengabaikan warning log(0)
warnings.filterwarnings('ignore')

# Load Images
under = cv2.imread(r"under.PNG", 0)
over = cv2.imread(r"over.PNG", 0)
uneven = cv2.imread(r"uneven.PNG", 0)
images = {
    "Underexposed": under,
    "Overexposed": over,
    "Uneven": uneven
}

# Point Processing
def negative(img):
    return 255 - img

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img.astype(float))
    return np.array(log_img, dtype=np.uint8)

def gamma_transform(img, gamma):
    table = np.array([(i / 255.0) ** gamma * 255
        for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# Contrast Stretching
def contrast_stretch_manual(img):
    rmin = np.min(img)
    rmax = np.max(img)
    if rmax == rmin:
        return img
    stretched = (img - rmin) / (rmax - rmin) * 255
    return stretched.astype(np.uint8)

def contrast_stretch_auto(img):
    p2, p98 = np.percentile(img, (2, 98))
    if p98 == p2:
        return img
    stretched = np.clip((img - p2) * 255 / (p98 - p2), 0, 255)
    return stretched.astype(np.uint8)

# Histogram Equalization
def hist_equalization(img):
    return cv2.equalizeHist(img)

# CLAHE
def clahe_enhancement(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

# Metrics
def contrast_ratio(img):
    Imax = float(np.max(img))
    Imin = float(np.min(img))
    return (Imax - Imin) / (Imax + Imin + 1e-5)

def entropy(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()
    hist = hist / hist.sum()

    ent = 0
    for p in hist:
        if p > 0:
            ent += -p * np.log2(p)
    return float(ent)

# Fungsi Histogram Bersebelahan
def show_histogram_comparison(img_before, title_before, img_after, title_after):
    """Fungsi untuk menampilkan dua histogram bersebelahan dalam 1 window"""
    plt.figure(figsize=(12, 5)) # Ukuran window memanjang ke samping

    # Plot Histogram Sebelum (Kiri)
    plt.subplot(1, 2, 1)
    plt.hist(img_before.ravel(), bins=256, range=[0,256], color='gray')
    plt.title(title_before)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Plot Histogram Sesudah (Kanan)
    plt.subplot(1, 2, 2)
    plt.hist(img_after.ravel(), bins=256, range=[0,256], color='black')
    plt.title(title_after)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Processing Pipeline
for name, img in images.items():

    if img is None:
        print(name, "image not found! Pastikan file berada di folder yang sama.")
        continue

    print("\n==========================")
    print("Processing:", name)
    print("==========================")

    neg = negative(img)
    log_img = log_transform(img)

    gamma05 = gamma_transform(img, 0.5)
    gamma1 = gamma_transform(img, 1.0)
    gamma2 = gamma_transform(img, 2.0)

    stretch_manual = contrast_stretch_manual(img)
    stretch_auto = contrast_stretch_auto(img)

    hist_eq = hist_equalization(img)
    clahe = clahe_enhancement(img)

    results = {
        "Original": img,
        "Negative": neg,
        "Log": log_img,
        "Gamma 0.5": gamma05,
        "Gamma 1.0": gamma1,
        "Gamma 2.0": gamma2,
        "Stretch Manual": stretch_manual,
        "Stretch Auto": stretch_auto,
        "Histogram EQ": hist_eq,
        "CLAHE": clahe
    }

    # Show Images
    plt.figure(figsize=(15, 8))

    for i,(k,v) in enumerate(results.items()):
        plt.subplot(3,4,i+1)
        plt.imshow(v, cmap='gray')
        plt.title(k)
        plt.axis("off")

    plt.suptitle(f"Hasil Enhancement: {name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Histogram Before / After
    # Memanggil fungsi baru agar tampil berdampingan
    show_histogram_comparison(
        img, f"{name} - Histogram Asli", 
        hist_eq, f"{name} - Setelah Histogram EQ"
    )

    # Metrics
    print("Metrics:")
    for k,v in results.items():
        print(
            f"{k:<15} | Contrast: {contrast_ratio(v):.3f} | Entropy: {entropy(v):.3f}"
        )