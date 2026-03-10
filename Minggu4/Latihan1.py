# LATIHAN 1: MANUAL HISTOGRAM EQUALIZATION
import cv2
import numpy as np
import matplotlib.pyplot as plt


# FUNGSI MANUAL HISTOGRAM EQUALIZATION
def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    
    Parameters:
    image: Input grayscale image (0-255)
    
    Returns:
    equalized_image : hasil citra setelah equalization
    transform : fungsi transformasi intensitas
    """

    # 1. Hitung histogram
    histogram = np.zeros(256)
    for pixel in image.flatten():
        histogram[pixel] += 1
    # 2. Hitung cumulative histogram (CDF)
    cumulative_histogram = np.cumsum(histogram)
    # Normalisasi CDF
    cdf_normalized = cumulative_histogram / cumulative_histogram[-1]
    # 3. Hitung transformation function
    transform = np.round(cdf_normalized * 255).astype(np.uint8)
    # 4. Apply transformation
    equalized_image = transform[image]
    # 5. Return equalized image dan transformation function
    return equalized_image, transform


# MAIN PROGRAM
print("=== LATIHAN 1: MANUAL HISTOGRAM EQUALIZATION ===")


# Membaca gambar grayscale
image = cv2.imread("cat.jpg", 0)
if image is None:
    # jika gambar tidak ada, buat gambar sintetis
    print("File gambar tidak ditemukan, menggunakan citra sintetis")
    image = np.random.normal(120, 30, (256,256))
    image = np.clip(image,0,255).astype(np.uint8)


# Jalankan histogram equalization manual
equalized_image, transform = manual_histogram_equalization(image)


# VISUALISASI HASIL
plt.figure(figsize=(12,8))

# Original Image
plt.subplot(2,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Histogram original
plt.subplot(2,2,2)
plt.hist(image.ravel(),256,[0,256],color='gray')
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# Equalized Image
plt.subplot(2,2,3)
plt.imshow(equalized_image, cmap='gray')
plt.title("Equalized Image (Manual)")
plt.axis("off")

# Histogram equalized
plt.subplot(2,2,4)
plt.hist(equalized_image.ravel(),256,[0,256],color='blue')
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# TAMPILKAN TRANSFORMATION FUNCTION
plt.figure(figsize=(6,4))
plt.plot(transform)
plt.title("Transformation Function")
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")
plt.grid(True)
plt.show()


print("\nProgram selesai.")