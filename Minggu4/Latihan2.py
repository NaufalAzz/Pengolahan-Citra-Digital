# LATIHAN 2: MEDICAL IMAGE ENHANCEMENT PIPELINE
import cv2
import numpy as np
import matplotlib.pyplot as plt


print("=== LATIHAN 2: MEDICAL IMAGE ENHANCEMENT ===")


# FUNGSI METRICS
def calculate_metrics(original, enhanced):
    """Menghitung beberapa metrik kualitas citra"""

    metrics = {}

    metrics['mean_before'] = np.mean(original)
    metrics['mean_after'] = np.mean(enhanced)
    metrics['std_before'] = np.std(original)
    metrics['std_after'] = np.std(enhanced)
    metrics['min_before'] = np.min(original)
    metrics['max_before'] = np.max(original)
    metrics['min_after'] = np.min(enhanced)
    metrics['max_after'] = np.max(enhanced)
    metrics['dynamic_range_before'] = np.max(original) - np.min(original)
    metrics['dynamic_range_after'] = np.max(enhanced) - np.min(enhanced)

    return metrics


# MEDICAL IMAGE ENHANCEMENT PIPELINE
def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement for medical images
    Parameters:
    medical_image: Input medical image
    modality: Image modality ('X-ray', 'MRI', 'CT', 'Ultrasound')
    Returns:
    Enhanced image and enhancement report
    """

    image = medical_image.copy()

    # Step 1: Noise Reduction
    denoised = cv2.GaussianBlur(image, (3,3), 0)
    # Step 2: Enhancement berdasarkan modality

    if modality == 'X-ray':

        # X-ray biasanya low contrast → gunakan CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

    elif modality == 'MRI':
        # MRI sering membutuhkan gamma correction
        gamma = 0.8
        normalized = denoised / 255.0
        gamma_corrected = np.power(normalized, gamma)
        enhanced = np.uint8(gamma_corrected * 255)

    elif modality == 'CT':
        # CT membutuhkan normalisasi intensitas
        enhanced = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

    elif modality == 'Ultrasound':
        # Ultrasound memiliki speckle noise
        median = cv2.medianBlur(denoised, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(median)


    else:
        enhanced = denoised


    # Step 3: Hitung metrik kualitas
    report = calculate_metrics(image, enhanced)

    return enhanced, report


# MAIN PROGRAM
# Membaca gambar medis
image = cv2.imread("xray.jpg", 0)
if image is None:
    print("File gambar tidak ditemukan, menggunakan citra sintetis")
    image = np.random.normal(120, 25, (256,256))
    image = np.clip(image,0,255).astype(np.uint8)

# Pilih modality
modality = 'X-ray'
# Jalankan enhancement
enhanced_image, report = medical_image_enhancement(image, modality)


# TAMPILKAN HASIL
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Medical Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(enhanced_image, cmap='gray')
plt.title("Enhanced Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# HISTOGRAM PERBANDINGAN
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(image.ravel(),256,[0,256],color='gray')
plt.title("Original Histogram")

plt.subplot(1,2,2)
plt.hist(enhanced_image.ravel(),256,[0,256],color='blue')
plt.title("Enhanced Histogram")

plt.tight_layout()
plt.show()


# CETAK ENHANCEMENT REPORT
print("\n=== Enhancement Report ===")

for key, value in report.items():
    print(f"{key} : {value:.2f}")