import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Citra Hasil Digitalisasi (Ganti nama file dengan foto Anda)
img = cv2.imread('mouse.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Konversi BGR ke RGB untuk Matplotlib

# 2. Representasi Matriks dan Vektor
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
matriks_5x5 = gray_img[0:5, 0:5] # Mengambil 5 baris dan 5 kolom pertama
vektor_1d = matriks_5x5.ravel() # Flattening matriks ke vektor

print("Representasi Matriks 5x5:\n", matriks_5x5)
print("Representasi Vektor:\n", vektor_1d)

# 3. Analisis Parameter Citra
height, width, channels = img.shape
aspect_ratio = width / height
depth_bits = img.dtype.itemsize * 8
memory_bytes = img.size * img.dtype.itemsize

print(f"Resolusi: {width} x {height}")
print(f"Aspect Ratio: {aspect_ratio:.2f}")
print(f"Bit Depth: {depth_bits}-bit, Channels: {channels}")
print(f"Memory: {memory_bytes/1024:.2f} KB | {memory_bytes/(1024*1024):.2f} MB")

# 4. Manipulasi Dasar
# Cropping (Sesuaikan koordinat [y1:y2, x1:x2])
img_cropped = img_rgb[100:400, 200:500] 

# Resizing
img_resized = cv2.resize(img_cropped, (300, 300))

# Flipping (1 = horizontal flip, 0 = vertical, -1 = both)
img_flipped = cv2.flip(img_resized, 1)

# Visualisasi dengan Matplotlib (Mirip fungsi display_image_grid di Praktikum1.py)
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(img_rgb); axes[0].set_title('Original')
axes[1].imshow(img_cropped); axes[1].set_title('Cropped')
axes[2].imshow(img_resized); axes[2].set_title('Resized (300x300)')
axes[3].imshow(img_flipped); axes[3].set_title('Flipped')

for ax in axes:
    ax.axis('off')
plt.show()