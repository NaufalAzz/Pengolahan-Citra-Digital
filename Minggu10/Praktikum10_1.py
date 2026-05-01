import cv2
import numpy as np
import matplotlib.pyplot as plt

def latihan_1():
    # Buat citra biner test pattern
    img = np.zeros((200, 300), dtype=np.uint8)
    
    # Tambahkan berbagai bentuk
    cv2.rectangle(img, (30, 30), (80, 80), 255, -1)      # Square
    cv2.circle(img, (150, 50), 20, 255, -1)              # Circle
    cv2.rectangle(img, (200, 30), (220, 70), 255, -1)    # Vertical line
    cv2.rectangle(img, (250, 40), (270, 60), 255, -1)    # Horizontal line
    
    # Tambahkan noise (salt and pepper)
    noise = np.random.random(img.shape) < 0.05
    img_noisy = img.copy()
    img_noisy[noise] = 255 - img_noisy[noise]
    
    # Define structuring elements
    kernels = {
        '3x3 Rectangle': cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)),
        '5x5 Rectangle': cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
        '3x3 Ellipse': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
        '3x3 Cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    }
    
    # Apply operations
    fig, axes = plt.subplots(5, 5, figsize=(15, 9))
    
    # Original images
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Original Clean')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img_noisy, cmap='gray')
    axes[0,1].set_title('Original Noisy')
    axes[0,1].axis('off')
    
    for i in range(3):
        axes[0,i+2].axis('off')
    
    # Apply erosion and dilation
    operations = ['Erosion', 'Dilation', 'Opening', 'Closing']
    
    for row, op_name in enumerate(operations, 1):
        for col, (kernel_name, kernel) in enumerate(kernels.items()):
            if op_name == 'Erosion':
                result = cv2.erode(img_noisy, kernel, iterations=1)
            elif op_name == 'Dilation':
                result = cv2.dilate(img_noisy, kernel, iterations=1)
            elif op_name == 'Opening':
                result = cv2.morphologyEx(img_noisy, cv2.MORPH_OPEN, kernel)
            elif op_name == 'Closing':
                result = cv2.morphologyEx(img_noisy, cv2.MORPH_CLOSE, kernel)
            
            axes[row, col].imshow(result, cmap='gray')
            axes[row, col].set_title(f'{op_name}\n{kernel_name}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analisis hasil
    print("ANALYSIS OF MORPHOLOGICAL OPERATIONS:")
    print("=" * 50)
    print("1. Erosion:")
    print("   - Menghilangkan noise kecil")
    print("   - Mengecilkan objek")
    print("   - Dapat memisahkan objek yang menyatu")
    
    print("\n2. Dilation:")
    print("   - Mengisi lubang kecil")
    print("   - Membesarkan objek")
    print("   - Dapat menyambungkan objek terpisah")
    
    print("\n3. Opening (Erosion lalu Dilation):")
    print("   - Menghilangkan noise tanpa mengubah ukuran objek")
    print("   - Efektif untuk noise removal")
    
    print("\n4. Closing (Dilation lalu Erosion):")
    print("   - Mengisi lubang tanpa mengubah ukuran objek")
    print("   - Efektif untuk hole filling")

# Jalankan latihan 1
latihan_1()

