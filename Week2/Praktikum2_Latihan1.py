import cv2
import numpy as np
import matplotlib.pyplot as plt

#Fungsi Analisis

def analyze_color_model_suitability(image, application):
    """
    Menganalisis model warna terbaik untuk aplikasi tertentu.
    Parameters:
        image: Citra input
        application: 'skin_detection', 'shadow_removal', 'text_extraction', 'object_detection'
    """
    result = {}
    
    # 1. SKIN DETECTION (Deteksi Kulit)
    if application == 'skin_detection':
        result['best_model_name'] = 'HSV'
        result['reason'] = "HSV memisahkan warna (Hue) dari kecerahan, konsisten untuk warna kulit."
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Range warna kulit (contoh umum)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        result['processed_image'] = cv2.bitwise_and(image, image, mask=mask)

    # 2. SHADOW REMOVAL (Penghapusan Bayangan)
    elif application == 'shadow_removal':
        result['best_model_name'] = 'LAB'
        result['reason'] = "Channel L (Lightness) diproses terpisah agar warna tidak pudar."
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Ratakan pencahayaan pada channel L
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        result['processed_image'] = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. TEXT EXTRACTION (Ekstraksi Teks)
    elif application == 'text_extraction':
        result['best_model_name'] = 'Grayscale'
        result['reason'] = "Komputasi ringan dan fokus pada kontras tinggi."
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        result['processed_image'] = thresh

    # 4. OBJECT DETECTION (Deteksi Objek)
    elif application == 'object_detection':
        result['best_model_name'] = 'HSV (Hue Channel)'
        result['reason'] = "Menggunakan Hue untuk membedakan objek berdasarkan warna murni."
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Mengembalikan channel Hue saja
        result['processed_image'] = hsv[:,:,0] 

    else:
        result['best_model_name'] = 'RGB'
        result['reason'] = "Default."
        result['processed_image'] = image

    return result

#Simluasi Aliasing

def simulate_image_aliasing():
    """
    Demonstrasi efek aliasing (pola moire) dan solusinya.
    """
    print("\n=== SIMULASI EFEK ALIASING ===")
    
    # Membuat Pola Frekuensi Tinggi
    width, height = 400, 400
    x = np.linspace(0, 100, width)
    y = np.linspace(0, 100, height)
    X, Y = np.meshgrid(x, y)
    z = np.sin(X**2 + Y**2)
    img_high_freq = ((z + 1) * 127.5).astype(np.uint8)
    
    # 1. Aliasing (Downsampling kasar)
    factor = 4
    img_aliased = img_high_freq[::factor, ::factor]
    img_aliased_recon = cv2.resize(img_aliased, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # 2. Anti-Aliasing (Blur dulu)
    img_blurred = cv2.GaussianBlur(img_high_freq, (5, 5), 1.5)
    img_aa = cv2.resize(img_blurred, (width//factor, height//factor), interpolation=cv2.INTER_AREA)
    img_aa_recon = cv2.resize(img_aa, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(img_high_freq, cmap='gray')
    plt.title("Asli (High Freq)")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(img_aliased_recon, cmap='gray')
    plt.title("Aliasing (Pola Palsu)")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(img_aa_recon, cmap='gray')
    plt.title("Anti-Aliasing (Blur)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#Main Program

if __name__ == "__main__":
    # 1. Load Image "hollow.jpg"
    filename = 'hollow.jpg'
    img = cv2.imread(filename)

    if img is None:
        print(f"Error: File '{filename}' tidak ditemukan di folder ini.")
    else:
        print(f"Berhasil memuat '{filename}'. Memulai analisis...")

        # 2. Jalankan Analisis
        apps = ['skin_detection', 'shadow_removal', 'text_extraction', 'object_detection']
        
        plt.figure(figsize=(15, 6))
        
        # Tampilkan Citra Asli
        plt.subplot(1, 5, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Citra Asli")
        plt.axis('off')

        # Loop untuk setiap aplikasi
        for i, app in enumerate(apps):
            res = analyze_color_model_suitability(img, app)
            processed = res['processed_image']
            
            plt.subplot(1, 5, i+2)
            if len(processed.shape) == 2:
                plt.imshow(processed, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                
            plt.title(f"{app}\n({res['best_model_name']})", fontsize=9)
            plt.xlabel(res['reason'], fontsize=7)
            plt.xticks([])
            plt.yticks([])
        
        plt.tight_layout()
        plt.show()

        # 3. Jalankan Simulasi Aliasing
        simulate_image_aliasing()