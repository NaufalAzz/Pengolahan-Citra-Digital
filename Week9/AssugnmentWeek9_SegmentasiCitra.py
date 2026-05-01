import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def evaluate_segmentation(pred, gt):
    """Menghitung metrik evaluasi segmentasi."""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    dice = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0
    
    return {'IoU': iou, 'Dice': dice, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}

def run_segmentation_pipeline(image_path, case_name):
    print(f"\n--- Memproses: {case_name} ({image_path}) ---")
    
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Gambar {image_path} tidak ditemukan!")
        return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Dummy Ground Truth (Ganti dengan file GT aslimu jika ada)
    # Di sini kita membuat GT palsu dari thresholding agar fungsi evaluasi bisa berjalan
    _, ground_truth = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    results = {}
    times = {}

    # === A. THRESHOLDING ===
    # Global
    start = time.time()
    _, results['Global'] = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    times['Global'] = time.time() - start

    # Otsu
    start = time.time()
    _, results['Otsu'] = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    times['Otsu'] = time.time() - start

    # Adaptive Gaussian
    start = time.time()
    results['Adaptive'] = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    times['Adaptive'] = time.time() - start

    # === B. EDGE DETECTION ===
    # Sobel
    start = time.time()
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    results['Sobel'] = cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    times['Sobel'] = time.time() - start

    # Canny
    start = time.time()
    results['Canny'] = cv2.Canny(cv2.GaussianBlur(img_gray, (5, 5), 1.4), 50, 150)
    times['Canny'] = time.time() - start

    # === C. WATERSHED (Untuk citra overlapping) ===
    start = time.time()
    _, bin_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    ws_result = np.zeros_like(img_gray)
    ws_result[markers > 1] = 255 # Extract segmented objects
    results['Watershed'] = ws_result
    times['Watershed'] = time.time() - start

    # === EVALUASI & TAMPILAN MATRIKS ===
    print(f"{'Metode':<12} | {'Waktu (s)':<10} | {'IoU':<6} | {'Accuracy':<8} | {'F1/Dice':<8}")
    print("-" * 55)
    for name, pred_img in results.items():
        metrics = evaluate_segmentation(pred_img, ground_truth)
        print(f"{name:<12} | {times[name]:<10.4f} | {metrics['IoU']:<6.2f} | {metrics['Accuracy']:<8.2f} | {metrics['Dice']:<8.2f}")

    # Visualisasi
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("Citra Asli")
    axes[0].axis('off')
    
    plot_idx = 1
    for name, img_res in results.items():
        axes[plot_idx].imshow(img_res, cmap='gray')
        axes[plot_idx].set_title(f"{name}")
        axes[plot_idx].axis('off')
        plot_idx += 1
        
    plt.tight_layout()
    plt.show()

# Jalankan Pipeline untuk 3 Citra
if __name__ == "__main__":
    images_to_test = {
        "Citra Bimodal": "bimodal.png",
        "Citra Iluminasi Tidak Rata": "iluminasitidakrata.png",
        "Citra Overlapping": "objekoverlapping.png"
    }
    
    for case, path in images_to_test.items():
        run_segmentation_pipeline(path, case)