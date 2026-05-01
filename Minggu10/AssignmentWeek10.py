import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

imgA = cv2.imread("noise.png", cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread("objekoverlapping.png")

if imgA is None or imgB is None:
    raise ValueError("Image tidak ditemukan")

kernels = {
    "3x3_square": cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)),
    "5x5_square": cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
    "7x7_square": cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)),
    "3x3_cross": cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)),
    "5x5_cross": cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)),
    "7x7_cross": cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7)),
    "3x3_ellipse": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
    "5x5_ellipse": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
    "7x7_ellipse": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),
}

def basic_ops(img, k):
    e1 = cv2.erode(img, k, iterations=1)
    e2 = cv2.erode(img, k, iterations=2)
    d1 = cv2.dilate(img, k, iterations=1)
    d2 = cv2.dilate(img, k, iterations=2)
    boundary = cv2.subtract(d1, e1)
    return e1, e2, d1, d2, boundary

def advanced_ops(img, k):
    o = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    c = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    g = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)
    t = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k)
    b = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)
    return o, c, g, t, b

def ocr_pipeline(img, k):
    start = time.time()
    blur = cv2.GaussianBlur(img, (5,5), 0)
    open_ = cv2.morphologyEx(blur, cv2.MORPH_OPEN, k, iterations=2)
    close_ = cv2.morphologyEx(open_, cv2.MORPH_CLOSE, k)
    thresh = cv2.adaptiveThreshold(close_,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return thresh, time.time() - start

def count_objects(img, k):
    start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    open_ = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
    dist = cv2.distanceTransform(open_, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist,0.5*dist.max(),255,0)
    fg = np.uint8(fg)
    unknown = cv2.subtract(open_, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)
    count = len(np.unique(markers)) - 2
    result = img.copy()
    result[markers==-1] = [0,0,255]
    return result, count, time.time() - start

def count_comp(img):
    n,_ = cv2.connectedComponents(img)
    return n-1

_, base = cv2.threshold(imgA,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

ocr_results = {}
count_results = []
erosion_res = {}
dilation_res = {}
gradient_res = {}
tophat_res = {}
blackhat_res = {}
count_images = {}

for name,k in kernels.items():
    e1,e2,d1,d2,bound = basic_ops(imgA, k)
    o,c,g,t,b = advanced_ops(imgA, k)
    ocr,t_ocr = ocr_pipeline(imgA, k)
    cnt_img,cnt,t_cnt = count_objects(imgB, k)

    erosion_res[name] = e1
    dilation_res[name] = d1
    gradient_res[name] = g
    tophat_res[name] = t
    blackhat_res[name] = b
    ocr_results[name] = ocr
    count_images[name] = cnt_img

    before = count_comp(base)
    after = count_comp(ocr)

    count_results.append((name, cnt, t_cnt, t_ocr, before, after))

manual_count = 10

print("\n" + "="*80)
print("HASIL EKSPERIMEN MORFOLOGI")
print("="*80)

print(f"{'Kernel':<15} | {'Count':<6} | {'Error':<5} | {'OCR Time (s)':<12} | {'Count Time (s)':<14} | {'Komponen':<12}")
print("-"*80)

for r in count_results:
    err = abs(r[1] - manual_count)
    print(f"{r[0]:<15} | {r[1]:<6} | {err:<5} | {r[3]:<12.4f} | {r[2]:<14.4f} | {r[4]} -> {r[5]}")

print("="*80)
print("Keterangan:")
print("Count        : Jumlah objek terdeteksi")
print("Error        : Selisih dengan jumlah manual")
print("OCR Time     : Waktu preprocessing OCR")
print("Count Time   : Waktu proses counting (watershed)")
print("Komponen     : Jumlah connected component (sebelum -> sesudah)")
print("="*80)

k_demo = kernels["5x5_ellipse"]
e1,e2,d1,d2,bound = basic_ops(imgA, k_demo)

plt.figure(figsize=(10,6))
plt.subplot(2,3,1); plt.imshow(imgA, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(2,3,2); plt.imshow(e1, cmap='gray'); plt.title("Erode 1x"); plt.axis('off')
plt.subplot(2,3,3); plt.imshow(e2, cmap='gray'); plt.title("Erode 2x"); plt.axis('off')
plt.subplot(2,3,4); plt.imshow(bound, cmap='gray'); plt.title("Boundary"); plt.axis('off')
plt.subplot(2,3,5); plt.imshow(d1, cmap='gray'); plt.title("Dilate 1x"); plt.axis('off')
plt.subplot(2,3,6); plt.imshow(d2, cmap='gray'); plt.title("Dilate 2x"); plt.axis('off')
plt.suptitle("Basic Morphology (Erosion & Dilation)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,12))
for i,(n,img) in enumerate(ocr_results.items()):
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(n)
    plt.axis('off')
plt.suptitle("OCR Preprocessing Result (All Kernels)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,12))
for i,(n,img) in enumerate(gradient_res.items()):
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(n)
    plt.axis('off')
plt.suptitle("Morphological Gradient (Boundary Detection)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,12))
for i,(n,img) in enumerate(tophat_res.items()):
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(n)
    plt.axis('off')
plt.suptitle("Top-Hat Transformation (Bright Features)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,12))
for i,(n,img) in enumerate(blackhat_res.items()):
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(n)
    plt.axis('off')
plt.suptitle("Black-Hat Transformation (Dark Features)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,12))
for i,(n,img) in enumerate(count_images.items()):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3,3,i+1)
    plt.imshow(img_rgb)
    plt.title(n)
    plt.axis('off')
plt.suptitle("Object Counting Result (Watershed)")
plt.tight_layout()
plt.show()

print("Done")