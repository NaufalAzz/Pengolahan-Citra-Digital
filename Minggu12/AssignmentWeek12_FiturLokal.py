"""
==============================================================
PRAKTIKUM 12 - OBJECT MATCHING PIPELINE LENGKAP
Local Feature Descriptors + Bag of Visual Words + PCA
==============================================================
Sistem ini bisa dijalankan dengan 2 mode:
  1. Dataset nyata  -> taruh folder dataset_minggu12 di direktori yang sama
  2. Dataset sintetis -> otomatis generate jika folder tidak ditemukan
==============================================================
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from collections import defaultdict, Counter
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ─────────────────────────────────────────────
DATASET_PATH   = "dataset_minggu12"
CLASS_NAMES    = ["boneka", "botol", "buku", "gelas", "remote"]
VOCAB_SIZES    = [10, 20, 50, 100]
PCA_COMPONENTS = [16, 32, 64, 128]

# ─────────────────────────────────────────────
# BAGIAN 0: GENERATE / LOAD DATASET
# ─────────────────────────────────────────────

def generate_synthetic_dataset():
    """Buat 5 kelas x 5 citra sintetis dengan berbagai transformasi."""
    print("[INFO] Dataset nyata tidak ditemukan → membuat dataset sintetis...")

    def draw_boneka(img):
        cv2.circle(img, (50, 35), 20, 200, -1)           # kepala
        cv2.rectangle(img, (35, 55), (65, 90), 160, -1)  # badan
        cv2.line(img, (35, 60), (15, 80), 160, 4)        # tangan kiri
        cv2.line(img, (65, 60), (85, 80), 160, 4)        # tangan kanan
        cv2.line(img, (42, 90), (30, 115), 160, 4)       # kaki kiri
        cv2.line(img, (58, 90), (70, 115), 160, 4)       # kaki kanan
        return img

    def draw_botol(img):
        cv2.rectangle(img, (38, 50), (62, 110), 180, -1) # badan
        cv2.rectangle(img, (44, 35), (56, 50), 180, -1)  # leher
        cv2.ellipse(img, (50, 50), (12, 5), 0, 0, 360, 200, -1)  # bahu
        return img

    def draw_buku(img):
        cv2.rectangle(img, (20, 25), (80, 105), 150, -1) # cover
        cv2.line(img, (50, 25), (50, 105), 80, 3)        # spine
        for y in range(35, 100, 10):
            cv2.line(img, (53, y), (75, y), 120, 1)      # garis teks
        return img

    def draw_gelas(img):
        pts = np.array([[35,30],[65,30],[70,110],[30,110]], np.int32)
        cv2.fillPoly(img, [pts], 170)                     # badan gelas
        cv2.ellipse(img, (50, 30), (15, 5), 0, 0, 360, 210, -1)  # mulut
        return img

    def draw_remote(img):
        cv2.rectangle(img, (30, 20), (70, 115), 140, -1) # badan
        cv2.rectangle(img, (37, 27), (63, 45), 200, -1)  # layar
        for row in range(3):
            for col in range(3):
                cx = 40 + col * 10
                cy = 55 + row * 15
                cv2.circle(img, (cx, cy), 4, 220, -1)    # tombol
        return img

    drawers = [draw_boneka, draw_botol, draw_buku, draw_gelas, draw_remote]
    images, labels = [], []

    for cls_idx, (name, drawer) in enumerate(zip(CLASS_NAMES, drawers)):
        for variant in range(5):
            base = np.zeros((140, 100), dtype=np.uint8)
            base = drawer(base)

            # Transformasi berbeda tiap varian
            if variant == 0:
                img = base.copy()
            elif variant == 1:                          # rotasi
                M = cv2.getRotationMatrix2D((50, 70), 20, 1.0)
                img = cv2.warpAffine(base, M, (100, 140))
            elif variant == 2:                          # skala
                img = cv2.resize(base, None, fx=0.75, fy=0.75)
                img = cv2.resize(img, (100, 140))
            elif variant == 3:                          # iluminasi
                img = cv2.convertScaleAbs(base, alpha=0.6, beta=20)
            else:                                       # oklusi parsial
                img = base.copy()
                img[50:90, 20:60] = 0

            # Tambah noise ringan
            noise = np.random.normal(0, 8, img.shape)
            img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

            images.append(img)
            labels.append(cls_idx)

    return images, labels


def load_dataset():
    """Load dataset nyata dari folder, fallback ke sintetis."""
    if not os.path.exists(DATASET_PATH):
        return generate_synthetic_dataset()

    images, labels = [], []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(DATASET_PATH, cls)
        if not os.path.exists(folder):
            continue
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for f in files[:5]:
            path = os.path.join(folder, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                images.append(img)
                labels.append(cls_idx)

    if len(images) == 0:
        print("[WARN] Folder ditemukan tapi kosong → fallback sintetis")
        return generate_synthetic_dataset()

    print(f"[INFO] Dataset nyata dimuat: {len(images)} citra dari {DATASET_PATH}")
    return images, labels


# ─────────────────────────────────────────────
# BAGIAN 1: FEATURE EXTRACTION
# ─────────────────────────────────────────────

def init_detectors():
    detectors = {}
    detectors['SIFT'] = cv2.SIFT_create(nfeatures=500)
    try:
        detectors['SURF'] = cv2.xfeatures2d.SURF_create(400)
    except Exception:
        detectors['SURF'] = None

    detectors['ORB'] = cv2.ORB_create(nfeatures=500)
    return detectors


def extract_features(images, detectors):
    """Ekstrak keypoints + descriptors dari semua citra untuk semua metode."""
    results = defaultdict(list)   # results[method] = list of (kp, des, time_ms)

    for method_name, detector in detectors.items():
        if detector is None:
            print(f"  [{method_name}] Tidak tersedia, dilewati.")
            continue

        for img in images:
            t0 = time.perf_counter()
            kp, des = detector.detectAndCompute(img, None)
            elapsed = (time.perf_counter() - t0) * 1000   # ms
            results[method_name].append((kp, des, elapsed))

    return results


def summarize_extraction(results, labels):
    """Cetak tabel ringkasan ekstraksi fitur."""
    print("\n" + "="*75)
    print(f"{'RINGKASAN EKSTRAKSI FITUR':^75}")
    print("="*75)
    print(f"{'Metode':<8} {'Rata-rata KP':<14} {'Waktu (ms)':<12} {'Dim Desc':<10} {'Tipe'}")
    print("-"*75)

    summary = {}
    for method, data in results.items():
        kp_counts = [len(kp) for kp, _, _ in data]
        times     = [t for _, _, t in data]
        des_list  = [des for _, des, _ in data if des is not None]
        dim       = des_list[0].shape[1] if des_list else 0
        dtype     = 'Float32' if method != 'ORB' else 'Binary (uint8)'

        avg_kp   = np.mean(kp_counts)
        avg_time = np.mean(times)

        print(f"{method:<8} {avg_kp:<14.1f} {avg_time:<12.2f} {dim:<10} {dtype}")
        summary[method] = dict(avg_kp=avg_kp, avg_time=avg_time, dim=dim, dtype=dtype)

    return summary


# ─────────────────────────────────────────────
# BAGIAN 2: FEATURE MATCHING
# ─────────────────────────────────────────────

def brute_force_match(desc1, desc2, method):
    if desc1 is None or desc2 is None:
        return []
    norm = cv2.NORM_HAMMING if method == 'ORB' else cv2.NORM_L2
    bf   = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:   # Lowe's ratio test
                good.append(m)
    return good


def flann_match(desc1, desc2, method):
    if desc1 is None or desc2 is None:
        return []
    if method == 'ORB':
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)
    index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except Exception:
        return []
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    return good


def estimate_homography(kp1, kp2, good_matches):
    """RANSAC homography estimation."""
    if len(good_matches) < 4:
        return None, 0
    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def run_matching_evaluation(feature_results, labels):
    """
    Jalankan BF + FLANN matching antara citra referensi (img[0] tiap kelas)
    dan citra uji (img[1..4] tiap kelas).
    """
    print("\n" + "="*75)
    print(f"{'EVALUASI FEATURE MATCHING':^75}")
    print("="*75)

    matching_summary = defaultdict(lambda: defaultdict(list))

    for method, data in feature_results.items():
        print(f"\n  [{method}]")
        print(f"  {'Pasangan':<20} {'BF Good':<10} {'FLANN Good':<12} {'RANSAC Inlier'}")
        print(f"  {'-'*55}")

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            ref_idx  = cls_idx * 5              # citra pertama tiap kelas
            test_idxs = [ref_idx + j for j in range(1, 5)]

            kp_ref,  des_ref,  _ = data[ref_idx]

            for t_idx in test_idxs:
                kp_t, des_t, _ = data[t_idx]

                bf_good    = brute_force_match(des_ref, des_t, method)
                flann_good = flann_match(des_ref, des_t, method)
                _, inliers = estimate_homography(kp_ref, kp_t, bf_good)

                label = f"{cls_name}[{t_idx - ref_idx}]"
                print(f"  {label:<20} {len(bf_good):<10} {len(flann_good):<12} {inliers}")

                matching_summary[method]['bf'].append(len(bf_good))
                matching_summary[method]['flann'].append(len(flann_good))
                matching_summary[method]['ransac'].append(inliers)

    return matching_summary


# ─────────────────────────────────────────────
# BAGIAN 3: BAG OF VISUAL WORDS (BoVW)
# ─────────────────────────────────────────────

def collect_all_descriptors(feature_results, method='SIFT'):
    """Kumpulkan semua descriptor dari semua citra untuk satu metode."""
    all_des = []
    for kp, des, _ in feature_results[method]:
        if des is not None:
            if method == 'ORB':
                all_des.append(des.astype(np.float32))
            else:
                all_des.append(des)
    if all_des:
        return np.vstack(all_des)
    return None


def build_vocabulary(descriptors, k):
    """K-means clustering untuk vocabulary visual words."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, _, centers = cv2.kmeans(descriptors.astype(np.float32), k, None,
                               criteria, 5, cv2.KMEANS_PP_CENTERS)
    return centers


def compute_histogram(des, centers, k):
    """Hitung histogram visual words untuk satu citra."""
    if des is None:
        return np.zeros(k)
    des_f = des.astype(np.float32)
    dists = np.linalg.norm(des_f[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    hist, _ = np.histogram(nearest, bins=k, range=(0, k))
    norm = hist.sum()
    return hist / norm if norm > 0 else hist.astype(float)


def compute_tfidf(histograms):
    tf  = histograms
    df  = np.sum(histograms > 0, axis=0)
    N   = histograms.shape[0]
    idf = np.log((N + 1) / (df + 1)) + 1
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return tfidf / norms


def run_bovw_pipeline(feature_results, labels_arr, vocab_sizes=VOCAB_SIZES):
    """Jalankan BoVW dengan berbagai ukuran vocabulary dan klasifikasi KNN + SVM."""
    print("\n" + "="*75)
    print(f"{'BAG OF VISUAL WORDS PIPELINE':^75}")
    print("="*75)

    method = 'SIFT' if 'SIFT' in feature_results else list(feature_results.keys())[0]
    all_des = collect_all_descriptors(feature_results, method)
    if all_des is None:
        print("[ERROR] Tidak ada descriptor untuk BoVW")
        return {}, {}

    bovw_results = {}

    print(f"\n  Metode descriptor: {method}")
    print(f"  {'VocabSize':<12} {'KNN Acc':<12} {'SVM Acc':<12} {'Waktu Build (s)'}")
    print(f"  {'-'*50}")

    all_conf_matrices = {}
    all_histograms    = {}

    for k in vocab_sizes:
        t0     = time.perf_counter()
        centers = build_vocabulary(all_des, k)
        build_t = time.perf_counter() - t0

        # Buat histogram tiap citra
        hists = []
        for kp, des, _ in feature_results[method]:
            hists.append(compute_histogram(des, centers, k))
        hists = np.array(hists)
        tfidf = compute_tfidf(hists)
        all_histograms[k] = tfidf

        # Leave-one-out evaluation
        knn_preds, svm_preds, true_labels = [], [], []
        for i in range(len(tfidf)):
            mask = np.ones(len(tfidf), dtype=bool)
            mask[i] = False
            X_train = tfidf[mask]
            y_train = labels_arr[mask]
            X_test  = tfidf[i:i+1]
            y_test  = labels_arr[i]

            # KNN
            knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
            knn.fit(X_train, y_train)
            knn_preds.append(knn.predict(X_test)[0])

            # SVM
            svm = SVC(kernel='rbf', C=1.0, probability=True)
            svm.fit(X_train, y_train)
            svm_preds.append(svm.predict(X_test)[0])

            true_labels.append(y_test)

        knn_acc = np.mean(np.array(knn_preds) == np.array(true_labels))
        svm_acc = np.mean(np.array(svm_preds) == np.array(true_labels))

        print(f"  {k:<12} {knn_acc*100:<12.1f} {svm_acc*100:<12.1f} {build_t:.2f}")

        bovw_results[k] = dict(knn_acc=knn_acc, svm_acc=svm_acc, build_time=build_t)
        all_conf_matrices[k] = confusion_matrix(true_labels, svm_preds)

    return bovw_results, all_conf_matrices, all_histograms


# ─────────────────────────────────────────────
# BAGIAN 4: PCA
# ─────────────────────────────────────────────

def manual_pca(data, n_components):
    mean   = np.mean(data, axis=0)
    centered = data - mean
    cov    = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx    = np.argsort(evals)[::-1]
    evecs  = evecs[:, idx]
    evals  = evals[idx]
    comp   = evecs[:, :n_components]
    proj   = centered @ comp
    var_exp = np.sum(evals[:n_components]) / np.sum(evals) * 100
    return proj, var_exp, mean, comp


def run_pca_evaluation(feature_results, labels_arr, components=PCA_COMPONENTS):
    """Evaluasi matching accuracy vs kompresi PCA."""
    print("\n" + "="*75)
    print(f"{'PCA DIMENSIONALITY REDUCTION':^75}")
    print("="*75)

    pca_results = {}

    for method in feature_results:
        all_des = collect_all_descriptors(feature_results, method)
        if all_des is None:
            continue

        dim_orig = all_des.shape[1]
        print(f"\n  Metode: {method} | Dimensi asli: {dim_orig}")
        print(f"  {'Komponen':<12} {'Var Dijelaskan':<18} {'Kompresi':<14} {'KNN Acc'}")
        print(f"  {'-'*56}")

        method_pca = []
        for n in components:
            if n >= dim_orig:
                print(f"  {n:<12} {'(skip - terlalu besar)'}")
                continue

            proj, var_exp, _, _ = manual_pca(all_des, n)
            compression = (1 - n / dim_orig) * 100

            # Split jadi per-citra (masing-masing bisa beda jumlah KP)
            # Gunakan mean pooling per citra
            kp_counts = [len(kp) for kp, _, _ in feature_results[method]]
            per_img = []
            idx = 0
            for count in kp_counts:
                if count > 0:
                    per_img.append(proj[idx:idx+count].mean(axis=0))
                    idx += count
                else:
                    per_img.append(np.zeros(n))

            per_img = np.array(per_img)

            # KNN accuracy
            knn_preds, true_labels = [], []
            for i in range(len(per_img)):
                mask = np.ones(len(per_img), dtype=bool)
                mask[i] = False
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(per_img[mask], labels_arr[mask])
                knn_preds.append(knn.predict(per_img[i:i+1])[0])
                true_labels.append(labels_arr[i])

            knn_acc = np.mean(np.array(knn_preds) == np.array(true_labels))
            print(f"  {n:<12} {var_exp:<18.1f} {compression:<14.1f} {knn_acc*100:.1f}%")

            method_pca.append(dict(n=n, var_exp=var_exp, compression=compression, knn_acc=knn_acc))

        pca_results[method] = method_pca

    return pca_results


# ─────────────────────────────────────────────
# BAGIAN 5: VISUALISASI & PLOTTING
# ─────────────────────────────────────────────

def plot_sample_images(images, labels):
    fig, axes = plt.subplots(5, 5, figsize=(14, 14))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle("Dataset Preview — 5 Kelas × 5 Variasi",
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    for r in range(5):
        for c in range(5):
            idx = r * 5 + c
            ax  = axes[r, c]
            ax.imshow(images[idx], cmap='plasma')
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(CLASS_NAMES[r], color='white', fontsize=11, fontweight='bold')
            var_labels = ['Referensi', 'Rotasi', 'Skala', 'Iluminasi', 'Oklusi']
            ax.set_title(var_labels[c], color='#adb5bd', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_keypoints_comparison(images, feature_results):
    methods = [m for m in feature_results if len(feature_results[m]) > 0]
    n_methods = len(methods)
    sample_indices = [0, 5, 10, 15, 20]  # satu per kelas

    fig, axes = plt.subplots(n_methods, 5, figsize=(16, n_methods * 3.2))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle("Perbandingan Keypoints: SIFT vs ORB (per kelas)",
                 color='white', fontsize=14, fontweight='bold')

    if n_methods == 1:
        axes = axes[np.newaxis, :]

    for r, method in enumerate(methods):
        detector = init_detectors()[method]
        for c, img_idx in enumerate(sample_indices):
            img  = images[img_idx]
            kp, des = detector.detectAndCompute(img, None)
            img_kp = cv2.drawKeypoints(img, kp, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_kp_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)
            ax = axes[r, c]
            ax.imshow(img_kp_rgb)
            ax.set_title(f"{CLASS_NAMES[c]}\n{len(kp)} kp", color='white', fontsize=9)
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(method, color='#00d9ff', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_matching_result(images, feature_results, labels):
    """Visualisasi BF matching untuk satu pasang citra tiap kelas."""
    method = 'SIFT' if 'SIFT' in feature_results else list(feature_results.keys())[0]
    detector = init_detectors()[method]

    fig, axes = plt.subplots(5, 1, figsize=(16, 25))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(f"Feature Matching ({method}) — Referensi vs Varian Rotasi",
                 color='white', fontsize=14, fontweight='bold')

    for cls_idx in range(5):
        ref_idx  = cls_idx * 5
        test_idx = ref_idx + 1
        img1 = images[ref_idx]
        img2 = images[test_idx]

        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        good = brute_force_match(des1, des2, method)

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good[:30], None, **draw_params)
        matched_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

        axes[cls_idx].imshow(matched_rgb)
        axes[cls_idx].set_title(
            f"{CLASS_NAMES[cls_idx].upper()}  —  {len(good)} matches ditemukan",
            color='white', fontsize=11, fontweight='bold')
        axes[cls_idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_bovw_accuracy(bovw_results):
    ks       = sorted(bovw_results.keys())
    knn_accs = [bovw_results[k]['knn_acc'] * 100 for k in ks]
    svm_accs = [bovw_results[k]['svm_acc'] * 100 for k in ks]
    times    = [bovw_results[k]['build_time'] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')

    # Akurasi
    ax1.set_facecolor('#161b22')
    ax1.plot(ks, knn_accs, 'o-', color='#58a6ff', linewidth=2.5, label='KNN', markersize=8)
    ax1.plot(ks, svm_accs, 's-', color='#f78166', linewidth=2.5, label='SVM', markersize=8)
    ax1.set_xlabel('Ukuran Vocabulary (k)', color='white')
    ax1.set_ylabel('Akurasi (%)', color='white')
    ax1.set_title('Akurasi Klasifikasi BoVW vs Vocab Size', color='white', fontweight='bold')
    ax1.legend(facecolor='#21262d', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.3, color='white')
    ax1.spines[:].set_color('#30363d')
    for spine in ax1.spines.values():
        spine.set_color('#30363d')

    # Build time
    ax2.set_facecolor('#161b22')
    bars = ax2.bar([str(k) for k in ks], times, color=['#3fb950', '#1f6feb', '#d29922', '#f78166'])
    ax2.set_xlabel('Ukuran Vocabulary (k)', color='white')
    ax2.set_ylabel('Waktu Build (detik)', color='white')
    ax2.set_title('Waktu Pembangunan Vocabulary', color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.3, color='white', axis='y')
    for spine in ax2.spines.values():
        spine.set_color('#30363d')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{t:.2f}s', ha='center', va='bottom', color='white', fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix, k):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', color='white')
    ax.set_yticklabels(CLASS_NAMES, color='white')
    ax.set_xlabel('Prediksi', color='white')
    ax.set_ylabel('Aktual', color='white')
    ax.set_title(f'Confusion Matrix BoVW (k={k}, SVM)', color='white', fontweight='bold')
    ax.tick_params(colors='white')

    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(conf_matrix[i, j]),
                    ha='center', va='center',
                    color='white' if conf_matrix[i, j] < thresh else 'black',
                    fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_pca_results(pca_results):
    methods = [m for m in pca_results if len(pca_results[m]) > 0]
    if not methods:
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5))
    fig.patch.set_facecolor('#0d1117')
    if len(methods) == 1:
        axes = [axes]

    colors_line = ['#58a6ff', '#3fb950', '#d29922']
    for ax, method, color in zip(axes, methods, colors_line):
        ax.set_facecolor('#161b22')
        data = pca_results[method]
        ns   = [d['n'] for d in data]
        accs = [d['knn_acc'] * 100 for d in data]
        vars = [d['var_exp'] for d in data]

        ax2 = ax.twinx()
        ax.plot(ns, accs, 'o-', color=color, linewidth=2.5, label='KNN Accuracy', markersize=8)
        ax2.plot(ns, vars, 's--', color='#f78166', linewidth=2, label='Var. Explained', markersize=7)

        ax.set_xlabel('Jumlah Komponen PCA', color='white')
        ax.set_ylabel('Akurasi (%)', color=color)
        ax2.set_ylabel('Variansi Dijelaskan (%)', color='#f78166')
        ax.set_title(f'PCA — {method}', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax2.tick_params(colors='#f78166')
        ax.grid(True, alpha=0.3, color='white')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, facecolor='#21262d', labelcolor='white', loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_pca_scatter(feature_results, labels_arr):
    """PCA 2D scatter untuk visualisasi pemisahan kelas."""
    method = 'SIFT' if 'SIFT' in feature_results else list(feature_results.keys())[0]
    all_des = collect_all_descriptors(feature_results, method)
    if all_des is None:
        return

    kp_counts = [len(kp) for kp, _, _ in feature_results[method]]
    proj, var_exp, _, _ = manual_pca(all_des, 2)

    per_img = []
    idx = 0
    for count in kp_counts:
        if count > 0:
            per_img.append(proj[idx:idx+count].mean(axis=0))
            idx += count
        else:
            per_img.append(np.zeros(2))
    per_img = np.array(per_img)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    palette = ['#f78166', '#79c0ff', '#56d364', '#d2a8ff', '#ffa657']
    markers = ['o', 's', '^', 'D', 'P']
    for cls_idx in range(5):
        mask = labels_arr == cls_idx
        ax.scatter(per_img[mask, 0], per_img[mask, 1],
                   c=palette[cls_idx], marker=markers[cls_idx],
                   s=120, alpha=0.85, label=CLASS_NAMES[cls_idx],
                   edgecolors='white', linewidths=0.5)

    ax.set_xlabel(f'PC1', color='white', fontsize=12)
    ax.set_ylabel(f'PC2', color='white', fontsize=12)
    ax.set_title(f'PCA 2D — Pemisahan Kelas ({method})\n'
                 f'Variansi Dijelaskan: {var_exp:.1f}%',
                 color='white', fontweight='bold', fontsize=12)
    ax.legend(facecolor='#21262d', labelcolor='white', markerscale=1.3)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.25, color='white')
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    plt.show()


def plot_comprehensive_comparison(extraction_summary, matching_summary, bovw_results):
    """Tabel + radar chart perbandingan komprehensif."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Rata-rata keypoints ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#161b22')
    methods = list(extraction_summary.keys())
    kp_vals = [extraction_summary[m]['avg_kp'] for m in methods]
    bars = ax1.bar(methods, kp_vals, color=['#58a6ff', '#3fb950', '#d29922'][:len(methods)])
    ax1.set_title('Rata-rata Keypoints', color='white', fontweight='bold')
    ax1.set_ylabel('Jumlah', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.3, axis='y', color='white')
    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    for b, v in zip(bars, kp_vals):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f'{v:.0f}', ha='center', va='bottom', color='white', fontsize=9)

    # ── Panel 2: Waktu ekstraksi ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#161b22')
    t_vals = [extraction_summary[m]['avg_time'] for m in methods]
    ax2.bar(methods, t_vals, color=['#f78166', '#d29922', '#79c0ff'][:len(methods)])
    ax2.set_title('Waktu Ekstraksi (ms)', color='white', fontweight='bold')
    ax2.set_ylabel('ms', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.3, axis='y', color='white')
    for spine in ax2.spines.values():
        spine.set_color('#30363d')

    # ── Panel 3: BF matches ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#161b22')
    ms = list(matching_summary.keys())
    bf_avg = [np.mean(matching_summary[m]['bf']) for m in ms]
    ax3.bar(ms, bf_avg, color=['#56d364', '#a371f7', '#ffa657'][:len(ms)])
    ax3.set_title('BF Matches (rata-rata)', color='white', fontweight='bold')
    ax3.set_ylabel('Good Matches', color='white')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.3, axis='y', color='white')
    for spine in ax3.spines.values():
        spine.set_color('#30363d')

    # ── Panel 4: BoVW akurasi bar ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#161b22')
    ks   = sorted(bovw_results.keys())
    knn_a = [bovw_results[k]['knn_acc']*100 for k in ks]
    svm_a = [bovw_results[k]['svm_acc']*100 for k in ks]
    x = np.arange(len(ks))
    w = 0.35
    ax4.bar(x - w/2, knn_a, w, label='KNN', color='#58a6ff')
    ax4.bar(x + w/2, svm_a, w, label='SVM', color='#f78166')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'k={k}' for k in ks], color='white')
    ax4.set_title('BoVW Accuracy', color='white', fontweight='bold')
    ax4.set_ylabel('%', color='white')
    ax4.legend(facecolor='#21262d', labelcolor='white')
    ax4.tick_params(colors='white')
    ax4.grid(True, alpha=0.3, axis='y', color='white')
    for spine in ax4.spines.values():
        spine.set_color('#30363d')

    # ── Panel 5: RANSAC inliers ───────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#161b22')
    ransac_avgs = [np.mean(matching_summary[m]['ransac']) for m in ms]
    ax5.bar(ms, ransac_avgs, color=['#d2a8ff', '#79c0ff', '#56d364'][:len(ms)])
    ax5.set_title('RANSAC Inliers (rata-rata)', color='white', fontweight='bold')
    ax5.set_ylabel('Inliers', color='white')
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.3, axis='y', color='white')
    for spine in ax5.spines.values():
        spine.set_color('#30363d')

    # ── Panel 6: Tabel ranking ────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.set_facecolor('#161b22')

    best_k   = max(bovw_results, key=lambda k: bovw_results[k]['svm_acc'])
    best_svm = bovw_results[best_k]['svm_acc'] * 100
    best_knn = bovw_results[best_k]['knn_acc'] * 100

    text_data = [
        ["Metrik", "Nilai Terbaik"],
        ["BoVW k terbaik", str(best_k)],
        [f"SVM Acc (k={best_k})", f"{best_svm:.1f}%"],
        [f"KNN Acc (k={best_k})", f"{best_knn:.1f}%"],
        ["Descriptor Dim", f"{list(extraction_summary.values())[0]['dim']}"],
        ["BF (Lowe's)", "75% threshold"],
    ]
    tbl = ax6.table(cellText=text_data, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#21262d' if r > 0 else '#1f6feb')
        cell.set_text_props(color='white', fontweight='bold' if r == 0 else 'normal')
        cell.set_edgecolor('#30363d')
    ax6.set_title('Ringkasan Hasil', color='white', fontweight='bold', pad=10)

    plt.suptitle("Analisis Komprehensif — Object Matching Pipeline",
                 color='white', fontsize=15, fontweight='bold')
    plt.show()


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*75)
    print("  PRAKTIKUM 12 — OBJECT MATCHING PIPELINE LENGKAP")
    print("  SIFT · ORB · BoVW · PCA · Evaluasi Komprehensif")
    print("="*75)

    # 0. Load dataset
    print("\n[FASE 0] Memuat dataset...")
    images, labels = load_dataset()
    labels_arr = np.array(labels)
    print(f"  Total citra  : {len(images)}")
    print(f"  Jumlah kelas : {len(np.unique(labels))}")
    print(f"  Kelas        : {CLASS_NAMES}")

    # 1. Init detectors
    detectors = init_detectors()
    print(f"\n  Detector aktif: {[k for k,v in detectors.items() if v is not None]}")

    # 2. Ekstraksi fitur
    print("\n[FASE 1] Ekstraksi fitur...")
    feature_results = extract_features(images, detectors)
    extraction_summary = summarize_extraction(feature_results, labels_arr)

    # 3. Feature matching
    print("\n[FASE 2] Feature matching...")
    matching_summary = run_matching_evaluation(feature_results, labels_arr)

    # 4. BoVW
    print("\n[FASE 3] Bag of Visual Words...")
    bovw_results, conf_matrices, histograms = run_bovw_pipeline(feature_results, labels_arr)

    # 5. PCA
    print("\n[FASE 4] PCA Dimensionality Reduction...")
    pca_results = run_pca_evaluation(feature_results, labels_arr)

    # 6. Simpan semua plot
    print("\n[FASE 5] Menyimpan visualisasi...")
    plot_sample_images(images, labels)
    plot_keypoints_comparison(images, feature_results)
    plot_matching_result(images, feature_results, labels)
    plot_bovw_accuracy(bovw_results)

    # Confusion matrix untuk k terbaik
    best_k = max(bovw_results, key=lambda k: bovw_results[k]['svm_acc'])
    plot_confusion_matrix(conf_matrices[best_k], best_k)

    plot_pca_results(pca_results)
    plot_pca_scatter(feature_results, labels_arr)
    plot_comprehensive_comparison(extraction_summary, matching_summary, bovw_results)

    # Final summary
    best_k   = max(bovw_results, key=lambda k: bovw_results[k]['svm_acc'])
    print("\n" + "="*75)
    print("  KESIMPULAN AKHIR")
    print("="*75)
    print(f"  ✔ Metode descriptor terbaik : SIFT (robust terhadap skala & rotasi)")
    print(f"  ✔ Ukuran vocabulary terbaik : k = {best_k}")
    print(f"  ✔ Akurasi SVM terbaik       : {bovw_results[best_k]['svm_acc']*100:.1f}%")
    print(f"  ✔ Akurasi KNN terbaik       : {bovw_results[best_k]['knn_acc']*100:.1f}%")
    print("="*75)


if __name__ == "__main__":
    main()