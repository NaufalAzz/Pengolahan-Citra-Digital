"""
=============================================================
TUGAS PRAKTIKUM 11 - Shape Analysis & Object Classification
=============================================================
Dataset: Buah (Apel, Pisang, Jeruk)
- 6 sampel per kelas
- Ekstraksi region properties, moments, boundary representation
- Fourier descriptors
- Klasifikasi k-NN + evaluasi akurasi
=============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# BAGIAN 0 - LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset(dataset_dir="dataset"):
    """
    Load gambar dari folder dataset/.
    Struktur: dataset/apel/, dataset/pisang/, dataset/jeruk/
    Return: list of (image_bgr, label_str, filename)
    """
    classes = ['apel', 'pisang', 'jeruk']
    data = []

    for label in classes:
        folder = os.path.join(dataset_dir, label)
        if not os.path.exists(folder):
            print(f"[WARNING] Folder tidak ditemukan: {folder}")
            continue
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"[WARNING] Gagal baca: {path}")
                continue
            data.append((img, label, fname))
        print(f"  Loaded {len(files)} sampel untuk kelas '{label}'")

    print(f"\nTotal sampel: {len(data)}")
    return data, classes


# ─────────────────────────────────────────────
# BAGIAN 1 - PREPROCESSING & SEGMENTASI
# ─────────────────────────────────────────────

def preprocess_image(img_bgr, target_size=(200, 200)):
    """
    Ubah gambar ke biner (foreground = objek buah).
    1. Resize ke ukuran seragam
    2. Grayscale → blur → threshold (Otsu)
    3. Morphological closing untuk mengisi lubang kecil
    4. Ambil contour terbesar (objek utama)
    Return: binary_mask, largest_contour, img_resized_bgr
    """
    img = cv2.resize(img_bgr, target_size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold Otsu
    _, binary = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # Ambil contour terbesar
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        # fallback: invert
        binary = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

    if not contours:
        # fallback akhir: buat mask putih
        binary = np.ones(target_size, dtype=np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

    largest = max(contours, key=cv2.contourArea)

    # Buat mask bersih dari contour terbesar
    mask = np.zeros(target_size, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask, largest, img


# ─────────────────────────────────────────────
# BAGIAN 2 - REGION PROPERTIES
# ─────────────────────────────────────────────

def extract_region_properties(contour, mask):
    """
    Ekstrak properti dasar region dari sebuah contour.
    Return: dict berisi semua properti numerik.
    """
    props = {}

    # --- Luas & Perimeter ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    props['area'] = area
    props['perimeter'] = perimeter

    # --- Moments & Centroid ---
    M = cv2.moments(contour)
    props['m00'] = M['m00']
    props['m10'] = M['m10']
    props['m01'] = M['m01']
    props['mu20'] = M['mu20']
    props['mu02'] = M['mu02']
    props['mu11'] = M['mu11']

    if M['m00'] != 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        cx, cy = 0, 0
    props['centroid_x'] = cx
    props['centroid_y'] = cy

    # --- Bounding Box ---
    x, y, w, h = cv2.boundingRect(contour)
    props['bb_x'] = x
    props['bb_y'] = y
    props['bb_w'] = w
    props['bb_h'] = h
    props['bb_area'] = w * h

    # --- Aspect Ratio ---
    props['aspect_ratio'] = w / h if h > 0 else 0

    # --- Extent (luas objek / luas bounding box) ---
    props['extent'] = area / (w * h) if (w * h) > 0 else 0

    # --- Convex Hull & Solidity ---
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    props['hull_area'] = hull_area
    props['solidity'] = area / hull_area if hull_area > 0 else 0

    # --- Compactness (circularitas) ---
    props['compactness'] = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else float('inf')

    # --- Equivalent Diameter ---
    props['equi_diameter'] = np.sqrt(4 * area / np.pi)

    # --- Hu Moments (7 invariant moments) ---
    hu = cv2.HuMoments(M).flatten()
    for i, hv in enumerate(hu):
        # Konversi ke log scale agar lebih stabil
        props[f'hu{i+1}'] = -np.sign(hv) * np.log10(abs(hv) + 1e-10)

    return props


# ─────────────────────────────────────────────
# BAGIAN 3 - CHAIN CODE
# ─────────────────────────────────────────────

def freeman_chain_code(points, mode=8):
    """
    Hitung Freeman Chain Code dari list titik kontur.
    mode = 4 (4-arah) atau 8 (8-arah)
    """
    if mode == 4:
        directions = [(1,0),(0,1),(-1,0),(0,-1)]
    else:
        directions = [(1,0),(1,1),(0,1),(-1,1),
                      (-1,0),(-1,-1),(0,-1),(1,-1)]

    chain = []
    for i in range(len(points)):
        curr = points[i]
        nxt  = points[(i+1) % len(points)]
        dx = int(nxt[0]) - int(curr[0])
        dy = int(nxt[1]) - int(curr[1])

        # Normalisasi dx,dy ke -1,0,1 untuk mencocokkan arah
        if dx != 0: dx = dx // abs(dx)
        if dy != 0: dy = dy // abs(dy)

        for code, (ddx, ddy) in enumerate(directions):
            if dx == ddx and dy == ddy:
                chain.append(code)
                break

    return chain


def normalize_chain_code(chain_code):
    """
    Normalisasi chain code:
    - Ubah ke bentuk perbedaan (difference chain code) → invariant terhadap rotasi
    - Mulai dari nilai terkecil (rotational normalization)
    """
    if not chain_code:
        return chain_code

    n = len(chain_code)
    mod = 8  # untuk 8-directional

    # Difference chain code
    diff = [(chain_code[(i+1) % n] - chain_code[i]) % mod for i in range(n)]

    # Rotational normalization: mulai dari indeks yang hasilkan urutan terkecil
    rotations = [diff[i:] + diff[:i] for i in range(n)]
    normalized = min(rotations)

    return normalized


def polygonal_approx(contour, epsilon_factor=0.02):
    """
    Douglas-Peucker polygonal approximation.
    epsilon_factor: fraksi dari perimeter sebagai toleransi
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, epsilon, True)
    return approx


# ─────────────────────────────────────────────
# BAGIAN 4 - FOURIER DESCRIPTORS
# ─────────────────────────────────────────────

def compute_fourier_descriptors(contour, num_points=128):
    """
    Hitung Fourier Descriptors dari contour.
    1. Resample contour ke num_points titik seragam
    2. Representasikan sebagai bilangan kompleks z = x + iy
    3. FFT → magnitude spectrum (normalized)
    Return: fd (raw), fd_norm (normalized)
    """
    # Ambil koordinat contour
    pts = contour.reshape(-1, 2).astype(float)

    # Resample seragam
    n = len(pts)
    if n < 4:
        pts = np.vstack([pts, pts[:4-n]])  # pad jika terlalu sedikit
    indices = np.linspace(0, n-1, num_points).astype(int)
    pts = pts[indices]

    # Representasi kompleks
    z = pts[:, 0] + 1j * pts[:, 1]

    # FFT
    fd = np.fft.fft(z)
    mag = np.abs(fd)

    # Normalisasi: bagi dengan DC component (invariant to translation & scale)
    if mag[1] != 0:
        fd_norm = mag / mag[1]
    else:
        fd_norm = mag

    return fd, fd_norm


def reconstruct_from_fourier(fd, num_coeffs):
    """
    Rekonstruksi bentuk dari sejumlah koefisien Fourier.
    """
    fd_recon = np.zeros_like(fd)
    fd_recon[:num_coeffs] = fd[:num_coeffs]
    fd_recon[-num_coeffs:] = fd[-num_coeffs:]

    z_recon = np.fft.ifft(fd_recon)
    return z_recon.real, z_recon.imag


def fourier_feature_vector(fd_norm, num_descriptors=20):
    """
    Ambil num_descriptors pertama sebagai feature vector.
    (Skip DC component, mulai dari indeks 1)
    """
    return fd_norm[1:num_descriptors+1]


# ─────────────────────────────────────────────
# BAGIAN 5 - FEATURE EXTRACTION PIPELINE
# ─────────────────────────────────────────────

def extract_all_features(img_bgr):
    """
    Pipeline lengkap: gambar → semua fitur numerik.
    Return: dict fitur + artefak visualisasi
    """
    mask, contour, img_resized = preprocess_image(img_bgr)

    # --- Region properties ---
    props = extract_region_properties(contour, mask)

    # --- Contour simple (tanpa duplikat) ---
    pts_simple = []
    for i in range(len(contour)):
        if i == 0 or not np.array_equal(contour[i][0], contour[i-1][0]):
            pts_simple.append(contour[i][0])
    pts_simple = np.array(pts_simple)

    # --- Chain codes ---
    cc4 = freeman_chain_code(pts_simple, mode=4)
    cc8 = freeman_chain_code(pts_simple, mode=8)
    cc8_norm = normalize_chain_code(cc8)

    # Fitur dari chain code: frekuensi tiap arah
    cc4_freq = [cc4.count(d) / max(len(cc4), 1) for d in range(4)]
    cc8_freq = [cc8.count(d) / max(len(cc8), 1) for d in range(8)]

    # --- Polygonal approximation ---
    poly = polygonal_approx(contour)
    props['num_poly_vertices'] = len(poly)

    # --- Fourier descriptors ---
    fd, fd_norm = compute_fourier_descriptors(contour, num_points=128)
    fd_feat = fourier_feature_vector(fd_norm, num_descriptors=20)

    # Gabungkan semua fitur
    features = {
        # Region
        'area':          props['area'],
        'perimeter':     props['perimeter'],
        'compactness':   props['compactness'],
        'aspect_ratio':  props['aspect_ratio'],
        'extent':        props['extent'],
        'solidity':      props['solidity'],
        'equi_diameter': props['equi_diameter'],
        'num_vertices':  props['num_poly_vertices'],

        # Moments
        'm00': props['m00'],
        'm10': props['m10'],
        'm01': props['m01'],
        'mu20': props['mu20'],
        'mu02': props['mu02'],
        'mu11': props['mu11'],

        # Hu moments (7 invariant)
        **{f'hu{i+1}': props[f'hu{i+1}'] for i in range(7)},

        # Chain code frequencies
        **{f'cc4_d{d}': cc4_freq[d] for d in range(4)},
        **{f'cc8_d{d}': cc8_freq[d] for d in range(8)},

        # Fourier (20 descriptors)
        **{f'fd{i+1}': fd_feat[i] for i in range(len(fd_feat))},
    }

    artifacts = {
        'mask': mask,
        'contour': contour,
        'img_resized': img_resized,
        'pts_simple': pts_simple,
        'cc8': cc8,
        'poly': poly,
        'fd': fd,
        'fd_norm': fd_norm,
        'props': props,
    }

    return features, artifacts


# ─────────────────────────────────────────────
# BAGIAN 6 - KLASIFIKASI k-NN
# ─────────────────────────────────────────────

FEATURE_GROUPS = {
    'Region': ['area', 'perimeter', 'compactness', 'aspect_ratio',
               'extent', 'solidity', 'equi_diameter', 'num_vertices'],

    'Moments': ['m00', 'mu20', 'mu02', 'mu11',
                'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7'],

    'ChainCode': [f'cc8_d{d}' for d in range(8)],

    'Fourier':  [f'fd{i+1}' for i in range(20)],

    'Best_Region': ['compactness', 'aspect_ratio', 'solidity'],

    'Best_Moments': ['hu1', 'hu2', 'hu3'],

    'Best_Fourier': ['fd1', 'fd2', 'fd3'],

    'Combined_Best': ['compactness', 'aspect_ratio', 'solidity',
                      'hu1', 'hu2', 'hu3',
                      'fd1', 'fd2', 'fd3'],

    'All': None,  # None = semua fitur
}


def build_feature_matrix(all_features, feature_keys=None):
    """
    Bangun matrix X dari list dict fitur.
    feature_keys = None → gunakan semua fitur numerik
    """
    if feature_keys is None:
        feature_keys = list(all_features[0].keys())

    X = []
    for feat in all_features:
        row = [feat.get(k, 0) for k in feature_keys]
        X.append(row)

    X = np.array(X, dtype=float)

    # Ganti NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_keys


def evaluate_knn(X, y, k=3):
    """
    Evaluasi k-NN dengan Leave-One-Out cross-validation.
    Return: akurasi rata-rata
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    loo = LeaveOneOut()

    scores = cross_val_score(knn, X_scaled, y, cv=loo, scoring='accuracy')
    return scores.mean()


def classify_all_groups(all_features, labels, k=3):
    """
    Evaluasi semua kombinasi grup fitur.
    Return: dict { group_name: accuracy }
    """
    results = {}

    print("\n" + "=" * 60)
    print("EVALUASI k-NN (Leave-One-Out) | k =", k)
    print("=" * 60)
    print(f"{'Grup Fitur':<20} {'Jumlah Fitur':<15} {'Akurasi':<10}")
    print("-" * 60)

    all_keys = list(all_features[0].keys())

    for group_name, keys in FEATURE_GROUPS.items():
        if keys is None:
            keys = all_keys

        # Pastikan semua key ada
        keys = [k for k in keys if k in all_keys]
        if not keys:
            continue

        X, used_keys = build_feature_matrix(all_features, keys)
        y = np.array(labels)

        acc = evaluate_knn(X, y, k=k)
        results[group_name] = acc

        print(f"{group_name:<20} {len(used_keys):<15} {acc*100:.1f}%")

    best_group = max(results, key=results.get)
    print("-" * 60)
    print(f"✓ Grup terbaik: {best_group} ({results[best_group]*100:.1f}%)")

    return results, best_group


def full_confusion_matrix(all_features, labels, best_group, classes, k=3):
    """
    Tampilkan confusion matrix dan classification report untuk grup terbaik.
    """
    keys = FEATURE_GROUPS[best_group]
    if keys is None:
        keys = list(all_features[0].keys())
    keys = [k for k in keys if k in all_features[0]]

    X, _ = build_feature_matrix(all_features, keys)
    y = np.array(labels)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k)
    loo = LeaveOneOut()

    y_pred = []
    for train_idx, test_idx in loo.split(X_sc):
        knn.fit(X_sc[train_idx], y[train_idx])
        y_pred.append(knn.predict(X_sc[test_idx])[0])

    y_pred = np.array(y_pred)

    print(f"\nClassification Report (grup: {best_group})")
    print("=" * 50)
    print(classification_report(y, y_pred, target_names=classes))

    cm = confusion_matrix(y, y_pred, labels=classes)
    return cm, y_pred


# ─────────────────────────────────────────────
# BAGIAN 7 - VISUALISASI
# ─────────────────────────────────────────────

COLORS = {
    'apel':   '#e74c3c',
    'pisang': '#f1c40f',
    'jeruk':  '#e67e22',
}

def visualize_dataset_overview(data, classes, max_per_class=6):
    """Plot galeri sampel dataset."""
    n_rows = len(classes)
    n_cols = max_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    fig.suptitle("Dataset Overview: Buah (Apel, Pisang, Jeruk)", fontsize=14, fontweight='bold')

    for r, cls in enumerate(classes):
        cls_imgs = [(img, lbl, fn) for img, lbl, fn in data if lbl == cls]
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(cls_imgs):
                img, _, fn = cls_imgs[c]
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(fn, fontsize=7)
            else:
                ax.axis('off')
                continue
            if c == 0:
                ax.set_ylabel(cls.capitalize(), fontsize=10,
                              color=COLORS.get(cls, 'black'), fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('output_1_dataset_overview.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_preprocessing(data, classes):
    """Plot proses preprocessing untuk 1 sampel per kelas."""
    n_cols = 4
    n_rows = len(classes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3))
    fig.suptitle("Preprocessing Pipeline (per kelas)", fontsize=13, fontweight='bold')
    col_titles = ['Original', 'Mask', 'Kontur + Centroid', 'Convex Hull + Poly']

    for r, cls in enumerate(classes):
        sample = next((img for img, lbl, _ in data if lbl == cls), None)
        if sample is None:
            continue

        mask, contour, img_res = preprocess_image(sample)
        _, arts = extract_all_features(sample)

        # Centroid
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00']) if M['m00'] else 100
        cy = int(M['m01']/M['m00']) if M['m00'] else 100

        # Convex Hull
        hull = cv2.convexHull(contour)

        for c, title in enumerate(col_titles):
            ax = axes[r, c]
            ax.axis('off')
            if r == 0:
                ax.set_title(title, fontsize=9, fontweight='bold')

            if c == 0:
                ax.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
                ax.set_ylabel(cls.capitalize(), fontsize=10,
                              color=COLORS.get(cls,'black'), fontweight='bold')

            elif c == 1:
                ax.imshow(mask, cmap='gray')

            elif c == 2:
                vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(vis, [contour], -1, (0,220,0), 2)
                cv2.circle(vis, (cx, cy), 5, (255,0,0), -1)
                ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

            elif c == 3:
                vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(vis, [hull], -1, (0,0,255), 2)
                cv2.drawContours(vis, [arts['poly']], -1, (255,165,0), 2)
                ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig('output_2_preprocessing.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_region_properties(all_features, labels, classes):
    """Scatter plot beberapa properti region."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Region Properties per Kelas", fontsize=13, fontweight='bold')

    pairs = [
        ('compactness', 'solidity'),
        ('aspect_ratio', 'extent'),
        ('equi_diameter', 'hu1'),
    ]

    for ax, (fx, fy) in zip(axes, pairs):
        for cls in classes:
            xs = [f[fx] for f, lbl in zip(all_features, labels) if lbl == cls]
            ys = [f[fy] for f, lbl in zip(all_features, labels) if lbl == cls]
            ax.scatter(xs, ys, label=cls, color=COLORS.get(cls,'gray'),
                       s=80, edgecolors='black', linewidths=0.5, alpha=0.85)
        ax.set_xlabel(fx, fontsize=9)
        ax.set_ylabel(fy, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_3_region_properties.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_hu_moments(all_features, labels, classes):
    """Radar-style bar chart Hu moments."""
    fig, axes = plt.subplots(1, len(classes), figsize=(14, 4))
    fig.suptitle("Hu Moments (rata-rata per kelas)", fontsize=13, fontweight='bold')

    hu_keys = [f'hu{i+1}' for i in range(7)]

    for ax, cls in zip(axes, classes):
        vals = np.array([
            [f[k] for k in hu_keys]
            for f, lbl in zip(all_features, labels) if lbl == cls
        ])
        mean_vals = vals.mean(axis=0) if len(vals) > 0 else np.zeros(7)
        ax.bar(range(1, 8), mean_vals, color=COLORS.get(cls, 'gray'),
               edgecolor='black', linewidth=0.6)
        ax.set_title(cls.capitalize(), fontsize=10, color=COLORS.get(cls,'black'),
                     fontweight='bold')
        ax.set_xlabel('Hu Moment')
        ax.set_ylabel('Nilai (log scale)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output_4_hu_moments.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_chain_code(data, classes):
    """Visualisasi chain code untuk 1 sampel per kelas."""
    fig, axes = plt.subplots(len(classes), 3, figsize=(12, len(classes)*3.5))
    fig.suptitle("Chain Code Analysis (8-directional)", fontsize=13, fontweight='bold')

    col_t = ['Kontur', 'Chain Code Arrows', 'Frekuensi Arah']

    for r, cls in enumerate(classes):
        sample = next((img for img, lbl, _ in data if lbl == cls), None)
        if sample is None:
            continue

        mask, contour, _ = preprocess_image(sample)

        pts = []
        for i in range(len(contour)):
            if i == 0 or not np.array_equal(contour[i][0], contour[i-1][0]):
                pts.append(contour[i][0])
        pts = np.array(pts)

        cc8 = freeman_chain_code(pts, mode=8)

        for c in range(3):
            ax = axes[r, c]
            ax.axis('off')
            if r == 0:
                ax.set_title(col_t[c], fontsize=9, fontweight='bold')

            if c == 0:
                ax.imshow(mask, cmap='gray')
                ax.set_ylabel(cls.capitalize(), fontsize=10,
                              color=COLORS.get(cls,'black'), fontweight='bold')
                ax.axis('off')

            elif c == 1:
                disp = np.zeros((200, 200), dtype=np.uint8)
                step = max(1, len(pts)//80)
                for i in range(0, len(pts), step):
                    p = pts[i]
                    cv2.circle(disp, tuple(p), 1, 200, -1)
                    if i < len(cc8):
                        dirs8 = [(1,0),(1,1),(0,1),(-1,1),
                                 (-1,0),(-1,-1),(0,-1),(1,-1)]
                        dx, dy = dirs8[cc8[i]]
                        ep = (int(p[0]+dx*4), int(p[1]+dy*4))
                        cv2.arrowedLine(disp, tuple(p), ep, 255, 1, tipLength=0.4)
                ax.imshow(disp, cmap='hot')
                ax.axis('off')

            elif c == 2:
                ax.axis('on')
                freqs = [cc8.count(d) for d in range(8)]
                dir_names = ['E','SE','S','SW','W','NW','N','NE']
                ax.bar(dir_names, freqs, color=COLORS.get(cls,'gray'),
                       edgecolor='black', linewidth=0.5)
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output_5_chain_code.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_fourier(data, classes):
    """Rekonstruksi bentuk dari Fourier descriptors."""
    fig, axes = plt.subplots(len(classes), 5, figsize=(16, len(classes)*3))
    fig.suptitle("Fourier Descriptors & Rekonstruksi Bentuk", fontsize=13, fontweight='bold')

    col_t = ['Original', 'Spektrum Fourier', 'Rekon (5 koef)', 'Rekon (10 koef)', 'Rekon (20 koef)']

    for r, cls in enumerate(classes):
        sample = next((img for img, lbl, _ in data if lbl == cls), None)
        if sample is None:
            continue

        mask, contour, _ = preprocess_image(sample)
        fd, fd_norm = compute_fourier_descriptors(contour, num_points=128)

        for c in range(5):
            ax = axes[r, c]
            ax.axis('off')
            if r == 0:
                ax.set_title(col_t[c], fontsize=9, fontweight='bold')

            if c == 0:
                ax.imshow(mask, cmap='gray')
                ax.set_ylabel(cls.capitalize(), fontsize=10,
                              color=COLORS.get(cls,'black'), fontweight='bold')
                ax.axis('off')

            elif c == 1:
                ax.axis('on')
                n = len(fd_norm)
                ax.plot(fd_norm[:n//2], color=COLORS.get(cls,'gray'), linewidth=1.5)
                ax.set_xlabel('Frekuensi')
                ax.set_ylabel('Magnitude')
                ax.grid(True, alpha=0.3)

            else:
                num_c = [5, 10, 20][c-2]
                xr, yr = reconstruct_from_fourier(fd, num_c)
                recon = np.zeros((200, 200), dtype=np.uint8)
                pts_r = np.column_stack([xr.astype(int), yr.astype(int)])
                pts_r = np.clip(pts_r, 0, 199)
                for i in range(len(pts_r)):
                    s = tuple(pts_r[i])
                    e = tuple(pts_r[(i+1) % len(pts_r)])
                    cv2.line(recon, s, e, 255, 1)
                ax.imshow(recon, cmap='gray')
                ax.set_title(f'{num_c} koef', fontsize=8)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig('output_6_fourier.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_accuracy_comparison(results):
    """Bar chart perbandingan akurasi antar grup fitur."""
    fig, ax = plt.subplots(figsize=(12, 5))

    groups = list(results.keys())
    accs   = [results[g]*100 for g in groups]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(groups)))

    bars = ax.bar(groups, accs, color=colors, edgecolor='black', linewidth=0.7)
    ax.axhline(y=100/3, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label='Random baseline (33.3%)')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Akurasi (%)', fontsize=10)
    ax.set_title('Perbandingan Akurasi k-NN (LOO-CV) per Grup Fitur', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output_7_accuracy_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualize_confusion_matrix(cm, classes, best_group):
    """Heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_yticklabels([c.capitalize() for c in classes], fontsize=10)
    ax.set_xlabel('Prediksi', fontsize=11)
    ax.set_ylabel('Label Asli', fontsize=11)
    ax.set_title(f'Confusion Matrix\n(Grup: {best_group})', fontsize=11, fontweight='bold')

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.tight_layout()
    plt.savefig('output_8_confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# BAGIAN 8 - PRINT TABEL HASIL
# ─────────────────────────────────────────────

def print_region_table(all_features, labels, filenames):
    print("\n" + "=" * 110)
    print("TABEL REGION PROPERTIES")
    print("=" * 110)
    header = f"{'File':<15} {'Kelas':<8} {'Luas':>8} {'Perim':>8} {'Compact':>9} {'AspRatio':>9} {'Extent':>8} {'Solid':>8} {'EquiDiam':>9}"
    print(header)
    print("-" * 110)
    for feat, lbl, fn in zip(all_features, labels, filenames):
        print(f"{fn:<15} {lbl:<8} {feat['area']:>8.0f} {feat['perimeter']:>8.0f} "
              f"{feat['compactness']:>9.3f} {feat['aspect_ratio']:>9.3f} "
              f"{feat['extent']:>8.3f} {feat['solidity']:>8.3f} {feat['equi_diameter']:>9.2f}")


def print_hu_table(all_features, labels, filenames):
    print("\n" + "=" * 100)
    print("TABEL HU MOMENTS (log scale)")
    print("=" * 100)
    header = f"{'File':<15} {'Kelas':<8} " + " ".join([f"{'Hu'+str(i):>9}" for i in range(1,8)])
    print(header)
    print("-" * 100)
    for feat, lbl, fn in zip(all_features, labels, filenames):
        hu_str = " ".join([f"{feat[f'hu{i}']:>9.4f}" for i in range(1,8)])
        print(f"{fn:<15} {lbl:<8} {hu_str}")


def print_fourier_table(all_features, labels, filenames):
    print("\n" + "=" * 80)
    print("TABEL FOURIER DESCRIPTORS (fd1 - fd5)")
    print("=" * 80)
    header = f"{'File':<15} {'Kelas':<8} " + " ".join([f"{'fd'+str(i):>9}" for i in range(1,6)])
    print(header)
    print("-" * 80)
    for feat, lbl, fn in zip(all_features, labels, filenames):
        fd_str = " ".join([f"{feat[f'fd{i}']:>9.4f}" for i in range(1,6)])
        print(f"{fn:<15} {lbl:<8} {fd_str}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TUGAS PRAKTIKUM 11 - Shape Analysis & Fruit Classification")
    print("=" * 70)

    # 0. Load dataset
    print("\n[1/8] Loading dataset...")
    data, classes = load_dataset("dataset")

    if not data:
        print("[ERROR] Dataset kosong! Pastikan folder 'dataset/' ada dan berisi gambar.")
        return

    # 1. Dataset overview
    print("\n[2/8] Visualisasi dataset...")
    visualize_dataset_overview(data, classes)

    # 2. Preprocessing
    print("\n[3/8] Preprocessing pipeline...")
    visualize_preprocessing(data, classes)

    # 3. Ekstraksi semua fitur
    print("\n[4/8] Ekstraksi fitur dari semua sampel...")
    all_features = []
    labels = []
    filenames = []

    for img, lbl, fn in data:
        try:
            feat, arts = extract_all_features(img)
            all_features.append(feat)
            labels.append(lbl)
            filenames.append(fn)
        except Exception as e:
            print(f"  [SKIP] {fn}: {e}")

    print(f"  Berhasil ekstrak {len(all_features)} sampel")

    # 4. Print tabel hasil
    print_region_table(all_features, labels, filenames)
    print_hu_table(all_features, labels, filenames)
    print_fourier_table(all_features, labels, filenames)

    # 5. Visualisasi properti region
    print("\n[5/8] Visualisasi region properties...")
    visualize_region_properties(all_features, labels, classes)
    visualize_hu_moments(all_features, labels, classes)

    # 6. Chain code & Fourier
    print("\n[6/8] Visualisasi chain code & Fourier descriptors...")
    visualize_chain_code(data, classes)
    visualize_fourier(data, classes)

    # 7. Klasifikasi k-NN
    print("\n[7/8] Klasifikasi k-NN (k=3, Leave-One-Out CV)...")
    results, best_group = classify_all_groups(all_features, labels, k=3)
    visualize_accuracy_comparison(results)

    # 8. Confusion matrix grup terbaik
    print("\n[8/8] Confusion matrix (grup terbaik)...")
    cm, y_pred = full_confusion_matrix(all_features, labels, best_group, classes, k=3)
    visualize_confusion_matrix(cm, classes, best_group)

    # Final summary
    print("\n" + "=" * 70)
    print("RINGKASAN HASIL")
    print("=" * 70)
    print(f"Total sampel      : {len(all_features)}")
    print(f"Jumlah kelas      : {len(classes)} ({', '.join(classes)})")
    print(f"Jumlah fitur total: {len(all_features[0])}")
    print()
    print("Akurasi per grup fitur:")
    for g, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = '█' * int(acc * 20)
        print(f"  {g:<20} {bar:<20} {acc*100:.1f}%")

    print(f"\n→ Grup fitur terbaik  : {best_group}")
    print(f"→ Akurasi tertinggi   : {results[best_group]*100:.1f}%")
    print("\nFile output tersimpan: output_1 s/d output_8 .png")
    print("=" * 70)


if __name__ == "__main__":
    main()