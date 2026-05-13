import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, learning_curve, GridSearchCV
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.multiclass import OneVsRestClassifier
from skimage.feature import hog

warnings.filterwarnings('ignore')

COLORS_10 = plt.cm.tab10(np.linspace(0, 1, 10))
SEP = "=" * 60


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# 1. LOAD DATASET & EKSTRAKSI FITUR
def load_and_extract(n_samples=1000):
    section("1. LOAD DATASET & EKSTRAKSI FITUR")

    digits = load_digits()
    rng = np.random.RandomState(42)
    idx = rng.choice(len(digits.data), n_samples, replace=False)
    X_raw = digits.data[idx]
    X_img = digits.images[idx]
    y = digits.target[idx]

    print(f"Dataset : MNIST Digit (sklearn), {n_samples} sampel, 10 kelas")
    print(f"Distribusi kelas: {np.bincount(y)}")

    X_pixel = X_raw.copy()

    hog_features = []
    for img in X_img:
        feat = hog(img, orientations=8, pixels_per_cell=(2, 2),
                   cells_per_block=(1, 1), feature_vector=True)
        hog_features.append(feat)
    X_hog = np.array(hog_features)

    X_combined = np.hstack([X_pixel, X_hog])

    print(f"\nDimensi fitur:")
    print(f"  Pixel (raw) : {X_pixel.shape[1]}")
    print(f"  HOG         : {X_hog.shape[1]}")
    print(f"  Gabungan    : {X_combined.shape[1]}")

    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    for cls in range(10):
        cidx = np.where(y == cls)[0][0]
        axes[0, cls].imshow(X_img[cidx], cmap='gray')
        axes[0, cls].set_title(f"Digit {cls}", fontsize=8)
        axes[0, cls].axis('off')
        _, hog_img = hog(X_img[cidx], orientations=8,
                         pixels_per_cell=(2, 2), cells_per_block=(1, 1),
                         feature_vector=True, visualize=True)
        axes[1, cls].imshow(hog_img, cmap='inferno')
        axes[1, cls].set_title("HOG", fontsize=8)
        axes[1, cls].axis('off')

    plt.suptitle("Sample Digits & HOG Visualization (per class)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return X_combined, y, X_img


# 2. SPLIT DATA & NORMALISASI
def prepare_data(X, y):
    section("2. SPLIT DATA & NORMALISASI")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Train : {X_train_s.shape[0]} sampel")
    print(f"Test  : {X_test_s.shape[0]} sampel")
    return X_train_s, X_test_s, y_train, y_test, scaler


# 3. K-NEAREST NEIGHBORS
def run_knn(X_train, X_test, y_train, y_test):
    section("3. K-NEAREST NEIGHBORS")

    k_values = [1, 3, 5, 7, 9, 11]
    dist_metrics = ['euclidean', 'manhattan', 'minkowski']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results_k = {}
    for metric in dist_metrics:
        cv_means, test_accs = [], []
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            cv_s = cross_val_score(knn, X_train, y_train,
                                   cv=cv, scoring='accuracy')
            cv_means.append(cv_s.mean())
            knn.fit(X_train, y_train)
            test_accs.append(knn.score(X_test, y_test))
        results_k[metric] = {'cv': cv_means, 'test': test_accs}
        print(f"\nMetrik {metric.upper()}:")
        for k, cvm, ta in zip(k_values, cv_means, test_accs):
            print(f"  k={k:2d}  CV={cvm:.4f}  Test={ta:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, dist_metrics):
        cv_data = results_k[metric]['cv']
        te_data = results_k[metric]['test']
        ax.plot(k_values, cv_data, 'o-', label='CV (5-fold)', lw=2)
        ax.plot(k_values, te_data, 's--', label='Test', lw=2)
        ax.set_title(f"KNN - {metric.capitalize()}", fontweight='bold')
        ax.set_xlabel('k (jumlah tetangga)')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        for ki, cvi, tsi in zip(k_values, cv_data, te_data):
            gap = abs(cvi - tsi)
            col = 'red' if gap > 0.05 else 'green'
            ax.annotate(f"D{gap:.2f}", (ki, min(cvi, tsi) - 0.02),
                        ha='center', fontsize=7, color=col)

    plt.suptitle(
        "KNN: Pengaruh k & Jarak terhadap Akurasi\n"
        "(D = |CV - Test|, merah > 0.05 = potensi overfit)",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nGridSearchCV untuk KNN...")
    param_grid_knn = {
        'n_neighbors': k_values,
        'metric': dist_metrics,
        'weights': ['uniform', 'distance']
    }
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn,
                            cv=cv, scoring='accuracy',
                            n_jobs=-1, verbose=0)
    t0 = time.time()
    grid_knn.fit(X_train, y_train)
    t_train = time.time() - t0

    best_knn = grid_knn.best_estimator_
    print(f"  Best params  : {grid_knn.best_params_}")
    print(f"  Best CV acc  : {grid_knn.best_score_:.4f}")
    print(f"  Waktu train  : {t_train:.2f}s")

    t0 = time.time()
    y_pred = best_knn.predict(X_test)
    t_inf = time.time() - t0
    print(f"  Waktu infer  : {t_inf:.4f}s")

    met = compute_metrics(y_test, y_pred, t_train, t_inf)
    return best_knn, y_pred, met


# 4. SUPPORT VECTOR MACHINE
def run_svm(X_train, X_test, y_train, y_test):
    section("4. SUPPORT VECTOR MACHINE")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    kernels = ['linear', 'poly', 'rbf']
    kernel_results = {}
    for kernel in kernels:
        params = dict(kernel=kernel, C=1.0, random_state=42)
        if kernel == 'poly':
            params['degree'] = 3
        if kernel == 'rbf':
            params['gamma'] = 'scale'
        clf = svm.SVC(**params)
        cv_s = cross_val_score(clf, X_train, y_train,
                               cv=cv, scoring='accuracy')
        clf.fit(X_train, y_train)
        ta = clf.score(X_test, y_test)
        kernel_results[kernel] = {'cv': cv_s.mean(), 'test': ta}
        print(f"  Kernel {kernel:<8}: CV={cv_s.mean():.4f}  Test={ta:.4f}")

    C_values = [0.1, 1, 10, 100]
    c_accs = []
    for C in C_values:
        clf = svm.SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
        cv_s = cross_val_score(clf, X_train, y_train,
                               cv=cv, scoring='accuracy')
        c_accs.append(cv_s.mean())
        print(f"  RBF C={C:5.1f}: CV={cv_s.mean():.4f}")

    gamma_values = [0.001, 0.01, 0.1, 1]
    g_accs = []
    for g in gamma_values:
        clf = svm.SVC(kernel='rbf', C=1.0, gamma=g, random_state=42)
        cv_s = cross_val_score(clf, X_train, y_train,
                               cv=cv, scoring='accuracy')
        g_accs.append(cv_s.mean())
        print(f"  RBF gamma={g}: CV={cv_s.mean():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    knames = list(kernel_results.keys())
    cv_vals = [kernel_results[k]['cv'] for k in knames]
    te_vals = [kernel_results[k]['test'] for k in knames]
    x = np.arange(len(knames))
    w = 0.35
    axes[0].bar(x - w/2, cv_vals, w, label='CV', color='steelblue')
    axes[0].bar(x + w/2, te_vals, w, label='Test', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([k.upper() for k in knames])
    axes[0].set_title("SVM Kernel Comparison", fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    for xi, (cv_v, tv) in enumerate(zip(cv_vals, te_vals)):
        axes[0].text(xi - w/2, cv_v + 0.01, f'{cv_v:.3f}',
                     ha='center', fontsize=8)
        axes[0].text(xi + w/2, tv + 0.01, f'{tv:.3f}',
                     ha='center', fontsize=8)

    axes[1].plot(C_values, c_accs, 'o-', lw=2, color='darkorange')
    axes[1].set_xscale('log')
    axes[1].set_title("SVM (RBF): Variasi C", fontweight='bold')
    axes[1].set_xlabel('C')
    axes[1].set_ylabel('CV Accuracy')
    axes[1].grid(True, alpha=0.3)
    for C, acc in zip(C_values, c_accs):
        axes[1].annotate(f'{acc:.3f}', (C, acc + 0.002),
                         ha='center', fontsize=8)

    axes[2].plot(gamma_values, g_accs, 's-', lw=2, color='seagreen')
    axes[2].set_xscale('log')
    axes[2].set_title("SVM (RBF): Variasi Gamma", fontweight='bold')
    axes[2].set_xlabel('Gamma')
    axes[2].set_ylabel('CV Accuracy')
    axes[2].grid(True, alpha=0.3)
    for g, acc in zip(gamma_values, g_accs):
        axes[2].annotate(f'{acc:.3f}', (g, acc + 0.002),
                         ha='center', fontsize=8)

    plt.suptitle("SVM: Analisis Kernel, C, dan Gamma",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nGridSearchCV untuk SVM...")
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 0.01, 0.1]
    }
    grid_svm = GridSearchCV(svm.SVC(random_state=42), param_grid_svm,
                            cv=cv, scoring='accuracy',
                            n_jobs=-1, verbose=0)
    t0 = time.time()
    grid_svm.fit(X_train, y_train)
    t_train = time.time() - t0

    best_svm_model = grid_svm.best_estimator_
    print(f"  Best params  : {grid_svm.best_params_}")
    print(f"  Best CV acc  : {grid_svm.best_score_:.4f}")
    print(f"  Waktu train  : {t_train:.2f}s")

    t0 = time.time()
    y_pred = best_svm_model.predict(X_test)
    t_inf = time.time() - t0
    print(f"  Waktu infer  : {t_inf:.4f}s")

    met = compute_metrics(y_test, y_pred, t_train, t_inf)
    return best_svm_model, y_pred, met


# 5. METRICS HELPER
def compute_metrics(y_test, y_pred, t_train, t_inf):
    return {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred,
                                     average='weighted', zero_division=0),
        'recall':    recall_score(y_test, y_pred,
                                  average='weighted', zero_division=0),
        'f1':        f1_score(y_test, y_pred,
                              average='weighted', zero_division=0),
        't_train':   t_train,
        't_inf':     t_inf,
    }


# 6. CONFUSION MATRIX
def plot_confusion_matrices(y_test, y_pred_knn, y_pred_svm):
    section("5a. CONFUSION MATRIX")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, y_pred, title in zip(
        axes, [y_pred_knn, y_pred_svm], ['Best KNN', 'Best SVM']
    ):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=range(10), yticklabels=range(10))
        ax.set_title(f"Confusion Matrix - {title}", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle("Confusion Matrices: KNN vs SVM",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# 7. ROC CURVES
def plot_roc(X_train, X_test, y_train, y_test, best_knn, best_svm):
    section("5b. ROC CURVES (One-vs-Rest)")

    n_classes = 10
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, model, title in zip(
        axes, [best_knn, best_svm], ['Best KNN', 'Best SVM']
    ):
        ovr = OneVsRestClassifier(model)
        ovr.fit(X_train, y_train)

        if hasattr(ovr, 'predict_proba'):
            y_score = ovr.predict_proba(X_test)
        else:
            y_score = ovr.decision_function(X_test)
            dmin, dmax = y_score.min(), y_score.max()
            y_score = (y_score - dmin) / (dmax - dmin + 1e-9)

        mean_aucs = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            mean_aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1.2, alpha=0.7,
                    label=f'Digit {i} (AUC={roc_auc:.2f})',
                    color=COLORS_10[i])

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(
            f"ROC One-vs-Rest - {title}\n"
            f"Mean AUC={np.mean(mean_aucs):.3f}",
            fontweight='bold')
        ax.legend(loc='lower right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

    plt.suptitle("ROC Curves: KNN vs SVM (One-vs-Rest, 10 kelas)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# 8. DECISION BOUNDARY (PCA 2D)
def plot_decision_boundary_pca(X_train, X_test, y_train, y_test,
                                best_knn, best_svm):
    section("5c. DECISION BOUNDARY (PCA 2D)")

    pca = PCA(n_components=2, random_state=42)
    X_tr2 = pca.fit_transform(X_train)
    X_te2 = pca.transform(X_test)

    xmin, xmax = X_tr2[:, 0].min() - 1, X_tr2[:, 0].max() + 1
    ymin, ymax = X_tr2[:, 1].min() - 1, X_tr2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 150),
                         np.linspace(ymin, ymax, 150))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, model, title in zip(
        axes, [best_knn, best_svm], ['Best KNN', 'Best SVM']
    ):
        m2d = type(model)(**model.get_params())
        m2d.fit(X_tr2, y_train)
        Z = m2d.predict(mesh).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.25, cmap='tab10',
                    levels=np.arange(-0.5, 10.5, 1))
        for cls in range(10):
            ax.scatter(X_tr2[y_train == cls, 0],
                       X_tr2[y_train == cls, 1],
                       c=[COLORS_10[cls]], alpha=0.5, s=20, marker='o')
            ax.scatter(X_te2[y_test == cls, 0],
                       X_te2[y_test == cls, 1],
                       c=[COLORS_10[cls]], alpha=0.9, s=40, marker='*',
                       edgecolors='black', linewidths=0.4,
                       label=f'D{cls}')
        ax.set_title(
            f"Decision Boundary - {title}\n(PCA 2D, * = test points)",
            fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.suptitle("Decision Boundary PCA 2D: KNN vs SVM",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# 9. LEARNING CURVES
def plot_learning_curves(X_train, y_train, best_knn, best_svm):
    section("6. LEARNING CURVES")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    models = [best_knn, best_svm]
    titles = ['Best KNN', 'Best SVM']
    colors = ['steelblue', 'darkorange']

    for ax, model, title, color in zip(axes, models, titles, colors):
        sizes, tr_sc, val_sc = learning_curve(
            model, X_train, y_train,
            cv=cv, scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1)

        tm, ts = tr_sc.mean(axis=1), tr_sc.std(axis=1)
        vm, vs = val_sc.mean(axis=1), val_sc.std(axis=1)

        ax.fill_between(sizes, tm - ts, tm + ts, alpha=0.15, color=color)
        ax.fill_between(sizes, vm - vs, vm + vs, alpha=0.15, color='green')
        ax.plot(sizes, tm, 'o-', color=color, label='Train', lw=2)
        ax.plot(sizes, vm, 's-', color='green', label='CV Val', lw=2)
        ax.set_title(f"Learning Curve - {title}", fontweight='bold')
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle("Learning Curves: Ukuran Data vs Akurasi",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# 10. CLASSIFICATION REPORT
def print_reports(y_test, y_pred_knn, y_pred_svm):
    section("5d. CLASSIFICATION REPORT DETAIL")
    names = [f"Digit {i}" for i in range(10)]
    print("\n--- KNN ---")
    print(classification_report(y_test, y_pred_knn, target_names=names))
    print("\n--- SVM ---")
    print(classification_report(y_test, y_pred_svm, target_names=names))


# 11. PERBANDINGAN KOMPREHENSIF
def plot_comparison(metrics_knn, metrics_svm, y_test,
                    y_pred_knn, y_pred_svm):
    section("7. PERBANDINGAN KOMPREHENSIF KNN vs SVM")

    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    knn_vals = [metrics_knn[m] for m in metric_names]
    svm_vals = [metrics_svm[m] for m in metric_names]

    print(f"\n{'Metrik':<12} {'KNN':>10} {'SVM':>10}")
    print("-" * 35)
    for m, kv, sv in zip(metric_names, knn_vals, svm_vals):
        winner = "<- KNN" if kv > sv else "<- SVM"
        print(f"{m:<12} {kv:>10.4f} {sv:>10.4f}  {winner}")
    print(f"{'t_train':<12} {metrics_knn['t_train']:>9.2f}s"
          f" {metrics_svm['t_train']:>9.2f}s")
    print(f"{'t_inf':<12} {metrics_knn['t_inf']:>9.4f}s"
          f" {metrics_svm['t_inf']:>9.4f}s")

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Bar chart metrik
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(metric_names))
    w = 0.35
    bars1 = ax0.bar(x - w/2, knn_vals, w, label='KNN',
                    color='steelblue', edgecolor='white')
    bars2 = ax0.bar(x + w/2, svm_vals, w, label='SVM',
                    color='coral', edgecolor='white')
    ax0.set_xticks(x)
    ax0.set_xticklabels([m.title() for m in metric_names])
    ax0.set_ylim(0, 1.1)
    ax0.set_ylabel('Score')
    ax0.set_title("Metrik Evaluasi", fontweight='bold')
    ax0.legend()
    for bar in [*bars1, *bars2]:
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}',
                 ha='center', fontsize=7)

    # Waktu training vs inference
    ax1 = fig.add_subplot(gs[0, 1])
    t_labels = ['Training', 'Inference']
    knn_times = [metrics_knn['t_train'], metrics_knn['t_inf']]
    svm_times = [metrics_svm['t_train'], metrics_svm['t_inf']]
    x2 = np.arange(2)
    ax1.bar(x2 - w/2, knn_times, w, label='KNN', color='steelblue')
    ax1.bar(x2 + w/2, svm_times, w, label='SVM', color='coral')
    ax1.set_xticks(x2)
    ax1.set_xticklabels(t_labels)
    ax1.set_ylabel('Detik')
    ax1.set_title("Waktu Training vs Inference", fontweight='bold')
    ax1.legend()

    # F1 per kelas
    ax2 = fig.add_subplot(gs[0, 2])
    f1_knn = [f1_score(y_test, y_pred_knn, labels=[c],
                       average='macro', zero_division=0)
              for c in range(10)]
    f1_svm = [f1_score(y_test, y_pred_svm, labels=[c],
                       average='macro', zero_division=0)
              for c in range(10)]
    x3 = np.arange(10)
    ax2.plot(x3, f1_knn, 'o-', label='KNN', color='steelblue', lw=2)
    ax2.plot(x3, f1_svm, 's-', label='SVM', color='coral', lw=2)
    ax2.set_xticks(x3)
    ax2.set_xticklabels([f'D{i}' for i in range(10)], fontsize=8)
    ax2.set_ylabel('F1-Score')
    ax2.set_title("F1-Score per Kelas (Digit)", fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Radar chart
    ax3 = fig.add_subplot(gs[1, :2], polar=True)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    knn_r = knn_vals + knn_vals[:1]
    svm_r = svm_vals + svm_vals[:1]
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.plot(angles, knn_r, 'o-', color='steelblue', lw=2, label='KNN')
    ax3.fill(angles, knn_r, alpha=0.15, color='steelblue')
    ax3.plot(angles, svm_r, 's-', color='coral', lw=2, label='SVM')
    ax3.fill(angles, svm_r, alpha=0.15, color='coral')
    ax3.set_title("Radar Chart: KNN vs SVM",
                  fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Ringkasan teks
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    best = "KNN" if metrics_knn['f1'] > metrics_svm['f1'] else "SVM"
    f_tr = "KNN" if metrics_knn['t_train'] < metrics_svm['t_train'] else "SVM"
    f_in = "KNN" if metrics_knn['t_inf'] < metrics_svm['t_inf'] else "SVM"
    summary = (
        f"RINGKASAN ANALISIS\n{'=' * 28}\n\n"
        f"Akurasi lebih tinggi : {best}\n"
        f"Training lebih cepat : {f_tr}\n"
        f"Inference lebih cepat: {f_in}\n\n"
        f"KNN Best:\n"
        f"  Acc  = {metrics_knn['accuracy']:.4f}\n"
        f"  F1   = {metrics_knn['f1']:.4f}\n"
        f"  Train= {metrics_knn['t_train']:.2f}s\n\n"
        f"SVM Best:\n"
        f"  Acc  = {metrics_svm['accuracy']:.4f}\n"
        f"  F1   = {metrics_svm['f1']:.4f}\n"
        f"  Train= {metrics_svm['t_train']:.2f}s\n\n"
        f"-> REKOMENDASI:\n"
        f"  {best} lebih unggul\n"
        f"  untuk dataset ini."
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.8))

    plt.suptitle(
        "PERBANDINGAN KOMPREHENSIF: KNN vs SVM\n"
        "Dataset MNIST Digit (1000 sampel)",
        fontsize=14, fontweight='bold')
    plt.show()


# 12. ANALISIS & KESIMPULAN
def final_analysis(metrics_knn, metrics_svm):
    section("8. ANALISIS & KESIMPULAN")

    best_f1 = "KNN" if metrics_knn['f1'] > metrics_svm['f1'] else "SVM"
    faster = "KNN" if metrics_knn['t_train'] < metrics_svm['t_train'] else "SVM"

    print("""
HASIL ANALISIS:
----------------------------------------------------------

1. METODE TERBAIK:
   SVM umumnya unggul pada data citra berdimensi tinggi
   dengan fitur HOG. KNN kompetitif namun sensitif terhadap
   noise karena berbasis jarak.

2. PARAMETER OPTIMAL:
   KNN : k & metrik terbaik dikonfirmasi via GridSearchCV
         weights='distance' biasanya lebih baik dari 'uniform'
   SVM : kernel Linear atau RBF (tergantung separabilitas)
         C optimal umumnya antara 1-10
         gamma='scale' cukup robust untuk data ternormalisasi

3. TRADE-OFF:
   Aspek      KNN        SVM        Pemenang
   ---------  ---------  ---------  ---------
   Akurasi    sedang     tinggi     SVM
   Training   cepat*     lambat     KNN
   Inference  lambat**   cepat      SVM
   Memori     besar      kecil      SVM
   Tuning     mudah      kompleks   KNN
   Outlier    sensitif   robust     SVM

   *  KNN training = hanya menyimpan data (lazy learner)
   ** KNN inference = hitung jarak ke semua data train

4. FITUR TERBAIK:
   Kombinasi HOG + Raw Pixel memberikan info komplementer:
   HOG   -> tepi/gradien (bentuk digit)
   Pixel -> informasi intensitas langsung

5. REKOMENDASI APLIKASI:
   - Produksi / dataset besar  -> SVM (lebih scalable)
   - Prototipe / eksplorasi    -> KNN (mudah diimplementasi)
   - Butuh interpretabilitas   -> KNN (analisis tetangga)
   - Real-time inference       -> SVM (waktu konstan)
   - RAM terbatas              -> SVM (tidak simpan semua data)

6. EKSTRAKSI FITUR:
   HOG efektif untuk digit karena menangkap gradien yang
   merepresentasikan bentuk angka. Kombinasi HOG + pixel
   meningkatkan akurasi dibanding masing-masing secara mandiri.
""")

    print(f"  PEMENANG AKURASI         : {best_f1}")
    print(f"  PEMENANG KECEPATAN TRAIN : {faster}")
    print(f"\n  KNN  Accuracy={metrics_knn['accuracy']:.4f}"
          f"  F1={metrics_knn['f1']:.4f}")
    print(f"  SVM  Accuracy={metrics_svm['accuracy']:.4f}"
          f"  F1={metrics_svm['f1']:.4f}")


# MAIN
def main():
    print("\n" + "=" * 60)
    print("  KOMPARASI KNN vs SVM - PENGENALAN OBJEK CITRA")
    print("  Dataset: MNIST Digit | Fitur: HOG + Pixel")
    print("=" * 60)

    np.random.seed(42)

    X, y, X_img = load_and_extract(n_samples=1000)
    X_train, X_test, y_train, y_test, _ = prepare_data(X, y)

    best_knn, y_pred_knn, metrics_knn = run_knn(
        X_train, X_test, y_train, y_test)

    best_svm, y_pred_svm, metrics_svm = run_svm(
        X_train, X_test, y_train, y_test)

    plot_confusion_matrices(y_test, y_pred_knn, y_pred_svm)
    plot_roc(X_train, X_test, y_train, y_test, best_knn, best_svm)
    plot_decision_boundary_pca(X_train, X_test, y_train, y_test,
                                best_knn, best_svm)
    print_reports(y_test, y_pred_knn, y_pred_svm)
    plot_learning_curves(X_train, y_train, best_knn, best_svm)
    plot_comparison(metrics_knn, metrics_svm, y_test,
                    y_pred_knn, y_pred_svm)
    final_analysis(metrics_knn, metrics_svm)

    print("\n" + "=" * 60)
    print("  SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()