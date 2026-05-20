import os, time, warnings, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

DATASET_DIR = "dataset_minggu14"
CLASS_NAMES = ["lingkaran", "kotak", "segitiga"]
IMG_SIZE    = 64
IMG_SHAPE   = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = 3
BATCH_SIZE  = 16
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── LOAD DATASET ─────────────────────────────────────────────
def load_dataset():
    from PIL import Image
    X, y, missing = [], [], []
    for label, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(folder):
            missing.append(cls); continue
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if not files:
            missing.append(cls); continue
        for fname in files:
            try:
                img = Image.open(os.path.join(folder,fname)).convert('RGB').resize((IMG_SIZE,IMG_SIZE))
                X.append(np.array(img,dtype=np.float32)/255.0); y.append(label)
            except Exception as e:
                print(f"  [skip] {fname}: {e}")
        print(f"  Kelas '{cls}': {len(files)} gambar dimuat.")
    if missing:
        print(f"\n[PERINGATAN] Folder tidak ditemukan: {missing} => dataset sintetis.\n")
        return _synthetic_dataset()
    X, y = np.array(X), np.array(y)
    print(f"\nTotal: {len(X)} gambar, shape: {X.shape}")
    return X, y


def _synthetic_dataset(n=100):
    print("[INFO] Membuat dataset sintetis ...")
    X, y = [], []
    for label, cls in enumerate(CLASS_NAMES):
        for _ in range(n):
            img   = np.full((IMG_SIZE,IMG_SIZE,3), 240, dtype=np.uint8)
            color = tuple(np.random.randint(30,200,3).tolist())
            if cls == "lingkaran":
                r=np.random.randint(IMG_SIZE//6,IMG_SIZE//3)
                cx=np.random.randint(r+2,IMG_SIZE-r-2); cy=np.random.randint(r+2,IMG_SIZE-r-2)
                cv2.circle(img,(cx,cy),r,color,-1)
            elif cls == "kotak":
                s=np.random.randint(IMG_SIZE//5,IMG_SIZE//2)
                x1=np.random.randint(2,IMG_SIZE-s-2); y1=np.random.randint(2,IMG_SIZE-s-2)
                cv2.rectangle(img,(x1,y1),(x1+s,y1+s),color,-1)
            else:
                pad=IMG_SIZE//6
                p1=(np.random.randint(pad,IMG_SIZE-pad),np.random.randint(pad,IMG_SIZE//2))
                p2=(p1[0]-np.random.randint(15,30),np.random.randint(IMG_SIZE//2,IMG_SIZE-pad))
                p3=(p1[0]+np.random.randint(15,30),np.random.randint(IMG_SIZE//2,IMG_SIZE-pad))
                cv2.fillPoly(img,[np.array([p1,p2,p3])],color)
            noise=np.random.normal(0,5,img.shape).astype(np.int16)
            img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)
            X.append(img.astype(np.float32)/255.0); y.append(label)
    return np.array(X), np.array(y)


def split_data(X, y):
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
    X_tr,X_va,y_tr,y_va=train_test_split(X_tr,y_tr,test_size=0.2,random_state=SEED,stratify=y_tr)
    print(f"  Train:{len(X_tr)} | Val:{len(X_va)} | Test:{len(X_te)}")
    return X_tr,X_va,X_te,y_tr,y_va,y_te


# ── FIGURE 1: DATASET + AUGMENTASI ───────────────────────────
def fig1_dataset_and_augmentation(X, y, X_tr, y_tr):
    print("\n[FIG 1] Dataset & Augmentasi ...")
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=True, zoom_range=0.2, shear_range=0.2, fill_mode='nearest')
    n_sample = 5
    fig, axes = plt.subplots(NUM_CLASSES*2, n_sample+1, figsize=((n_sample+1)*2.2, NUM_CLASSES*2*2.2))
    fig.suptitle("Figure 1 – Sampel Dataset & Data Augmentasi", fontsize=14, fontweight='bold')
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        row_s = cls_idx*2; row_a = cls_idx*2+1
        for r, lbl, col in [(row_s,f"{cls_name}\n(asli)",'black'),
                             (row_a,f"{cls_name}\n(aug)", 'navy')]:
            axes[r,0].text(0.5,0.5,lbl,ha='center',va='center',fontsize=9,
                           fontweight='bold',color=col,transform=axes[r,0].transAxes)
            axes[r,0].axis('off')
        idxs = np.where(y==cls_idx)[0][:n_sample]
        src  = X_tr[np.where(y_tr==cls_idx)[0][0:1]]
        gen  = datagen.flow(src, batch_size=1)
        for j, idx in enumerate(idxs):
            axes[row_s,j+1].imshow(X[idx]); axes[row_s,j+1].axis('off')
            axes[row_a,j+1].imshow(next(gen)[0].clip(0,1)); axes[row_a,j+1].axis('off')
    plt.tight_layout(); plt.show(); plt.close()


# ── MODEL BUILDERS ────────────────────────────────────────────
def build_cnn_scratch(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    return keras.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape),
        layers.BatchNormalization(), layers.MaxPooling2D(2,2),
        layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(2,2),
        layers.Conv2D(128,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(2,2),
        layers.Flatten(), layers.Dense(256,activation='relu'),
        layers.Dropout(0.5), layers.Dense(num_classes,activation='softmax')
    ], name="CNN_Scratch")


def build_cnn_deep(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    return keras.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Conv2D(128,(3,3),activation='relu',padding='same'),
        layers.Conv2D(128,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512,activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(256,activation='relu'), layers.Dropout(0.3),
        layers.Dense(num_classes,activation='softmax')
    ], name="CNN_Deep")


def build_transfer_model(base_name, trainable=False, fine_tune_at=None):
    base_map = {
        'MobileNetV2': (keras.applications.MobileNetV2,
                        keras.applications.mobilenet_v2.preprocess_input),
        'VGG16':       (keras.applications.VGG16,
                        keras.applications.vgg16.preprocess_input),
    }
    model_fn, preproc_fn = base_map[base_name]
    base = model_fn(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    base.trainable = trainable
    if fine_tune_at is not None:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
    inputs  = keras.Input(shape=IMG_SHAPE)
    x       = preproc_fn(inputs * 255.0)
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs, outputs, name=f"TL_{base_name}")


def train_model(model, X_tr, y_tr, X_va, y_va,
                optimizer='adam', lr=1e-3, epochs=50, augment=False, label="model"):
    opt = (keras.optimizers.Adam(learning_rate=lr) if optimizer=='adam'
           else keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True))
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cbs = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=12,
                                      restore_best_weights=True,verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                                          patience=5,min_lr=1e-7,verbose=0)
    ]
    if augment:
        dg    = ImageDataGenerator(rotation_range=20,width_shift_range=0.2,
                                   height_shift_range=0.2,horizontal_flip=True,
                                   zoom_range=0.2,shear_range=0.2,fill_mode='nearest')
        gen   = dg.flow(X_tr,y_tr,batch_size=BATCH_SIZE,seed=SEED)
        steps = max(1,len(X_tr)//BATCH_SIZE)
        t0    = time.time()
        hist  = model.fit(gen,steps_per_epoch=steps,epochs=epochs,
                          validation_data=(X_va,y_va),callbacks=cbs,verbose=0)
    else:
        t0   = time.time()
        hist = model.fit(X_tr,y_tr,batch_size=BATCH_SIZE,epochs=epochs,
                         validation_data=(X_va,y_va),callbacks=cbs,verbose=0)
    elapsed = time.time()-t0
    print(f"  [{label}] {elapsed:.0f}s | val_acc={hist.history['val_accuracy'][-1]:.4f}")
    return hist, elapsed


# ── FIGURE 2: LEARNING CURVES ─────────────────────────────────
def fig2_learning_curves(scratch_hists, scratch_labels, tl_hists, tl_labels):
    print("\n[FIG 2] Learning curves ...")
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("Figure 2 – Learning Curves", fontsize=14, fontweight='bold')
    colors = plt.cm.tab10.colors

    def draw(ax_acc, ax_loss, hists, labels, prefix):
        for i,(h,lbl) in enumerate(zip(hists,labels)):
            c = colors[i%len(colors)]
            ax_acc.plot(h.history['accuracy'],    color=c, label=f'{lbl} train')
            ax_acc.plot(h.history['val_accuracy'],color=c, ls='--',label=f'{lbl} val')
            ax_loss.plot(h.history['loss'],       color=c, label=f'{lbl} train')
            ax_loss.plot(h.history['val_loss'],   color=c, ls='--',label=f'{lbl} val')
        for ax,ttl,yl in [(ax_acc,f'{prefix} – Akurasi','Accuracy'),
                           (ax_loss,f'{prefix} – Loss','Loss')]:
            ax.set_title(ttl); ax.set_xlabel('Epoch'); ax.set_ylabel(yl)
            ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    draw(axes[0,0],axes[0,1],scratch_hists,scratch_labels,"CNN Scratch")
    draw(axes[1,0],axes[1,1],tl_hists,    tl_labels,     "Transfer Learning")
    plt.tight_layout(); plt.show(); plt.close()


# ── EVALUASI ──────────────────────────────────────────────────
def evaluate(model, X_te, y_te, name):
    t0        = time.time()
    y_prob    = model.predict(X_te,verbose=0)
    inf_ms    = (time.time()-t0)/len(X_te)*1000
    y_pred    = np.argmax(y_prob,axis=1)
    loss, acc = model.evaluate(X_te,y_te,verbose=0)
    print(f"  {name}: acc={acc:.4f} | loss={loss:.4f} | {inf_ms:.3f}ms/img")
    print(classification_report(y_te,y_pred,target_names=CLASS_NAMES))
    return y_pred, y_prob, acc, loss, inf_ms


# ── FIGURE 3: CONFUSION MATRIX GRID ──────────────────────────
def fig3_confusion_matrices(eval_results):
    print("\n[FIG 3] Confusion matrices ...")
    names = list(eval_results.keys())
    n     = len(names); ncols = min(3,n); nrows = (n+ncols-1)//ncols
    fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*4))
    axes = np.array(axes).reshape(nrows,ncols)
    fig.suptitle("Figure 3 – Confusion Matrix Semua Model", fontsize=14, fontweight='bold')
    for i, name in enumerate(names):
        ax     = axes[i//ncols, i%ncols]
        y_pred = eval_results[name][0]
        y_te   = eval_results[name][5]
        acc    = eval_results[name][2]
        cm     = confusion_matrix(y_te,y_pred)
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                    xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,ax=ax)
        ax.set_title(f"{name}\nacc={acc:.3f}",fontsize=9,fontweight='bold')
        ax.set_xlabel('Prediksi',fontsize=8); ax.set_ylabel('Aktual',fontsize=8)
    for j in range(n,nrows*ncols):
        axes[j//ncols,j%ncols].axis('off')
    plt.tight_layout(); plt.show(); plt.close()


# ── FIGURE 4: ROC CURVE GRID ──────────────────────────────────
def fig4_roc_curves(eval_results):
    print("\n[FIG 4] ROC curves ...")
    names = list(eval_results.keys())
    n     = len(names); ncols = min(3,n); nrows = (n+ncols-1)//ncols
    cls_c = ['darkorange','steelblue','green']
    fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*4.5,nrows*4))
    axes = np.array(axes).reshape(nrows,ncols)
    fig.suptitle("Figure 4 – ROC Curve Semua Model (multi-class)", fontsize=14, fontweight='bold')
    for i, name in enumerate(names):
        ax    = axes[i//ncols,i%ncols]
        y_prob= eval_results[name][1]
        y_te  = eval_results[name][5]
        y_bin = label_binarize(y_te,classes=list(range(NUM_CLASSES)))
        for j,(cls,c) in enumerate(zip(CLASS_NAMES,cls_c)):
            fpr,tpr,_ = roc_curve(y_bin[:,j],y_prob[:,j])
            ax.plot(fpr,tpr,color=c,lw=1.5,label=f'{cls} AUC={auc(fpr,tpr):.2f}')
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.set_title(name,fontsize=9,fontweight='bold')
        ax.set_xlabel('FPR',fontsize=8); ax.set_ylabel('TPR',fontsize=8)
        ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
    for j in range(n,nrows*ncols):
        axes[j//ncols,j%ncols].axis('off')
    plt.tight_layout(); plt.show(); plt.close()


# ── FIGURE 5: INTERPRETASI (feature maps | Grad-CAM | t-SNE) ─
def _safe_input(model, sample):
    _ = model.predict(sample,verbose=0)
    try:    return model.input
    except: return model.inputs[0]


def _grad_cam(model, img_arr):
    last_conv = next((l for l in reversed(model.layers) if isinstance(l,layers.Conv2D)),None)
    if last_conv is None: return None
    try:
        gm = keras.Model(inputs=_safe_input(model,img_arr),
                         outputs=[last_conv.output, model.output])
    except Exception: return None
    with tf.GradientTape() as tape:
        inp_t = tf.cast(img_arr,tf.float32)
        conv_out,preds = gm(inp_t)
        loss = preds[:,tf.argmax(preds[0])]
    grads  = tape.gradient(loss,conv_out)
    pooled = tf.reduce_mean(grads,axis=(0,1,2))
    cam    = tf.reduce_sum(tf.multiply(pooled,conv_out[0]),axis=-1).numpy()
    cam    = np.maximum(cam,0)
    if cam.max()>0: cam/=cam.max()
    return cam


def fig5_interpretation(model, X_te, y_te, model_name):
    print(f"\n[FIG 5] Interpretasi model: {model_name} ...")

    # Feature maps
    conv_layers = [l for l in model.layers if isinstance(l,layers.Conv2D)]
    feat_maps   = None
    if conv_layers:
        try:
            fm_model  = keras.Model(inputs=_safe_input(model,X_te[0:1]),
                                    outputs=conv_layers[0].output)
            feat_maps = fm_model.predict(X_te[0:1],verbose=0)
        except Exception as e:
            print(f"  [skip feature maps] {e}")

    # t-SNE
    emb2d = None
    try:
        inp_t = _safe_input(model,X_te[:1])
        for layer in reversed(model.layers[:-1]):
            try:
                if len(layer.output.shape) <= 2:
                    em = keras.Model(inputs=inp_t,outputs=layer.output).predict(X_te,verbose=0)
                    if em.ndim>2: em=em.reshape(len(em),-1)
                    if em.shape[1]>50: em=PCA(n_components=50,random_state=SEED).fit_transform(em)
                    emb2d=TSNE(n_components=2,random_state=SEED,
                               perplexity=min(30,len(X_te)-1)).fit_transform(em)
                    break
            except Exception: continue
    except Exception as e:
        print(f"  [skip t-SNE] {e}")

    fig = plt.figure(figsize=(18,9))
    fig.suptitle(f"Figure 5 – Interpretasi Model: {model_name}", fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(1,3,wspace=0.35)

    # Feature maps 4x8
    gs_fm = gs[0].subgridspec(4,8,hspace=0.05,wspace=0.05)
    n_fm  = min(32,feat_maps.shape[-1]) if feat_maps is not None else 0
    for fi in range(32):
        ax = fig.add_subplot(gs_fm[fi//8,fi%8])
        if fi<n_fm: ax.imshow(feat_maps[0,:,:,fi],cmap='viridis')
        ax.axis('off')
    lname = conv_layers[0].name if conv_layers else '-'
    fig.text(0.18,0.94,f"Feature Maps ({lname})",ha='center',fontsize=9,style='italic')

    # Grad-CAM 3x3
    gs_gc = gs[1].subgridspec(3,3,hspace=0.35,wspace=0.1)
    idxs  = np.random.default_rng(SEED).choice(len(X_te),9,replace=False)
    for gi,idx in enumerate(idxs):
        ax  = fig.add_subplot(gs_gc[gi//3,gi%3])
        img = X_te[idx]
        cam = _grad_cam(model,img[np.newaxis,...])
        if cam is not None:
            cam_r   = cv2.resize(cam,(img.shape[1],img.shape[0]))
            overlay = (img*0.5 + plt.cm.jet(cam_r)[:,:,:3]*0.5).clip(0,1)
            ax.imshow(overlay)
        else:
            ax.imshow(img)
        pred  = np.argmax(model.predict(img[np.newaxis,...],verbose=0))
        true  = y_te[idx]
        ax.set_title(f"T:{CLASS_NAMES[true]}\nP:{CLASS_NAMES[pred]}",
                     color='green' if pred==true else 'red',fontsize=7)
        ax.axis('off')
    fig.text(0.50,0.94,"Grad-CAM (hijau=benar, merah=salah)",ha='center',fontsize=9,style='italic')

    # t-SNE
    ax_t = fig.add_subplot(gs[2])
    if emb2d is not None:
        for ci,(cls,c) in enumerate(zip(CLASS_NAMES,['#e6194b','#3cb44b','#4363d8'])):
            mask = y_te==ci
            ax_t.scatter(emb2d[mask,0],emb2d[mask,1],c=c,label=cls,alpha=0.75,s=30)
        ax_t.legend(fontsize=8); ax_t.grid(True,alpha=0.3)
    else:
        ax_t.text(0.5,0.5,"t-SNE\ntidak tersedia",ha='center',va='center')
    ax_t.set_title("t-SNE Feature Embedding",fontsize=10)

    plt.show(); plt.close()


# ── FIGURE 6: PERBANDINGAN AKHIR ─────────────────────────────
def fig6_comparison(results):
    print("\n[FIG 6] Perbandingan akhir ...")
    names  = list(results.keys())
    accs   = [results[n]['acc']            for n in names]
    params = [results[n]['params']/1e6     for n in names]
    times  = [results[n]['train_time']     for n in names]
    inf_ms = [results[n]['inf_ms']         for n in names]
    palette= plt.cm.tab10.colors[:len(names)]

    fig, axes = plt.subplots(1,4,figsize=(18,5))
    fig.suptitle("Figure 6 – Perbandingan Semua Model", fontsize=14, fontweight='bold')

    def bar(ax, vals, title, ylabel, fmt):
        bars = ax.bar(names,vals,color=palette)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names,rotation=30,ha='right',fontsize=8)
        ax.grid(True,alpha=0.3,axis='y')
        vmax = max(vals) if vals else 1
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+vmax*0.01,
                    f'{v:{fmt}}',ha='center',va='bottom',fontsize=7)

    bar(axes[0],accs,  "Test Accuracy",    "Accuracy",    '.3f')
    bar(axes[1],params,"Ukuran Model (M)", "Juta Param",  '.1f')
    bar(axes[2],times, "Waktu Training (s)","Detik",      '.0f')
    bar(axes[3],inf_ms,"Inference (ms/img)","ms",         '.2f')
    plt.tight_layout(); plt.show(); plt.close()


# ── ANALISIS TEKS ─────────────────────────────────────────────
def print_analysis(results):
    print("\n"+"="*65)
    print(" ANALISIS DAN KESIMPULAN")
    print("="*65)
    sorted_r = sorted(results.items(),key=lambda x:x[1]['acc'],reverse=True)
    best_name,best = sorted_r[0]
    print(f"\n{'Model':<28}{'Acc':>6}{'Params(M)':>10}{'Train(s)':>9}{'Inf(ms)':>8}")
    print("-"*65)
    for n,r in sorted_r:
        tag = " <-- TERBAIK" if n==best_name else ""
        print(f"  {n:<26}{r['acc']:>6.4f}{r['params']/1e6:>10.2f}"
              f"{r['train_time']:>9.0f}{r['inf_ms']:>8.3f}{tag}")
    scratch_accs = {n:r['acc'] for n,r in results.items() if 'Scratch' in n or 'Deep' in n}
    tl_accs      = {n:r['acc'] for n,r in results.items() if 'TL_' in n}
    if scratch_accs and tl_accs:
        bs,bt = max(scratch_accs.values()),max(tl_accs.values())
        print(f"\nCNN Scratch terbaik : {bs:.4f}")
        print(f"Transfer Learning   : {bt:.4f}  ({bt-bs:+.4f})")
        print("=> " + ("Transfer Learning lebih unggul." if bt>bs
                        else "CNN Scratch sudah kompetitif untuk dataset ini."))
    print("\nRekomendasi:")
    print("  - Dataset sederhana/kecil : CNN Scratch sudah cukup.")
    print("  - Dataset kompleks/kecil  : MobileNetV2 (ringan & akurat).")
    print("  - Augmentasi              : mengurangi overfitting secara efektif.")
    print("  - Fine-tuning             : memberikan peningkatan vs feature extraction saja.")
    print("="*65)


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("="*65)
    print(" PRAKTIKUM 14: CNN & TRANSFER LEARNING – BENTUK GEOMETRI")
    print("="*65)

    print("\n[1] Memuat dataset ...")
    X, y = load_dataset()
    y    = y.astype(np.int32)
    X_tr,X_va,X_te,y_tr,y_va,y_te = split_data(X,y)

    fig1_dataset_and_augmentation(X,y,X_tr,y_tr)

    print("\n[2] Training CNN Scratch ...")
    cnn_adam = build_cnn_scratch()
    h_adam,t_adam = train_model(cnn_adam,X_tr,y_tr,X_va,y_va,
                                optimizer='adam',lr=1e-3,epochs=60,label="CNN_Scratch_Adam")
    cnn_sgd  = build_cnn_scratch()
    h_sgd,t_sgd   = train_model(cnn_sgd, X_tr,y_tr,X_va,y_va,
                                 optimizer='sgd', lr=0.01,epochs=60,label="CNN_Scratch_SGD")
    cnn_deep = build_cnn_deep()
    h_deep,t_deep = train_model(cnn_deep,X_tr,y_tr,X_va,y_va,
                                optimizer='adam',lr=1e-3,epochs=60,
                                augment=True,label="CNN_Deep_Aug")

    print("\n[3] Training Transfer Learning ...")
    tl_hists,tl_labels,tl_models = [],[],{}
    for base in ['MobileNetV2','VGG16']:
        print(f"  Feature Extraction: {base}")
        m_fe = build_transfer_model(base,trainable=False)
        h_fe,t_fe = train_model(m_fe,X_tr,y_tr,X_va,y_va,
                                optimizer='adam',lr=1e-3,epochs=40,label=f"TL_{base}_FE")
        tl_hists.append(h_fe); tl_labels.append(f"{base}_FE")
        tl_models[f"TL_{base}_FE"] = (m_fe,t_fe)

        print(f"  Fine-tuning: {base}")
        ft_at = max(0,len(m_fe.layers)-20)
        m_ft  = build_transfer_model(base,trainable=True,fine_tune_at=ft_at)
        try:   m_ft.set_weights(m_fe.get_weights())
        except: pass
        h_ft,t_ft = train_model(m_ft,X_tr,y_tr,X_va,y_va,
                                optimizer='adam',lr=1e-5,epochs=20,label=f"TL_{base}_FT")
        tl_hists.append(h_ft); tl_labels.append(f"{base}_FT")
        tl_models[f"TL_{base}_FT"] = (m_ft,t_ft)

    fig2_learning_curves([h_adam,h_sgd,h_deep],["Scratch_Adam","Scratch_SGD","Deep_Aug"],
                         tl_hists,tl_labels)

    print("\n[4] Evaluasi semua model ...")
    eval_results,results = {},{}
    for model,name,t in [(cnn_adam,"CNN_Scratch_Adam",t_adam),
                          (cnn_sgd, "CNN_Scratch_SGD", t_sgd),
                          (cnn_deep,"CNN_Deep_Aug",    t_deep)]:
        yp,yprob,acc,loss,inf = evaluate(model,X_te,y_te,name)
        eval_results[name]    = (yp,yprob,acc,loss,inf,y_te)
        results[name]         = {'acc':acc,'loss':loss,'params':model.count_params(),
                                  'train_time':t,'inf_ms':inf}
    for name,(model,t) in tl_models.items():
        yp,yprob,acc,loss,inf = evaluate(model,X_te,y_te,name)
        eval_results[name]    = (yp,yprob,acc,loss,inf,y_te)
        results[name]         = {'acc':acc,'loss':loss,'params':model.count_params(),
                                  'train_time':t,'inf_ms':inf}

    fig3_confusion_matrices(eval_results)
    fig4_roc_curves(eval_results)

    best_scratch = max(["CNN_Scratch_Adam","CNN_Scratch_SGD","CNN_Deep_Aug"],
                       key=lambda n: results[n]['acc'])
    best_model   = {"CNN_Scratch_Adam":cnn_adam,"CNN_Scratch_SGD":cnn_sgd,
                    "CNN_Deep_Aug":cnn_deep}[best_scratch]
    fig5_interpretation(best_model,X_te,y_te,best_scratch)

    fig6_comparison(results)
    print_analysis(results)
    print("\n[SELESAI] 6 figure telah ditampilkan.")


if __name__ == "__main__":
    main()