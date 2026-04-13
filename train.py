import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pickle
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── DirectML GPU Setup ────────────────────────────────────────────────────────
try:
    import tensorflow.experimental.numpy_experimental  # noqa
    import tensorflow_directml as tf
    print("✅ Using tensorflow-directml (GPU via DirectML)")
except ImportError:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Standard TF GPU detected: {[g.name for g in gpus]}")
        except RuntimeError as e:
            print(f"⚠️  GPU config error: {e}")
    else:
        print("⚠️  No GPU found — training on CPU")

from tensorflow import keras
from tensorflow.keras import layers

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data/train"
VAL_DIR       = "data/validation"
TEST_DIR      = "data/test"
MODEL_DIR     = "models"
MODEL_PATH    = f"{MODEL_DIR}/face_emotion_cnn.h5"

# MobileNetV2 needs RGB input — we upscale 48-px grayscale to 96×96
IMG_SIZE      = 96
BATCH_SIZE    = 32        # smaller batch → better generalisation with pretrained backbone
MAX_PER_CLASS = 3000

# Two-phase training
PHASE1_EPOCHS = 30        # backbone frozen  — train head only
PHASE2_EPOCHS = 40        # backbone unfrozen — fine-tune top layers
UNFREEZE_FROM = 100       # unfreeze MobileNetV2 layers from this index onward

EMOTION_MAP = {
    "angry":    "Angry",
    "disgust":  "Disgust",
    "fear":     "Fear",
    "happy":    "Happy",
    "neutral":  "Neutral",
    "sad":      "Sad",
    "surprise": "Surprised",
}
EMOTIONS = sorted(set(EMOTION_MAP.values()))

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_images(root_dir, max_per_class=None):
    X, y = [], []
    rng  = np.random.default_rng(42)
    found = [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d)) and d.lower() in EMOTION_MAP]
    if not found:
        print(f"  No emotion folders found in '{root_dir}'")
        return np.array([]), np.array([])

    for folder in sorted(found):
        label = EMOTION_MAP[folder.lower()]
        fpath = os.path.join(root_dir, folder)
        files = [f for f in os.listdir(fpath) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if max_per_class and len(files) > max_per_class:
            files = list(rng.choice(files, max_per_class, replace=False))
        loaded = 0
        for fname in files:
            img = cv2.imread(os.path.join(fpath, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
            loaded += 1
        print(f"    {folder:>12} -> {label:<12}  loaded: {loaded}")
    return np.array(X, dtype=np.float32), np.array(y)


def encode_and_normalize(X, y, le):
    """
    Grayscale → 3-channel RGB (stack the same channel 3×),
    then apply MobileNetV2 preprocessing (scales pixels to [-1, 1]).
    """
    y_enc  = np.array([le[lbl] for lbl in y])
    X_rgb  = np.stack([X, X, X], axis=-1)           # (N, H, W, 3)
    X_prep = keras.applications.mobilenet_v2.preprocess_input(X_rgb)
    return X_prep.astype(np.float32), y_enc


# ── Model ─────────────────────────────────────────────────────────────────────
def build_transfer_model(num_classes):
    """
    MobileNetV2 pretrained on ImageNet as backbone + custom classification head.
    Phase 1 : backbone fully frozen → only head trains.
    Phase 2 : top layers unfrozen  → fine-tune with low LR.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # frozen for Phase 1

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


# ── Augmentation (stronger than before) ──────────────────────────────────────
def make_datagen():
    return keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],   # handles real-world lighting variation
        fill_mode="nearest",
    )


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("=" * 65)
    print("  Facial Emotion CNN — Improved Pipeline (MobileNetV2)")
    print("=" * 65)

    physical = tf.config.list_physical_devices()
    print("\n  Available devices:")
    for d in physical:
        print(f"    {d.device_type} — {d.name}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  → Training on: {'GPU' if gpus else 'CPU'}\n")

    # ── 1. Load data ──────────────────────────────────────────────────
    print(f"[1/5] Loading training images from '{DATA_DIR}' ...")
    X_train, y_train = load_images(DATA_DIR, max_per_class=MAX_PER_CLASS)
    print(f"      Total loaded : {len(X_train)}")

    print(f"\n[2/5] Loading validation images from '{VAL_DIR}' ...")
    if os.path.exists(VAL_DIR):
        X_val, y_val = load_images(VAL_DIR)
        print(f"      Total loaded : {len(X_val)}")
    else:
        raise FileNotFoundError(f"Validation directory '{VAL_DIR}' not found.")

    print(f"\n      Loading test images from '{TEST_DIR}' ...")
    if os.path.exists(TEST_DIR):
        X_test, y_test = load_images(TEST_DIR)
        print(f"      Total loaded : {len(X_test)}")
    else:
        print("      No test/ folder found — skipping final evaluation.")
        X_test, y_test = None, None

    # ── 2. Preprocess ─────────────────────────────────────────────────
    print("\n[3/5] Preprocessing ...")
    le         = {e: i for i, e in enumerate(EMOTIONS)}
    idx_to_lbl = {i: e for e, i in le.items()}

    X_train_n, y_train_enc = encode_and_normalize(X_train, y_train, le)
    X_val_n,   y_val_enc   = encode_and_normalize(X_val,   y_val,   le)

    print(f"      Train : {len(X_train_n)}  |  Val : {len(X_val_n)}"
          + (f"  |  Test : {len(X_test)}" if X_test is not None else ""))
    print(f"      Classes  : {EMOTIONS}")
    print(f"      Img size : {IMG_SIZE}×{IMG_SIZE} RGB (grayscale → 3-channel stack)")

    cw_arr = compute_class_weight("balanced", classes=np.arange(len(EMOTIONS)), y=y_train_enc)
    class_weights = dict(enumerate(cw_arr))
    print("      Class weights:")
    for i, e in enumerate(EMOTIONS):
        print(f"        {e:>12} : {class_weights[i]:.3f}")

    datagen = make_datagen()
    datagen.fit(X_train_n)

    # ── 3. Build model ────────────────────────────────────────────────
    print("\n[4/5] Building MobileNetV2 transfer model ...")
    model, base_model = build_transfer_model(num_classes=len(EMOTIONS))
    model.summary()

    # ── Phase 1: frozen backbone — train head only ────────────────────
    print(f"\n{'─'*65}")
    print(f"  PHASE 1 — Head training ({PHASE1_EPOCHS} epochs, backbone frozen)")
    print(f"{'─'*65}")

    compile_model(model, lr=1e-3)

    p1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    model.fit(
        datagen.flow(X_train_n, y_train_enc, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train_n) // BATCH_SIZE,
        epochs=PHASE1_EPOCHS,
        validation_data=(X_val_n, y_val_enc),
        class_weight=class_weights,
        callbacks=p1_callbacks,
        verbose=1,
    )

    p1_loss, p1_acc = model.evaluate(X_val_n, y_val_enc, verbose=0)
    print(f"\n  ✅ Phase 1 Val Accuracy : {p1_acc*100:.1f}%")

    # ── Phase 2: unfreeze top layers, fine-tune with cosine LR ───────
    print(f"\n{'─'*65}")
    print(f"  PHASE 2 — Fine-tuning (layers {UNFREEZE_FROM}+, {PHASE2_EPOCHS} epochs)")
    print(f"{'─'*65}")

    base_model.trainable = True
    for layer in base_model.layers[:UNFREEZE_FROM]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Trainable backbone layers : {trainable_count} / {len(base_model.layers)}")

    # Cosine decay — smooth LR schedule prevents destroying pretrained weights
    total_steps  = (len(X_train_n) // BATCH_SIZE) * PHASE2_EPOCHS
    cosine_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(cosine_decay),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    p2_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    model.fit(
        datagen.flow(X_train_n, y_train_enc, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train_n) // BATCH_SIZE,
        epochs=PHASE2_EPOCHS,
        validation_data=(X_val_n, y_val_enc),
        class_weight=class_weights,
        callbacks=p2_callbacks,
        verbose=1,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────
    print("\n[5/5] Evaluation ...")
    val_loss, val_acc = model.evaluate(X_val_n, y_val_enc, verbose=0)
    print(f"\n  Phase 1 Val Accuracy : {p1_acc*100:.1f}%")
    print(f"  Final  Val Accuracy  : {val_acc*100:.1f}%  (Δ {(val_acc - p1_acc)*100:+.1f}%)")

    if X_test is not None:
        X_test_n, y_test_enc = encode_and_normalize(X_test, y_test, le)
        y_pred    = np.argmax(model.predict(X_test_n, verbose=0), axis=1)
        test_acc  = (y_pred == y_test_enc).mean() * 100
        print(f"  Test Accuracy        : {test_acc:.1f}%\n")
        print(classification_report(
            y_test_enc, y_pred,
            target_names=[idx_to_lbl[i] for i in range(len(EMOTIONS))]
        ))
    else:
        y_pred = np.argmax(model.predict(X_val_n, verbose=0), axis=1)
        print("\n  Classification report on validation set:\n")
        print(classification_report(
            y_val_enc, y_pred,
            target_names=[idx_to_lbl[i] for i in range(len(EMOTIONS))]
        ))

    # ── 5. Save ───────────────────────────────────────────────────────
    with open(f"{MODEL_DIR}/cnn_label_encoder.pkl", "wb") as f:
        pickle.dump({
            "label_to_idx": le,
            "idx_to_label": idx_to_lbl,
            "emotions":     EMOTIONS,
            "img_size":     IMG_SIZE,   # main.py reads this to resize correctly
        }, f)

    print(f"\n  Model saved  -> {MODEL_PATH}")
    print(f"  Labels saved -> {MODEL_DIR}/cnn_label_encoder.pkl\n")


if __name__ == "__main__":
    train()