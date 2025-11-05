# ============================================================
#  AER850 - Project 2  |  Version A  (Baseline CNN)
#  Steps 1–4 : Data Processing → Model Design → Training → Eval
# ============================================================

# ----- STEP 1: Data Processing -----
import os, random, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # cleaner TensorFlow logs

# ---- Define image parameters ----
Img_width, Img_height, Img_channel = 500, 500, 3
Img_shape = (Img_width, Img_height, Img_channel)
BATCH_SIZE = 32

# ---- Define directories ----
Train_direct = r"Data\train"
Validation_direct = r"Data\valid"

# ---- Data augmentation ----
Train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=10,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    fill_mode='nearest'
)
Validation_datagenerator = ImageDataGenerator(rescale=1./255)

# ---- Data generators ----
Train_generator = Train_datagenerator.flow_from_directory(
    Train_direct,
    target_size=(Img_width, Img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
Validation_generator = Validation_datagenerator.flow_from_directory(
    Validation_direct,
    target_size=(Img_width, Img_height),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Classes found:", Train_generator.class_indices)

#  STEP 2: Neural Network Architecture Design

from tensorflow.keras.layers import LeakyReLU

model = models.Sequential([
    layers.Input(shape=Img_shape),

    # Conv Block 1
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Conv Block 2
    layers.Conv2D(64, (3,3)),
    LeakyReLU(negative_slope=0.01),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Conv Block 3
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    # Conv Block 4
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.4),

    # Flatten + Dense Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    # Output
    layers.Dense(3, activation='softmax')
])

model.summary()

#  STEP 3: Model Hyperparameter Analysis
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---- Compute class weights for imbalance ----
counts = Counter(Train_generator.classes)
total = sum(counts.values())
class_weight = {k: total / (len(counts) * counts[k]) for k in counts}
print("Class weights:", class_weight)

# ---- Callbacks ----
cb = [
    ModelCheckpoint("final_model_v1.keras", monitor="val_loss", save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
]

# ---- Train the model ----
history = model.fit(
    Train_generator,
    epochs=60,
    validation_data=Validation_generator,
    callbacks=cb,
    class_weight=class_weight,
    verbose=1
)
#  STEP 4: Model Evaluation (Loss, Accuracy, Confusion Matrix)
val_loss, val_acc = model.evaluate(Validation_generator, verbose=0)
print(f"\nValidation Accuracy: {val_acc:.4f} | Validation Loss: {val_loss:.4f}")

# ---- Accuracy & Loss Curves ----
h = history.history
plt.figure(figsize=(12,4))
plt.plot(h['accuracy'], label='Training Accuracy')
plt.plot(h['val_accuracy'], label='Validation Accuracy')
plt.title('Model Training vs Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout(); plt.savefig("perf_accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(12,4))
plt.plot(h['loss'], label='Training Loss')
plt.plot(h['val_loss'], label='Validation Loss')
plt.title('Model Training vs Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout(); plt.savefig("perf_loss.png", dpi=150)
plt.close()

print("[saved] perf_accuracy.png, perf_loss.png")

# ---- Confusion Matrix & Report ----
Validation_generator.reset()
y_true = Validation_generator.classes
y_prob = model.predict(Validation_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
class_names = list(Validation_generator.class_indices.keys())

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\n=== Confusion Matrix ===\n", cm)

# Plot CM
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title("Validation Confusion Matrix")
plt.colorbar()
tick = np.arange(len(class_names))
plt.xticks(tick, class_names, rotation=45, ha="right")
plt.yticks(tick, class_names)
thresh = cm.max()/2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 color='white' if cm[i,j]>thresh else 'black')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("val_confusion_matrix.png", dpi=150)
plt.close()

print("[saved] val_confusion_matrix.png")

# ---- Save model & class indices ----
import json
model.save("final_model_v1.keras")
with open("class_indices.json", "w") as f:
    json.dump(Validation_generator.class_indices, f, indent=2)

print("[saved] final_model_v1.keras, class_indices.json")