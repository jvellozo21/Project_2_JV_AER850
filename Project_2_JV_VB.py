#  Student Number: 501106139 - Jayden Vellozo
# AER850 - Project 2 — Version B (Steps 1–4, VS Code)
#  Steps 1–4 : Data Processing → Model Design → Training → Eval

# Step 1: Data Processing
import os, random, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, LeakyReLU
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

# Reproducibility
SEED = 42
random.seed(SEED); 
np.random.seed(SEED); 
tf.random.set_seed(SEED)

# Define image parameters 
Img_width, Img_height, Img_channel = 500, 500, 3
Img_shape = (Img_width, Img_height, Img_channel)
BATCH_SIZE = 32

# Define directories
Train_direct = r"Data\train"
Validation_direct = r"Data\valid"

# Train augmentation (a bit stronger for V2) 
Train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.25,
    rotation_range=12,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.85, 1.15],
    horizontal_flip=True,
    fill_mode='nearest'
)
Validation_datagenerator = ImageDataGenerator(rescale=1./255)

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
print("Classes:", Train_generator.class_indices)

# Model Architecture (Version B) 
model = models.Sequential([
    layers.Input(shape=Img_shape),

    # Convolution Block 1
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Convolution Block 2
    layers.Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.01),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.20),

    # Convolution Block 3
    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Convolution Block 4
    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.30),

    # Convolution Block 5
    layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),

    # GAP head (smaller parameter count than Flatten)
    GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.30),
    layers.Dense(3, activation='softmax')
])
model.summary()

# Step 3: Hyperparamete Analysis & Training
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

# Class weights (cap to avoid over-penalizing any class)
counts = Counter(Train_generator.classes)
total = sum(counts.values())
class_weight = {k: total / (len(counts) * counts[k]) for k in counts}
class_weight = {k: min(v, 2.0) for k, v in class_weight.items()}
print("Class Weights:", class_weight)

# Callbacks
cb = [
    ModelCheckpoint("final_model_v2.keras", monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
]


# Train
history = model.fit(
    Train_generator,
    epochs=60,
    validation_data=Validation_generator,
    callbacks=cb,
    class_weight=class_weight,
    verbose=1
)

# Step 4 Evaluation (Accuracy/Loss + Confusion Matrix) 
val_loss, val_acc = model.evaluate(Validation_generator, verbose=0)
print(f"Validation accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")

# Accuracy plot (with axis labels)
h = history.history
plt.figure(figsize=(12,4))
plt.plot(h["accuracy"], label="Training Accuracy")
plt.plot(h["val_accuracy"], label="Validation Accuracy")
plt.title("Model Training and Validation Accuracy")
plt.xlabel("Epoch")                     # X label added
plt.ylabel("Accuracy")                  # Y label added
plt.legend()
plt.tight_layout()
plt.savefig("perf_accuracy_v2.png", dpi=150)

# Loss plot (with axis labels)
plt.figure(figsize=(12,4))
plt.plot(h["loss"], label="Training Loss")
plt.plot(h["val_loss"], label="Validation Loss")
plt.title("Model Training and Validation Loss")
plt.xlabel("Epoch")                     # X label added
plt.ylabel("Loss")                      # Y label added
plt.legend()
plt.tight_layout()
plt.savefig("perf_loss_v2.png", dpi=150)
print("[saved] perf_accuracy_v2.png, perf_loss_v2.png")

# Confusion matrix + report 
Validation_generator.reset()
y_true = Validation_generator.classes
y_prob = model.predict(Validation_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
class_names = list(Validation_generator.class_indices.keys())

print("\n=== Classification Report (V2) ===")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Validation Confusion Matrix (V2)")
plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)
th = cm.max()/2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 color="white" if cm[i, j] > th else "black")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("val_confusion_matrix_v2.png", dpi=150)
plt.close()
print("[saved] val_confusion_matrix_v2.png")

# Save model for Step 5
import json
model.save("final_model_v2.keras")
with open("class_indices.json", "w") as f:
    json.dump(Validation_generator.class_indices, f, indent=2)
print("[saved] class_indices.json")
