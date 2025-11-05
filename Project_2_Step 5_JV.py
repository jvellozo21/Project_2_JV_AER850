"STEP 5: MODEL Testing"

"Import Librarires"

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

"Load the Model"
Model = load_model("final_model_v1.keras")

"Define Input Image Shape"
Img_width = 500
Img_height = 500

"Define Class Labels"
class_labels = ['crack', 'missing-head', 'paint-off']

"Data Preprocessing"
def preprocess (image_path, target_size = (Img_width, Img_height)):
    image = load_img (image_path, target_size = target_size)
    image_array = img_to_array (image)
    image_array = image_array/255
    image_array = np.expand_dims(image_array, 0)
    return image_array

"Data Prediction"
def predict (image_array, model):
    predictions = model.predict (image_array, verbose = 0)
    return predictions

"Data Display"
def display (image_path, predictions, true_label, class_labels):
    predicted_label = class_labels [np.argmax(predictions)]
    fig, ax = plt.subplots(figsize = (6,6))
    img = plt.imread(image_path)
    plt.imshow(img)
    ax.axis('off')
    plt.title (f"True Crack Classification Label: {true_label}\n"
               f"Predicted Crack Classification Label: {predicted_label}\n")
    
    # Sort by probability (descending)
    pairs = list(zip(class_labels, predictions[0]))
    pairs.sort(key=lambda x: x[1], reverse=True)

    for index, (label, p) in enumerate(pairs):
        ax.text(
            10, 25 + index * 30,
            f"{label}: {p * 100:.2f}%",
            bbox=dict(facecolor="blue"), fontsize=10, color='white'
        )
    plt.tight_layout()
    plt.show()

"Test Specfic Images"
test_images = [
    (r"Data\test\crack\test_crack.jpg", "crack"),
    (r"Data\test\missing-head\test_missinghead.jpg", "missing Head"),
    (r"Data\test\paint-off\test_paintoff.jpg", "paint-Off")
    ]

for image_path, true_label in test_images:
    img_preprocess = preprocess(image_path)
    predictions = predict(img_preprocess, Model)
    display(image_path, predictions, true_label, class_labels)