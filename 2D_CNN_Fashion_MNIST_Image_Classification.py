import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalMaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. REPRODUCIBILITY CONFIGURATION
# ============================================================

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

sns.set_theme(style="whitegrid", context="notebook")

print("=" * 70)
print("ADVANCED 2D CNN: IMAGE CLASSIFICATION (FASHION-MNIST)")
print("=" * 70)

# ============================================================
# 2. LOAD AND INSPECT DATASET
# ============================================================

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = (
    fashion_mnist.load_data()
)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

num_classes = len(class_names)

print(f"\nRaw Training Images: {X_train_raw.shape}")
print(f"Raw Testing Images: {X_test_raw.shape}")
print(f"Image Resolution: {X_train_raw.shape[1]} x {X_train_raw.shape[2]}")
print(f"Number of Classes: {num_classes}")

# Display sample images
plt.figure(figsize=(12, 6))

for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_train_raw[i], cmap="gray")
    plt.title(class_names[y_train_raw[i]])
    plt.axis("off")

plt.suptitle("Fashion-MNIST Sample Images")
plt.tight_layout()
plt.show()

# Class distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train_raw)
plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
plt.title("Fashion-MNIST Class Distribution")
plt.xlabel("Clothing Category")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# ============================================================
# 3. PREPROCESSING
# ============================================================

# Normalize pixel values to the range [0, 1]
X_train_norm = X_train_raw.astype("float32") / 255.0
X_test_norm = X_test_raw.astype("float32") / 255.0

# Reshape images into 4D tensors:
# (samples, height, width, channels)
X_train_2d = np.expand_dims(X_train_norm, axis=-1)
X_test_2d = np.expand_dims(X_test_norm, axis=-1)

# One-hot encode target labels
y_train_encoded = to_categorical(y_train_raw, num_classes)
y_test_encoded = to_categorical(y_test_raw, num_classes)

# Create training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = (
    train_test_split(
        X_train_2d,
        y_train_encoded,
        test_size=0.15,
        random_state=SEED,
        stratify=y_train_raw
    )
)

print("\nPREPROCESSED TENSOR SHAPES")
print(f"Training Input:   {X_train_split.shape}")
print(f"Validation Input: {X_val_split.shape}")
print(f"Test Input:       {X_test_2d.shape}")

# ============================================================
# 4. BUILD THE ADVANCED 2D CNN
# ============================================================

inputs = Input(shape=(28, 28, 1), name="image_input")

# Convolutional Block 1
x = Conv2D(
    32, (3, 3), padding="same",
    use_bias=False, name="conv_block_1"
)(inputs)
x = BatchNormalization(name="bn_1")(x)
x = Activation("relu", name="relu_1")(x)
x = MaxPooling2D((2, 2), name="max_pool_1")(x)

# Convolutional Block 2
x = Conv2D(
    64, (3, 3), padding="same",
    use_bias=False, name="conv_block_2"
)(x)
x = BatchNormalization(name="bn_2")(x)
x = Activation("relu", name="relu_2")(x)
x = MaxPooling2D((2, 2), name="max_pool_2")(x)

# Convolutional Block 3
x = Conv2D(
    128, (3, 3), padding="same",
    use_bias=False, name="conv_block_3"
)(x)
x = BatchNormalization(name="bn_3")(x)
x = Activation("relu", name="relu_3")(x)

# Parallel global feature fusion
max_pool = GlobalMaxPooling2D(name="global_max_pool")(x)
avg_pool = GlobalAveragePooling2D(name="global_avg_pool")(x)

x = Concatenate(name="feature_fusion")([max_pool, avg_pool])

# Dense classification head
x = Dense(128, activation="relu", name="dense_1")(x)
x = Dropout(0.30, name="dropout_1")(x)

x = Dense(64, activation="relu", name="dense_2")(x)
x = Dropout(0.20, name="dropout_2")(x)

outputs = Dense(
    num_classes,
    activation="softmax",
    name="softmax_output"
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs,
    name="Advanced_2D_Image_CNN"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n2D CNN MODEL SUMMARY")
model.summary()

# ============================================================
# 5. TRAINING CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_2d_cnn.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# 6. TRAIN THE MODEL
# ============================================================

print("\nSTARTING 2D CNN TRAINING")

history = model.fit(
    X_train_split,
    y_train_split,
    epochs=50,
    batch_size=64,
    validation_data=(X_val_split, y_val_split),
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

# ============================================================
# 7. EVALUATE MODEL PERFORMANCE
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test_2d, y_test_encoded, verbose=0
)

y_prob = model.predict(X_test_2d, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_test_raw, y_pred)
precision = precision_score(
    y_test_raw, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_test_raw, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_test_raw, y_pred, average="macro", zero_division=0
)

print("\nFINAL TEST RESULTS")
print("=" * 60)
print(f"Test Loss:        {test_loss:.4f}")
print(f"Test Accuracy:    {test_accuracy:.4f}")
print(f"Macro Precision:  {precision:.4f}")
print(f"Macro Recall:     {recall:.4f}")
print(f"Macro F1-score:   {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test_raw,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# 8. TRAINING ACCURACY AND LOSS GRAPHS
# ============================================================

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test_raw, y_pred)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5
)

plt.title("2D CNN Confusion Matrix (Fashion-MNIST)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================================
# 10. CLASS-WISE CLASSIFICATION METRICS
# ============================================================

report = classification_report(
    y_test_raw,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

class_metrics = pd.DataFrame(report).T
class_metrics = class_metrics.loc[class_names, ["precision", "recall", "f1-score"]]

print("\nCLASS-WISE METRICS")
display(class_metrics.round(4))

class_metrics.plot(kind="bar", figsize=(13, 6))
plt.title("Class-wise Precision, Recall and F1-score")
plt.xlabel("Clothing Category")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ============================================================
# 11. VISUALIZE INTERMEDIATE FEATURE MAPS
# ============================================================

feature_layers = ["relu_1", "relu_2", "relu_3"]

feature_model = Model(
    inputs=model.input,
    outputs=[model.get_layer(name).output for name in feature_layers]
)

sample_idx = 0
sample_img = X_test_2d[sample_idx:sample_idx + 1]

activations = feature_model.predict(sample_img, verbose=0)

plt.figure(figsize=(14, 5))

plt.subplot(1, 4, 1)
plt.imshow(sample_img[0, :, :, 0], cmap="gray")
plt.title(f"Input: {class_names[y_test_raw[sample_idx]]}")
plt.axis("off")

for i, activation in enumerate(activations):
    plt.subplot(1, 4, i + 2)
    plt.imshow(activation[0, :, :, 0], cmap="viridis")
    plt.title(f"Feature Map: {feature_layers[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 12. DISPLAY MISCLASSIFIED IMAGES
# ============================================================

misclassified_indices = np.where(y_pred != y_test_raw)[0]

print(f"\nMisclassified images: {len(misclassified_indices)}")

if len(misclassified_indices) > 0:
    number_to_show = min(12, len(misclassified_indices))

    plt.figure(figsize=(12, 8))

    for i, index in enumerate(misclassified_indices[:number_to_show]):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X_test_raw[index], cmap="gray")
        plt.title(
            f"True: {class_names[y_test_raw[index]]}\n"
            f"Pred: {class_names[y_pred[index]]}"
        )
        plt.axis("off")

    plt.suptitle("Misclassified Fashion-MNIST Images")
    plt.tight_layout()
    plt.show()

# ============================================================
# 13. SAVE THE TRAINED MODEL
# ============================================================

model.save("best_2d_cnn.keras")
print("\nModel saved as best_2d_cnn.keras")

# ============================================================
# 14. INTERACTIVE GRADIO FRONTEND
# ============================================================

# Install Gradio in Google Colab if required:
# %pip -q install gradio

import gradio as gr

def classify_image(image):
    if image is None:
        return "Please upload an image.", {}

    # Convert image to grayscale and resize to 28x28
    image = tf.convert_to_tensor(image)

    if image.shape.rank == 3:
        if image.shape[-1] == 4:
            image = tf.image.rgb_to_grayscale(
                tf.image.resize(image[..., :3], (28, 28))
            )
        elif image.shape[-1] == 3:
            image = tf.image.rgb_to_grayscale(
                tf.image.resize(image, (28, 28))
            )
        elif image.shape[-1] == 1:
            image = tf.image.resize(image, (28, 28))
        else:
            return "Unsupported image format.", {}

    image = tf.image.resize(image, (28, 28))
    image = tf.cast(image, tf.float32)

    # Normalize pixel values
    image = image / 255.0

    # Add batch and channel dimensions when needed
    if image.shape.rank == 2:
        image = image[..., tf.newaxis]

    image = image[tf.newaxis, ...]

    probabilities = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    prediction_text = (
        f"Predicted Category: {class_names[predicted_index]}\n"
        f"Confidence: {probabilities[predicted_index] * 100:.2f}%"
    )

    scores = {
        class_names[i]: float(probabilities[i])
        for i in range(num_classes)
    }

    return prediction_text, scores


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="FashionVision | 2D CNN Classifier"
) as demo:

    gr.Markdown(
        """
        # FashionVision
        ### AI-powered clothing image classification

        Upload a clothing image to classify it into one of ten
        Fashion-MNIST categories using a trained 2D CNN model.

        **Supported categories:** T-shirt/top, Trouser, Pullover,
        Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Clothing Image",
                type="numpy"
            )
            predict_button = gr.Button(
                "Classify Image",
                variant="primary"
            )

        with gr.Column():
            prediction_output = gr.Textbox(
                label="Prediction",
                lines=3
            )
            probability_output = gr.Label(
                label="Class Probabilities",
                num_top_classes=10
            )

    predict_button.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[prediction_output, probability_output]
    )

    gr.Examples(
        examples=[
            [X_test_raw[i]]
            for i in [0, 1, 2, 3, 4]
        ],
        inputs=image_input,
        label="Try example Fashion-MNIST images"
    )

    gr.Markdown(
        """
        ---
        **Model:** 2D Convolutional Neural Network
        **Dataset:** Fashion-MNIST
        **Input size:** 28 × 28 grayscale image
        **Output:** Predicted category and class probabilities
        """
    )

print("\nLaunching FashionVision frontend...")
demo.launch(share=True, debug=True)
