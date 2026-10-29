# ============================================================
# TOPIC: 1D CNN FOR TABULAR DATA - IRIS FLOWER CLASSIFICATION
# Google Colab Ready Implementation
# ============================================================

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    GlobalMaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

sns.set_theme(style="whitegrid", context="notebook")

print("=" * 70)
print("ADVANCED 1D CNN: IRIS FLOWER CLASSIFICATION")
print("=" * 70)

# ============================================================
# 2. LOAD AND INSPECT DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names
num_classes = len(class_names)

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [class_names[i] for i in y]

print("\nDataset shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", feature_names)
print("Class names:", class_names)

print("\nClass distribution:")
print(df["species"].value_counts())

print("\nStatistical summary:")
print(df.describe().round(3))

# ============================================================
# 3. DATASET VISUALIZATION
# ============================================================

# 3.1 Class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="species", order=class_names)
plt.title("Distribution of Iris Flower Classes")
plt.xlabel("Flower Species")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# 3.2 Feature distributions
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for i, ax in enumerate(axes.flat):
    sns.histplot(
        data=df,
        x=feature_names[i],
        hue="species",
        kde=True,
        element="step",
        ax=ax
    )
    ax.set_title(f"Distribution of {feature_names[i]}")

plt.suptitle("Feature Distribution by Iris Species", fontsize=15)
plt.tight_layout()
plt.show()

# 3.3 Correlation matrix
plt.figure(figsize=(9, 7))
correlation = df[feature_names].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5
)

plt.title("Correlation Matrix of Iris Features")
plt.tight_layout()
plt.show()

# 3.4 Pairwise feature relationships
sns.pairplot(
    df,
    hue="species",
    vars=feature_names,
    diag_kind="hist"
)

plt.suptitle("Pairwise Relationships Among Iris Species", y=1.02)
plt.show()

# ============================================================
# 4. TRAIN-TEST SPLIT AND PREPROCESSING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# Fit scaler only on training data to prevent data leakage
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for Conv1D: (samples, sequence_length, channels)
X_train_cnn = X_train_scaled.reshape(
    X_train_scaled.shape[0],
    X_train_scaled.shape[1],
    1
)

X_test_cnn = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    1
)

y_train_encoded = to_categorical(y_train, num_classes)
y_test_encoded = to_categorical(y_test, num_classes)

print("\nTraining samples:", X_train_cnn.shape[0])
print("Testing samples:", X_test_cnn.shape[0])
print("CNN input shape:", X_train_cnn.shape)
print("CNN test input shape:", X_test_cnn.shape)

# ============================================================
# 5. BUILD THE 1D CNN MODEL
# ============================================================

inputs = Input(
    shape=(X_train_cnn.shape[1], 1),
    name="iris_input"
)

# Convolutional block 1
x = Conv1D(
    filters=32,
    kernel_size=2,
    padding="same",
    use_bias=False,
    name="conv_block_1"
)(inputs)

x = BatchNormalization(name="batch_norm_1")(x)
x = Activation("relu", name="relu_1")(x)

# Convolutional block 2
x = Conv1D(
    filters=64,
    kernel_size=2,
    padding="same",
    use_bias=False,
    name="conv_block_2"
)(x)

x = BatchNormalization(name="batch_norm_2")(x)
x = Activation("relu", name="relu_2")(x)

# Convolutional block 3
x = Conv1D(
    filters=64,
    kernel_size=2,
    padding="same",
    use_bias=False,
    name="conv_block_3"
)(x)

x = BatchNormalization(name="batch_norm_3")(x)
x = Activation("relu", name="relu_3")(x)

# Parallel feature aggregation
max_features = GlobalMaxPooling1D(
    name="global_max_pooling"
)(x)

avg_features = GlobalAveragePooling1D(
    name="global_average_pooling"
)(x)

x = Concatenate(name="feature_fusion")(
    [max_features, avg_features]
)

# Dense classification head
x = Dense(64, activation="relu", name="dense_1")(x)
x = Dropout(0.30, name="dropout_1")(x)

x = Dense(32, activation="relu", name="dense_2")(x)
x = Dropout(0.20, name="dropout_2")(x)

outputs = Dense(
    num_classes,
    activation="softmax",
    name="class_probabilities"
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs,
    name="Advanced_Iris_1D_CNN"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL ARCHITECTURE")
print("=" * 70)
model.summary()

# Display model architecture
try:
    tf.keras.utils.plot_model(
        model,
        to_file="iris_cnn_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        dpi=120
    )
    from IPython.display import Image, display
    display(Image(filename="iris_cnn_architecture.png"))
except Exception as error:
    print("Architecture visualization unavailable:", error)

# ============================================================
# 6. TRAINING CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="best_iris_cnn.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ============================================================
# 7. TRAIN THE MODEL
# ============================================================

print("\nSTARTING CNN TRAINING")
print("=" * 70)

history = model.fit(
    X_train_cnn,
    y_train_encoded,
    epochs=150,
    batch_size=16,
    validation_split=0.20,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1,
    shuffle=True
)

# ============================================================
# 8. EVALUATE THE MODEL
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test_cnn,
    y_test_encoded,
    verbose=0
)

y_prob = model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

print("\nFINAL TEST RESULTS")
print("=" * 70)
print(f"Test loss:        {test_loss:.4f}")
print(f"Test accuracy:    {test_accuracy:.4f}")
print(f"Accuracy score:   {accuracy:.4f}")
print(f"Macro precision:  {precision:.4f}")
print(f"Macro recall:     {recall:.4f}")
print(f"Macro F1-score:   {f1:.4f}")

# ============================================================
# 9. TRAINING AND VALIDATION CURVES
# ============================================================

epochs_ran = range(1, len(history.history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(
    epochs_ran,
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

axes[0].plot(
    epochs_ran,
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

axes[0].set_title("CNN Accuracy Across Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    epochs_ran,
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

axes[1].plot(
    epochs_ran,
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

axes[1].set_title("CNN Loss Across Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Categorical Cross-Entropy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 10. LEARNING RATE HISTORY
# ============================================================

learning_rates = history.history.get("learning_rate")

if learning_rates is not None:
    plt.figure(figsize=(9, 4))
    plt.plot(
        range(1, len(learning_rates) + 1),
        learning_rates,
        marker="o"
    )
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Adaptation During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# 11. VISUALIZE LEARNED CONVOLUTIONAL FILTERS
# ============================================================

for layer_name in [
    "conv_block_1",
    "conv_block_2",
    "conv_block_3"
]:
    conv_layer = model.get_layer(layer_name)
    weights = conv_layer.get_weights()[0]

    n_filters = weights.shape[-1]
    n_channels = weights.shape[1]

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(12, 3 * n_channels),
        squeeze=False
    )

    for channel in range(n_channels):
        ax = axes[channel, 0]

        for filter_index in range(n_filters):
            ax.plot(
                weights[:, channel, filter_index],
                alpha=0.55,
                linewidth=1
            )

        ax.set_title(
            f"{layer_name}: Learned Kernel Weights, "
            f"Input Channel {channel + 1}"
        )
        ax.set_xlabel("Kernel Position")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

# ============================================================
# 12. VISUALIZE INTERMEDIATE FEATURE MAPS
# ============================================================

feature_layer_names = ["relu_1", "relu_2", "relu_3"]

feature_model = Model(
    inputs=model.input,
    outputs=[
        model.get_layer(name).output
        for name in feature_layer_names
    ]
)

sample_index = 0
sample = X_test_cnn[sample_index:sample_index + 1]

feature_maps = feature_model.predict(sample, verbose=0)

print("\nActual class:", class_names[y_test[sample_index]])
print("Predicted class:", class_names[y_pred[sample_index]])

for layer_name, activations in zip(
    feature_layer_names,
    feature_maps
):
    activations = activations[0]
    number_to_display = min(16, activations.shape[-1])

    fig, axes = plt.subplots(4, 4, figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        if i < number_to_display:
            ax.plot(
                activations[:, i],
                marker="o",
                linewidth=1.5
            )
            ax.set_title(f"Filter {i + 1}")
            ax.set_xlabel("Feature Position")
            ax.set_ylabel("Activation")
            ax.grid(True, alpha=0.2)
        else:
            ax.axis("off")

    plt.suptitle(
        f"Intermediate Feature Maps: {layer_name}",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

# ============================================================
# 13. GLOBAL POOLING VISUALIZATION
# ============================================================

pooling_model = Model(
    inputs=model.input,
    outputs=[
        model.get_layer("global_max_pooling").output,
        model.get_layer("global_average_pooling").output
    ]
)

max_pool_values, avg_pool_values = pooling_model.predict(
    sample,
    verbose=0
)

max_pool_values = max_pool_values[0]
avg_pool_values = avg_pool_values[0]

pool_indices = np.arange(len(max_pool_values))

plt.figure(figsize=(14, 5))

plt.plot(
    pool_indices,
    max_pool_values,
    marker="o",
    label="Global Max Pooling"
)

plt.plot(
    pool_indices,
    avg_pool_values,
    marker="s",
    label="Global Average Pooling"
)

plt.title("Comparison of Global Pooling Representations")
plt.xlabel("Feature Channel")
plt.ylabel("Activation Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(9, 7))

ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=1.2,
    linecolor="white",
    square=True,
    cbar=True,
    annot_kws={"size": 16, "weight": "bold"}
)

for i in range(num_classes):
    for j in range(num_classes):
        if i == j:
            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor="green",
                    linewidth=2.5
                )
            )
        elif cm[i, j] > 0:
            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.5
                )
            )

plt.title(
    "1D CNN Confusion Matrix\n"
    "Green Outline: Correct | Red Outline: Misclassified"
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ============================================================
# 16. PREDICTION PROBABILITY VISUALIZATION
# ============================================================

sample_number = min(10, len(X_test_cnn))

fig, axes = plt.subplots(
    sample_number,
    1,
    figsize=(11, max(4, 2.8 * sample_number)),
    squeeze=False
)

for i in range(sample_number):
    ax = axes[i, 0]
    probabilities = y_prob[i]

    bars = ax.bar(class_names, probabilities)
    predicted_index = np.argmax(probabilities)

    bars[predicted_index].set_edgecolor("black")
    bars[predicted_index].set_linewidth(2)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability")

    ax.set_title(
        f"Sample {i + 1}: "
        f"Actual = {class_names[y_test[i]]}, "
        f"Predicted = {class_names[y_pred[i]]}"
    )

    for bar, probability in zip(bars, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{probability:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.suptitle(
    "CNN Softmax Probabilities for Test Samples",
    fontsize=15
)
plt.tight_layout()
plt.show()

# ============================================================
# 17. ACTUAL VERSUS PREDICTED CLASS
# ============================================================

sample_indices = np.arange(len(y_test))

plt.figure(figsize=(13, 5))

plt.scatter(
    sample_indices,
    y_test,
    label="Actual Class",
    marker="o",
    s=80
)

plt.scatter(
    sample_indices,
    y_pred,
    label="Predicted Class",
    marker="x",
    s=90
)

plt.yticks(range(num_classes), class_names)
plt.xlabel("Test Sample Index")
plt.ylabel("Flower Class")
plt.title("Actual versus Predicted Iris Classes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 18. CLASS-WISE PERFORMANCE METRICS
# ============================================================

class_precision = precision_score(
    y_test,
    y_pred,
    labels=np.arange(num_classes),
    average=None,
    zero_division=0
)

class_recall = recall_score(
    y_test,
    y_pred,
    labels=np.arange(num_classes),
    average=None,
    zero_division=0
)

class_f1 = f1_score(
    y_test,
    y_pred,
    labels=np.arange(num_classes),
    average=None,
    zero_division=0
)

metrics_df = pd.DataFrame({
    "Class": class_names,
    "Precision": class_precision,
    "Recall": class_recall,
    "F1-score": class_f1
})

print("\nCLASS-WISE PERFORMANCE")
print(metrics_df.round(4).to_string(index=False))

metrics_long = metrics_df.melt(
    id_vars="Class",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=metrics_long,
    x="Class",
    y="Score",
    hue="Metric"
)

plt.ylim(0, 1.05)
plt.title("Class-Wise Classification Performance")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# ============================================================
# 19. MISCLASSIFICATION ANALYSIS
# ============================================================

misclassified_indices = np.where(y_test != y_pred)[0]

print("\nMISCLASSIFICATION ANALYSIS")
print("=" * 70)

if len(misclassified_indices) == 0:
    print("No misclassifications were observed on the test set.")
else:
    for idx in misclassified_indices:
        print(f"\nTest sample index: {idx}")
        print("Actual class:", class_names[y_test[idx]])
        print("Predicted class:", class_names[y_pred[idx]])
        print("Standardized features:", X_test_scaled[idx])
        print("Class probabilities:", np.round(y_prob[idx], 4))

# ============================================================
# 20. MULTICLASS ROC CURVES
# ============================================================

y_test_binary = label_binarize(
    y_test,
    classes=np.arange(num_classes)
)

plt.figure(figsize=(9, 7))

for i in range(num_classes):
    if len(np.unique(y_test_binary[:, i])) < 2:
        print(
            f"ROC curve skipped for {class_names[i]}: "
            "test labels contain only one binary outcome."
        )
        continue

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        y_prob[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_names[i]} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves for Iris Classification")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL CNN EXPERIMENT SUMMARY")
print("=" * 70)

print(f"Dataset samples:       {len(X)}")
print(f"Training samples:      {len(X_train)}")
print(f"Testing samples:       {len(X_test)}")
print(f"Number of classes:     {num_classes}")
print(f"Epochs executed:       {len(history.history['loss'])}")
print(f"Final test accuracy:   {accuracy:.4f}")
print(f"Macro precision:       {precision:.4f}")
print(f"Macro recall:          {recall:.4f}")
print(f"Macro F1-score:        {f1:.4f}")
print(f"Correct predictions:   {np.sum(y_test == y_pred)}")
print(f"Incorrect predictions: {np.sum(y_test != y_pred)}")

print("\nExperiment completed successfully!")

# ============================================================
# 22. TEST A NEW IRIS FLOWER
# ============================================================

print("\nNEW FLOWER PREDICTION")
print("=" * 70)

# Example measurements in centimeters:
# Sepal length, Sepal width, Petal length, Petal width

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

new_flower_scaled = scaler.transform(new_flower)

new_flower_cnn = new_flower_scaled.reshape(
    1, new_flower_scaled.shape[1], 1
)

new_probabilities = model.predict(new_flower_cnn, verbose=0)

predicted_class = np.argmax(new_probabilities[0])

print("Flower measurements:", new_flower[0])
print("Predicted species:", class_names[predicted_class])

print("\nClass probabilities:")

for name, probability in zip(class_names, new_probabilities[0]):
    print(f"{name}: {probability:.4f} ({probability * 100:.2f}%)")
