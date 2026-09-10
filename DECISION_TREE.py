# ============================================================
# DECISION TREE CLASSIFICATION USING IRIS DATASET
# ============================================================

# ------------------------------------------------------------
# 1. Import Required Libraries
# ------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 2. Load the Iris Dataset
# ------------------------------------------------------------

iris = load_iris()

# Display dataset information
print("========== IRIS DATASET ==========")
print("Total Samples:", len(iris.data))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)


# ------------------------------------------------------------
# 3. Separate Features and Target
# ------------------------------------------------------------

X = iris.data
y = iris.target

print("\n========== DATA INFORMATION ==========")
print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# ------------------------------------------------------------
# 4. Split Dataset into Training and Testing Data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ------------------------------------------------------------
# 5. Create Decision Tree Classifier
# ------------------------------------------------------------

model = DecisionTreeClassifier(
    random_state=42
)


# ------------------------------------------------------------
# 6. Train the Decision Tree Model
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# ------------------------------------------------------------
# 7. Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 8. Display Predictions
# ------------------------------------------------------------

print("\n========== PREDICTIONS ==========")

for i in range(len(y_test)):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]

    print(
        f"Sample {i + 1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )


# ------------------------------------------------------------
# 9. Calculate Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# ------------------------------------------------------------
# 10. Generate Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ------------------------------------------------------------
# 11. Display Confusion Matrix Graph
# ------------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("Confusion Matrix - Iris Dataset")
plt.show()


# ------------------------------------------------------------
# 12. Visualize the Decision Tree
# ------------------------------------------------------------

plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Iris Dataset")
plt.show()


# ------------------------------------------------------------
# 13. Scatter Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset - Sepal Length vs Sepal Width")

plt.show()


# ------------------------------------------------------------
# 14. Test the Model with a New Flower
# ------------------------------------------------------------

# Example:
# [Sepal Length, Sepal Width, Petal Length, Petal Width]

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

predicted_class = iris.target_names[prediction[0]]

print("\n========== NEW FLOWER PREDICTION ==========")
print("Input:", new_flower)
print("Predicted Species:", predicted_class)


# ------------------------------------------------------------
# 15. Final Summary
# ------------------------------------------------------------

print("\n==========================================")
print("       DECISION TREE EXPERIMENT")
print("==========================================")
print("Dataset           : Iris Dataset")
print("Total Samples     : 150")
print("Training Samples  :", len(X_train))
print("Testing Samples   :", len(X_test))
print("Algorithm         : Decision Tree Classifier")
print("Accuracy          :", accuracy * 100, "%")
print("==========================================")
