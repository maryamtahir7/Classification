"""
Project 2: Data Classification Using AI
-----------------------------------------
Build a basic classification model using a small dataset.

Key Requirements covered:
- Load and understand a dataset
- Split data into training and testing sets
- Apply a simple classification algorithm

Key Skills demonstrated:
- Data handling
- Supervised learning basics
- Model training

Dataset used: Iris flower dataset (built into scikit-learn)
This is a classic small dataset with 150 samples of iris flowers,
each described by 4 features (sepal length, sepal width, petal length,
petal width) and labeled with one of 3 species (setosa, versicolor,
virginica).
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_understand_dataset():
    """
    Step 1: Load and understand the dataset.
    Loads the Iris dataset and prints basic info about it.
    """
    print("=" * 60)
    print("STEP 1: Load and Understand the Dataset")
    print("=" * 60)

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df["species"].map(
        {i: name for i, name in enumerate(iris.target_names)}
    )

    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFeatures: {list(iris.feature_names)}")
    print(f"Target classes: {list(iris.target_names)}")

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nBasic statistics:")
    print(df.describe())

    print("\nClass distribution:")
    print(df["species_name"].value_counts())

    return df, iris


def split_dataset(iris):
    """
    Step 2: Split data into training and testing sets.
    Uses an 80/20 train-test split.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Split Data into Training and Testing Sets")
    print("=" * 60)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTotal samples: {len(X)}")
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples: {len(X_test)} (20%)")

    return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train):
    """
    Step 3: Apply a simple classification algorithm.
    Uses a Decision Tree Classifier (easy to understand and interpret).
    """
    print("\n" + "=" * 60)
    print("STEP 3: Apply a Simple Classification Algorithm")
    print("=" * 60)

    print("\nAlgorithm: Decision Tree Classifier")
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete!")

    return model


def evaluate_model(model, X_test, y_test, target_names):
    """
    Step 4 (bonus): Evaluate the trained model on the test set.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Evaluate the Model")
    print("=" * 60)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def predict_new_sample(model, target_names):
    """
    Bonus: Demonstrate predicting a brand-new, unseen flower sample.
    """
    print("\n" + "=" * 60)
    print("BONUS: Predict a New, Unseen Sample")
    print("=" * 60)

    # Example measurements: sepal length, sepal width, petal length, petal width
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)
    predicted_species = target_names[prediction[0]]

    print(f"\nSample measurements: {sample[0]}")
    print(f"Predicted species: {predicted_species}")


def main():
    df, iris = load_and_understand_dataset()
    X_train, X_test, y_train, y_test = split_dataset(iris)
    model = train_classifier(X_train, y_train)
    evaluate_model(model, X_test, y_test, iris.target_names)
    predict_new_sample(model, iris.target_names)

    print("\n" + "=" * 60)
    print("Project 2 complete! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
