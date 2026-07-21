Project 2: Data Classification Using AI 🌸
Goal
Build a basic classification model using a small dataset.
Key Requirements
✅ Load and understand a dataset
✅ Split data into training and testing sets
✅ Apply a simple classification algorithm
Key Skills Demonstrated
Data handling
Supervised learning basics
Model training
Dataset
This project uses the Iris flower dataset — a classic, beginner-friendly dataset built into scikit-learn. It contains 150 samples of iris flowers, each with 4 measurements (sepal length, sepal width, petal length, petal width) and a label for one of 3 species: setosa, versicolor, virginica.
No external file needs to be downloaded — it loads directly from the sklearn.datasets module.
Files
classification.py — Main script
requirements.txt — Python packages needed
How to Run
Install the required packages:
pip install -r requirements.txt
Run the script:
python3 classification.py
What the Script Does
Load and understand the dataset — loads Iris data into a pandas DataFrame, prints its shape, features, target classes, summary statistics, and class distribution.
Split the data — splits samples into 80% training / 20% testing using train_test_split, keeping class proportions balanced.
Apply a classification algorithm — trains a DecisionTreeClassifier (simple and easy to interpret) on the training data.
Evaluate the model — checks accuracy, precision/recall, and a confusion matrix on the unseen test data (~96–97% accuracy).
Bonus: Predict a new sample — shows how the trained model can classify a brand-new flower measurement it has never seen before.
Possible Extensions (Optional Ideas)
Try a different algorithm (KNN, Logistic Regression, Random Forest)
Visualize the decision tree or feature importance
Use a different small dataset (e.g. Wine dataset, your own CSV)
Add cross-validation instead of a single train/test split