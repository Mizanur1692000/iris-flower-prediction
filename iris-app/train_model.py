# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier (you can choose different classifiers)
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model (Optional, but good to check accuracy)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file (model.pkl)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model has been trained and saved as 'model.pkl'")