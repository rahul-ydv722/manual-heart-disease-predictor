import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the data
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)  # Features
y = df["target"]  # Label

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model to model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
