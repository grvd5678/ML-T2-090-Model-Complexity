import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load our classic tabular dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# --- The Simple Baseline ---
start_time = time.time()
simple_model = LogisticRegression(max_iter=10000)
simple_model.fit(X_train, y_train)
simple_preds = simple_model.predict(X_test)
simple_time = time.time() - start_time
simple_acc = accuracy_score(y_test, simple_preds)

# --- The Complex Model ---
start_time = time.time()
complex_model = RandomForestClassifier(n_estimators=100, random_state=42)
complex_model.fit(X_train, y_train)
complex_preds = complex_model.predict(X_test)
complex_time = time.time() - start_time
complex_acc = accuracy_score(y_test, complex_preds)

# Let's compare!
print(f"Simple Model (Logistic Regression) - Accuracy: {simple_acc:.4f}, Time: {simple_time:.4f}s")
print(f"Complex Model (Random Forest)     - Accuracy: {complex_acc:.4f}, Time: {complex_time:.4f}s")

# --- The Visualization (Old-School Charting!) ---
labels = ['Logistic Regression', 'Random Forest']
accuracies = [simple_acc, complex_acc]
times = [simple_time, complex_time]

# Set up a figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Chart 1: Accuracy
ax1.bar(labels, accuracies, color=['#7f8c8d', '#e74c3c'])
ax1.set_ylabel('Accuracy Score')
ax1.set_title('Predictive Accuracy')
ax1.set_ylim(0.9, 1.0) # Zooming in so the difference is obvious!

# Chart 2: Time
ax2.bar(labels, times, color=['#7f8c8d', '#e74c3c'])
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Engineering Cost (Time)')

plt.tight_layout()

# Save it directly into your docs folder for the final paper
plt.savefig('docs/baseline_tradeoff.png')
print("Chart successfully generated and saved to docs/baseline_tradeoff.png!")

# Pop it open on your screen so we can admire it
plt.show()