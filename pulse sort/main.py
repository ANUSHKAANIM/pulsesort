from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.load_data import load_emg_data
from src.prepare_dataset import prepare_dataset
from src.train_models import train_svm, train_knn, train_lr
from src.evaluate import evaluate

# Load ALL EMG gesture CSVs
data = load_emg_data("data/EMG")

# Feature extraction
X, y, encoder = prepare_dataset(data, window_size=50)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
svm = train_svm(X_train, y_train)
knn = train_knn(X_train, y_train)
lr  = train_lr(X_train, y_train)

# Evaluate
print("SVM Accuracy:", evaluate(svm, X_test, y_test))
print("KNN Accuracy:", evaluate(knn, X_test, y_test))
print("Logistic Regression Accuracy:", evaluate(lr, X_test, y_test))

print("Gesture classes:", encoder.classes_)
