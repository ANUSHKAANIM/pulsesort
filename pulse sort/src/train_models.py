from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def train_svm(X, y):
    model = SVC(kernel="rbf")
    model.fit(X, y)
    return model

def train_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model

def train_lr(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model
