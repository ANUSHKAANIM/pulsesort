import numpy as np
from sklearn.preprocessing import LabelEncoder
from .feature_extraction import extract_features

def prepare_dataset(data, window_size=50):
    X = []
    y = []

    for i in range(0, len(data) - window_size, window_size):
        ch1 = data["ch1"].values[i:i+window_size]
        ch2 = data["ch2"].values[i:i+window_size]

        features = []
        features.extend(extract_features(ch1))
        features.extend(extract_features(ch2))

        X.append(features)
        y.append(data["label"].iloc[i])

    X = np.array(X)
    y = np.array(y)

    # Encode string labels → numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y, encoder
