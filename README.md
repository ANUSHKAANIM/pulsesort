# PulseSort ⚡  
EMG-Based Hand & Finger Gesture Recognition using Machine Learning

PulseSort is a machine learning project that recognizes hand and finger gestures using Electromyography (EMG) signals. The system processes raw EMG data, extracts features, and compares multiple machine learning models to classify gestures.

Project Overview  
PulseSort uses EMG signal data stored in CSV files to classify 12 different hand and finger gestures. The project implements and compares Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Logistic Regression models using a modular machine learning pipeline consisting of data loading, preprocessing, training, and evaluation.

Gesture Classes  
clenched, fist, four, index_finger, okay, peace, rest, rock, spread, three, thumb, up

Project Structure  
PulseSort/  
│── data/  
│   └── EMG/  
│       └── *.csv  
│── src/  
│   ├── load_data.py  
│   ├── prepare_dataset.py  
│   ├── feature_extraction.py  
│   ├── train_models.py  
│   ├── evaluate.py  
│── main.py  
│── requirements.txt  
│── README.md  

Methodology  
All EMG CSV files are loaded and combined programmatically. Gesture labels are encoded numerically and the data is segmented using a sliding window approach. The dataset is split into training and testing sets in an 80:20 ratio. Feature scaling is applied using StandardScaler after the split to prevent data leakage. Three classifiers—SVM, KNN, and Logistic Regression—are trained using the same dataset and evaluated using classification accuracy.

How to Run  
Install dependencies using:  
pip install -r requirements.txt  

Run the project using:  
python main.py  

Results  
SVM Accuracy: ~0.08  
KNN Accuracy: ~0.08  
Logistic Regression Accuracy: ~0.08  

The accuracy is close to random guessing (1/12 ≈ 0.083), indicating that raw EMG signals alone are insufficient for reliable gesture recognition.

Key Learnings  
The project demonstrates the importance of feature extraction in EMG-based classification and highlights the need for advanced signal processing techniques to improve performance.

Future Work  
Planned improvements include time-domain feature extraction (RMS, MAV, Zero Crossing, Waveform Length), frequency-domain analysis, deep learning models such as CNNs and LSTMs, and real-time EMG signal acquisition.

Technologies Used  
Python, NumPy, Pandas, Scikit-learn, VS Code

Acknowledgement  
This project was developed as part of an academic learning initiative to understand EMG signal processing and machine learning-based gesture recognition.
