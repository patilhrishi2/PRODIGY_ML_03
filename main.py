import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            images.append(img)
            if 'dog' in filename.lower():
                labels.append(1) 
            elif 'cat' in filename.lower():
                labels.append(0)  
    return np.array(images), np.array(labels)


def load_test_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            images.append(img)
            filenames.append(filename)
    return np.array(images), filenames

folder = 'train'
images, labels = load_images_from_folder(folder)

num_samples, height, width = images.shape
X = images.reshape(num_samples, height * width)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

test_folder = 'test1'
test_images, test_filenames = load_test_images(test_folder)

num_samples, height, width = test_images.shape
X_test1 = test_images.reshape(num_samples, height * width)

y_test1_pred = model.predict(X_test1)

test_results = pd.DataFrame({'filename': test_filenames, 'label': y_test1_pred})

test_results.to_csv('test1_labels.csv', index=False)