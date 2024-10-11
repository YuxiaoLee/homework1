import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

def estimate_gaussian_params(X, y, num_classes):
    means = []
    covariances = []
    priors = []
    for i in range(num_classes):
        X_class = X[y == i]
        mean = np.mean(X_class, axis=0)
        covariance = np.cov(X_class, rowvar=False)
        lambda_reg = 0.01
        covariance += lambda_reg * np.eye(covariance.shape[0])
        means.append(mean)
        covariances.append(covariance)
        priors.append(len(X_class) / len(X))
    return np.array(means), np.array(covariances), np.array(priors)

def minimum_error_classifier(X, means, covariances, priors, num_classes):
    likelihoods = np.zeros((X.shape[0], num_classes))
    for i in range(num_classes):
        likelihoods[:, i] = multivariate_normal.pdf(X, mean=means[i], cov=covariances[i]) * priors[i]
    predictions = np.argmax(likelihoods, axis=1)
    return predictions

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
resp = urlopen(url)
har_zip = ZipFile(BytesIO(resp.read()))

X_train = pd.read_csv(har_zip.open('UCI HAR Dataset/train/X_train.txt'), delim_whitespace=True, header=None)
X_test = pd.read_csv(har_zip.open('UCI HAR Dataset/test/X_test.txt'), delim_whitespace=True, header=None)

y_train = pd.read_csv(har_zip.open('UCI HAR Dataset/train/y_train.txt'), delim_whitespace=True, header=None)
y_test = pd.read_csv(har_zip.open('UCI HAR Dataset/test/y_test.txt'), delim_whitespace=True, header=None)

le_har = LabelEncoder()
y_train_har = le_har.fit_transform(y_train)
y_test_har = le_har.transform(y_test)

scaler = StandardScaler()
X_train_har = scaler.fit_transform(X_train)
X_test_har = scaler.transform(X_test)

num_classes_har = len(np.unique(y_train_har))
means_har, covariances_har, priors_har = estimate_gaussian_params(X_train_har, y_train_har, num_classes_har)

y_pred_har = minimum_error_classifier(X_test_har, means_har, covariances_har, priors_har, num_classes_har)

conf_matrix_har = confusion_matrix(y_test_har, y_pred_har)

unique_labels_har = np.unique(y_test_har)

fig, ax = plt.subplots(figsize=(8, 8))
disp_har = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_har, display_labels=unique_labels_har)
disp_har.plot(cmap='plasma', ax=ax, colorbar=True)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title('Confusion Matrix - Human Activity Recognition Dataset')
plt.show()

pca_har = PCA(n_components=2)
X_har_pca = pca_har.fit_transform(X_test_har)

plt.figure(figsize=(8,6))
plt.scatter(X_har_pca[:, 0], X_har_pca[:, 1], c=y_test_har, cmap='viridis', edgecolor='k', s=40)
plt.title('PCA - Human Activity Recognition Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
