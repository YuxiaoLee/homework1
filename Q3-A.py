# Reference from Mark Zolotas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import multivariate_normal

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

wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
wine_df = pd.read_csv(wine_url, delimiter=';')

X_wine = wine_df.iloc[:, :-1].values
y_wine = wine_df.iloc[:, -1].values
le_wine = LabelEncoder()
y_wine = le_wine.fit_transform(y_wine)

scaler = StandardScaler()
X_wine = scaler.fit_transform(X_wine)

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

num_classes_wine = len(np.unique(y_train_wine))
means_wine, covariances_wine, priors_wine = estimate_gaussian_params(X_train_wine, y_train_wine, num_classes_wine)

y_pred_wine = minimum_error_classifier(X_test_wine, means_wine, covariances_wine, priors_wine, num_classes_wine)

conf_matrix_wine = confusion_matrix(y_test_wine, y_pred_wine)

unique_labels = np.unique(y_test_wine)

fig, ax = plt.subplots(figsize=(8, 8))
disp_wine = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_wine, display_labels=unique_labels)
disp_wine.plot(cmap='plasma', ax=ax, colorbar=True)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title('Confusion Matrix - Wine Quality Dataset')
plt.show()
from sklearn.decomposition import PCA

pca_wine = PCA(n_components=2)
X_wine_pca = pca_wine.fit_transform(X_wine)

plt.figure(figsize=(8,6))
plt.scatter(X_wine_pca[:, 0], X_wine_pca[:, 1], c=y_wine, cmap='viridis', edgecolor='k', s=40)
plt.title('PCA - Wine Quality Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
