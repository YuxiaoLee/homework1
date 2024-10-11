import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, eig

# Ensure results are reproducible by fixing the seed
np.random.seed(5)

# Configuration constants
NUM_FEATURES = 4
NUM_SAMPLES = 10000
NUM_CLASSES = 2

# Defining class means
means = np.ones((NUM_CLASSES, NUM_FEATURES))
means[0, :] = [-1, -1, -1, -1]

# Defining class covariance matrices
covariances = np.zeros((NUM_CLASSES, NUM_FEATURES, NUM_FEATURES))
covariances[0,:,:] = [[2, -0.5, 0.3, 0],
                  [-0.5, 1, -0.5, 0],
                  [0.3, -0.5, 1, 0],
                  [0, 0, 0, 2]]
covariances[1,:,:] = [[1, 0.3, -0.2, 0],
                  [0.3, 2, 0.3, 0],
                  [-0.2, 0.3, 1, 0],
                  [0, 0, 0, 3]]

# Class priors
class_priors = [0.65, 0.35]

# Assign class labels based on priors
labels = (np.random.rand(NUM_SAMPLES) >= class_priors[1]).astype(int)

# Generate samples for each class based on their respective means and covariances
data = np.zeros((NUM_SAMPLES, NUM_FEATURES))
for i in range(NUM_SAMPLES):
    class_idx = labels[i]
    data[i] = np.random.multivariate_normal(means[class_idx], covariances[class_idx])

# Calculate between-class and within-class scatter matrices
mean_diff = means[0] - means[1]
between_class_scatter = np.outer(mean_diff, mean_diff)
within_class_scatter = covariances[0] + covariances[1]

# Solve generalized eigenvalue problem for LDA
eigvals, eigvecs = eig(inv(within_class_scatter).dot(between_class_scatter))
lda_vector = eigvecs[:, np.argmax(eigvals)]

# Project data onto the LDA vector
projected_class_0 = data[labels == 0].dot(lda_vector)
projected_class_1 = data[labels == 1].dot(lda_vector)

# Combine projections for threshold computation
projected_all = np.concatenate([projected_class_0, projected_class_1])
sorted_projection = np.sort(projected_all)
thresholds = (sorted_projection[:-1] + sorted_projection[1:]) / 2

# Initialize lists to store classification metrics
TPR = []
FPR = []
min_error = []

# Perform classification for each threshold and compute the corresponding errors
for threshold in thresholds:
    decisions = projected_all >= threshold
    true_positives = np.sum((decisions == 1) & (labels == 1)) / np.sum(labels == 1)
    false_positives = np.sum((decisions == 1) & (labels == 0)) / np.sum(labels == 0)

    TPR.append(true_positives)
    FPR.append(false_positives)

    error = class_priors[0] * false_positives + class_priors[1] * (1 - true_positives)
    min_error.append(error)

# Ideal decision threshold and corresponding error
gamma_ideal = class_priors[0] / class_priors[1]
ideal_decision = projected_all >= gamma_ideal
TP_ideal = np.sum((ideal_decision == 1) & (labels == 1)) / np.sum(labels == 1)
FP_ideal = np.sum((ideal_decision == 1) & (labels == 0)) / np.sum(labels == 0)
min_error_ideal = class_priors[0] * FP_ideal + class_priors[1] * (1 - TP_ideal)

# Output ideal results
print(f"Ideal Threshold: {gamma_ideal:.6f}, Minimum Error: {min_error_ideal:.6f}")

# Plot the ROC curve
plt.plot(FPR, TPR, 'r-', label='ROC Curve')
plt.plot(FPR[np.argmin(min_error)], TPR[np.argmin(min_error)], 'bo', label='Min Error Point')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
