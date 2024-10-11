import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.set_printoptions(threshold=np.inf)
plt.rcParams['figure.figsize'] = [9, 9]
num_features = 4
num_samples = 10000
num_classes = 2
means = np.ones((num_classes, num_features))
means[0, :] = [-1, -1, -1, -1]
covariances = np.ones((num_classes, num_features, num_features))
covariances[0] = [[2, -0.5, 0.3, 0],
                  [-0.5, 1, -0.5, 0],
                  [0.3, -0.5, 1, 0],
                  [0, 0, 0, 2]]
covariances[1] = [[1, 0.3, -0.2, 0],
                  [0.3, 2, 0.3, 0],
                  [-0.2, 0.3, 1, 0],
                  [0, 0, 0, 3]]

# Set random seed for reproducibility
np.random.seed(5)
class_priors = [0.65, 0.35]
assigned_labels = (np.random.rand(num_samples) >= class_priors[1]).astype(int)

samples = np.zeros((num_samples, num_features))
samples[assigned_labels == 0] = np.random.multivariate_normal(
    means[0], covariances[0], size=np.sum(assigned_labels == 0)
)
samples[assigned_labels == 1] = np.random.multivariate_normal(
    means[1], covariances[1], size=np.sum(assigned_labels == 1)
)
logpdf_class0 = np.log(multivariate_normal.pdf(samples, mean=means[0], cov=covariances[0]))
logpdf_class1 = np.log(multivariate_normal.pdf(samples, mean=means[1], cov=covariances[1]))
discriminant = logpdf_class1 - logpdf_class0
sorted_discriminant = np.sort(discriminant)
thresholds = (sorted_discriminant[:-1] + sorted_discriminant[1:]) / 2.0
tpr = np.zeros(len(thresholds))
fpr = np.zeros(len(thresholds))
min_error = np.zeros(len(thresholds))
for i, threshold in enumerate(thresholds):
    decisions = discriminant >= threshold
    tpr[i] = np.sum((decisions == 1) & (assigned_labels == 1)) / np.sum(assigned_labels == 1)
    fpr[i] = np.sum((decisions == 1) & (assigned_labels == 0)) / np.sum(assigned_labels == 0)
    min_error[i] = class_priors[0] * fpr[i] + class_priors[1] * (1 - tpr[i])
log_gamma_ideal = np.log(class_priors[0] / class_priors[1])
ideal_decisions = (discriminant >= log_gamma_ideal)
tpr_ideal = np.sum((ideal_decisions == 1) & (assigned_labels == 1)) / np.sum(assigned_labels == 1)
fpr_ideal = np.sum((ideal_decisions == 1) & (assigned_labels == 0)) / np.sum(assigned_labels == 0)
min_error_ideal = class_priors[0] * fpr_ideal + class_priors[1] * (1 - tpr_ideal)
print(f"Theoretical Gamma: {np.exp(log_gamma_ideal):.6f}, Ideal Minimum Error: {min_error_ideal:.6f}")
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
min_error_idx = np.argmin(min_error)
plt.plot(fpr[min_error_idx], tpr[min_error_idx], 'o', color='red', label='Min Error Point')
plt.legend()
plt.show()
print(f"Empirical Gamma: {np.exp(thresholds[min_error_idx]):.6f}, Empirical Minimum Error: {min(min_error):.6f}")
fig, ax = plt.subplots()
class0_scatter = ax.scatter(samples[assigned_labels == 0, 3], samples[assigned_labels == 0, 1], marker='x', color='blue', label="Class 0")
class1_scatter = ax.scatter(samples[assigned_labels == 1, 3], samples[assigned_labels == 1, 1], marker='o', color='red', label="Class 1")
ax.set_xlabel('X3')
ax.set_ylabel('X1')
plt.title('2D Data Distribution (X3 vs X1)')
ax.legend()
plt.show()
# reference from Mark Zolotas