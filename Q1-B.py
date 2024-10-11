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
covariances[0, :, :] = [[2, -0.5, 0.3, 0],
                         [-0.5, 1, -0.5, 0],
                         [0.3, -0.5, 1, 0],
                         [0, 0, 0, 2]]
covariances[1, :, :] = [[1, 0.3, -0.2, 0],
                         [0.3, 2, 0.3, 0],
                         [-0.2, 0.3, 1, 0],
                         [0, 0, 0, 3]]


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
log_pdf0 = np.log(multivariate_normal.pdf(samples, mean=means[0], cov=np.eye(num_features)))
log_pdf1 = np.log(multivariate_normal.pdf(samples, mean=means[1], cov=np.eye(num_features)))

disc_scores = log_pdf1 - log_pdf0

sorted_scores = np.sort(disc_scores)
thresholds = (sorted_scores[:-1] + sorted_scores[1:]) / 2.0

tpr = np.zeros(len(thresholds))
fpr = np.zeros(len(thresholds))
min_error = np.zeros(len(thresholds))
for idx, threshold in enumerate(thresholds):
    decisions = disc_scores >= threshold
    tpr[idx] = np.sum((decisions == 1) & (assigned_labels == 1)) / np.sum(assigned_labels == 1)
    fpr[idx] = np.sum((decisions == 1) & (assigned_labels == 0)) / np.sum(assigned_labels == 0)
    min_error[idx] = class_priors[0] * fpr[idx] + class_priors[1] * (1 - tpr[idx])
log_gamma_ideal = np.log(class_priors[0] / class_priors[1])
ideal_decisions = disc_scores >= log_gamma_ideal
tpr_ideal = np.sum((ideal_decisions == 1) & (assigned_labels == 1)) / np.sum(assigned_labels== 1)
fpr_ideal = np.sum((ideal_decisions == 1) & (assigned_labels == 0)) / np.sum(assigned_labels == 0)
min_error_ideal = class_priors[0] * fpr_ideal + class_priors[1] * (1 - tpr_ideal)

print(f"Ideal Gamma: {np.exp(log_gamma_ideal):.6f}, Minimum Theoretical Error: {min_error_ideal:.6f}")

plt.plot(fpr, tpr, color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
min_error_idx = np.argmin(min_error)
plt.plot(fpr[min_error_idx], tpr[min_error_idx], 'o', color='red', label='Min Error')
plt.legend()
plt.show()
