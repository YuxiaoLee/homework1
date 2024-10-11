# Reference from Mark Zolotas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Custom function to generate data from a Gaussian Mixture Model (GMM)
def generate_data_from_gmm(N, pdf_params):
    n = len(pdf_params['mu'][0])  # Dimensionality of the vectors
    X = np.zeros([N, n])
    labels = np.zeros(N, dtype=int)

    # Randomly assign samples to each component
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0)  # Insert 0 to set intervals for class membership

    for l in range(len(pdf_params['priors'])):
        # Get indices of the samples assigned to this component
        indices = np.argwhere((thresholds[l] <= u) & (u < thresholds[l + 1])).flatten()
        Nl = len(indices)
        labels[indices] = l

        # Generate samples for this component
        if l == 2:  # Class 3 is a mixture of 2 Gaussians with equal weights
            half_Nl = Nl // 2
            X[indices[:half_Nl], :] = multivariate_normal.rvs(mean=pdf_params['mu'][2][0],
                                                              cov=pdf_params['Sigma'][2][0], size=half_Nl)
            X[indices[half_Nl:], :] = multivariate_normal.rvs(mean=pdf_params['mu'][2][1],
                                                              cov=pdf_params['Sigma'][2][1], size=Nl - half_Nl)
        else:  # Classes 1 and 2 have a single Gaussian component
            X[indices, :] = multivariate_normal.rvs(mean=pdf_params['mu'][l], cov=pdf_params['Sigma'][l], size=Nl)

    return X, labels


def perform_erm_classification(X, Lambda, gmm_pdf, C):
    priors = gmm_pdf['priors']
    mu = gmm_pdf['mu']
    Sigma = gmm_pdf['Sigma']

    decisions = np.zeros(X.shape[0], dtype=int)

    for i, x in enumerate(X):
        risks = np.zeros(C)
        for j in range(C):
            if j == 2:  # Class 3 is a mixture of 2 Gaussians
                p_x_given_class = 0.5 * multivariate_normal.pdf(x, mean=mu[2][0], cov=Sigma[2][0]) + \
                                  0.5 * multivariate_normal.pdf(x, mean=mu[2][1], cov=Sigma[2][1])
            else:
                p_x_given_class = multivariate_normal.pdf(x, mean=mu[j], cov=Sigma[j])
            for k in range(C):
                risks[k] += Lambda[k, j] * priors[j] * p_x_given_class
        decisions[i] = np.argmin(risks)

    return decisions

# Parameters for the GMM
N = 10000  # Number of samples
n = 3  # Dimensionality of input random vectors
C = 3  # Number of classes

gmm_pdf = {}

# Class priors
gmm_pdf['priors'] = np.array([0.3, 0.3, 0.4])

# Set mean vectors for each class
gmm_pdf['mu'] = [
    np.array([0, 0, 1]),  # Mean for class 1
    np.array([2, 0, 0]),  # Mean for class 2
    [np.array([0, 2, 0]), np.array([2, 2, 0])]  # Mixture of 2 Gaussians for class 3
]

# Set covariance matrices for each class
gmm_pdf['Sigma'] = [
    np.array([[1, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 1]]),  # Class 1 covariance
    np.array([[1, 0.1, 0], [0.1, 1, 0.1], [0, 0.1, 1]]),  # Class 2 covariance
    [np.array([[1, 0.3, 0], [0.3, 1, 0.3], [0, 0.3, 1]]),  # Class 3: first Gaussian covariance
     np.array([[1, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 1]])]  # Class 3: second Gaussian covariance
]

# Generate data from the GMM
X, labels = generate_data_from_gmm(N, gmm_pdf)

# Loss matrix
Lambda = np.array([[0, 100, 100],
                   [1, 0, 100],
                   [1, 1, 0]])

# Perform ERM classification
decisions = perform_erm_classification(X, Lambda, gmm_pdf, C)

# Compute confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
conf_display = ConfusionMatrixDisplay.from_predictions(decisions, labels, display_labels=['1', '2', '3'],
                                                       colorbar=False)
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Number of Misclassified Samples: {:d}".format(N - correct_class_samples))

prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

# 3D Plot for decisions vs true labels
fig_map = plt.figure(figsize=(10, 10))
ax = fig_map.add_subplot(111, projection='3d')
L = np.arange(C)
marker_shapes = ['.', 'o', '^']
ax_map = fig_map.add_subplot(111, projection='3d')
for r in L:  # Each decision option
    for c in L:  # Each true class
        ind_rc = np.argwhere((decisions == r) & (labels == c)).flatten()
        if len(ind_rc) > 0:
            ax_map.scatter(X[ind_rc, 0], X[ind_rc, 1], X[ind_rc, 2],
                           marker=marker_shapes[r],
                           color='g' if r == c else 'r',
                           label=f"D={r + 1} | L={c + 1}")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
plt.title("Classification Decisions: Marker Shape/Predictions, Color/True Labels")
plt.tight_layout()
plt.legend()
plt.show()
