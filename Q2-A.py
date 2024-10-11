# Reference from Mark Zolotas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def generate_data_from_gmm(N, pdf_params):
    # Determine dimensionality from the first Gaussian component (since it's a list of arrays)
    n = len(pdf_params['mu'][0])
    # Output samples and labels
    X = np.zeros([N, n])
    labels = np.zeros(N, dtype=int)

    # Randomly assign samples to each component
    u = np.random.rand(N)
    # Determine the thresholds based on the mixture weights/priors for the GMM
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0)  # Insert 0 to set intervals for class membership

    L = np.arange(len(pdf_params['priors']))  # Now labels will be 0, 1, 2 for 3 classes
    for l in L:
        # Get indices of the samples assigned to this component
        indices = np.argwhere((thresholds[l] <= u) & (u < thresholds[l + 1])).flatten()
        # Number of samples for this component
        Nl = len(indices)
        labels[indices] = l
        # Generate samples for this component
        if l == 2:  # Class 3 is a mixture of 2 Gaussians with equal weights
            half_Nl = Nl // 2
            X[indices[:half_Nl], :] = multivariate_normal.rvs(mean=pdf_params['mu'][2][0], cov=pdf_params['Sigma'][2][0], size=half_Nl)
            X[indices[half_Nl:], :] = multivariate_normal.rvs(mean=pdf_params['mu'][2][1], cov=pdf_params['Sigma'][2][1], size=Nl - half_Nl)
        else:  # Classes 1 and 2 have a single Gaussian component
            X[indices, :] = multivariate_normal.rvs(mean=pdf_params['mu'][l], cov=pdf_params['Sigma'][l], size=Nl)

    return X, labels



# Parameters for the GMM
N = 10000  # Number of samples
n = 3  # Dimensionality of input random vectors
C = 3  # Number of classes

gmm_pdf = {}

# Class priors
gmm_pdf['priors'] = np.array([0.3, 0.3, 0.4])

# Set mean vectors for each class (with 2-3 times the average standard deviation between means)
gmm_pdf['mu'] = np.array([
    [0, 0, 1],  # Mean for class 1
    [2, 0, 0],  # Mean for class 2
    [[0, 2, 0], [2, 2, 0]]  # Mixture of 2 Gaussians for class 3
])

# Set covariance matrices (adjusted to create more spread and overlap)
gmm_pdf['Sigma'] = np.array([
    [[1, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 1]],  # Class 1 covariance
    [[1, 0.1, 0], [0.1, 1, 0.1], [0, 0.1, 1]],  # Class 2 covariance
    [[[1, 0.3, 0], [0.3, 1, 0.3], [0, 0.3, 1]],  # Class 3: first Gaussian covariance
     [[1, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 1]]]  # Class 3: second Gaussian covariance
])

# Generate data from the GMM using the custom function
X, labels = generate_data_from_gmm(N, gmm_pdf)

# Count the number of samples per class
L = np.arange(C)
N_per_l = np.array([sum(labels == l) for l in L])
print("Samples per class:", N_per_l)

# Plot the original data and their true labels
fig = plt.figure(figsize=(10, 10))
ax_gmm = fig.add_subplot(111, projection='3d')

# Plotting data points with markers for different classes
ax_gmm.plot(X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2], 'r.', label="Class 1", markerfacecolor='none')
ax_gmm.plot(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2], 'bo', label="Class 2", markerfacecolor='none')
ax_gmm.plot(X[labels == 2, 0], X[labels == 2, 1], X[labels == 2, 2], 'g^', label="Class 3", markerfacecolor='none')
ax_gmm.set_xlabel(r"$x_1$")
ax_gmm.set_ylabel(r"$x_2$")
ax_gmm.set_zlabel(r"$x_3$")

plt.title("Data and True Class Labels")
plt.legend()
plt.tight_layout()
plt.show()


# Implementing the Empirical Risk Minimization (ERM) classifier
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


# ERM decision rule with 0-1 loss (MAP classifier)
Lambda = np.ones((C, C)) - np.eye(C)  # 0-1 loss matrix

# Perform classification
decisions = perform_erm_classification(X, Lambda, gmm_pdf, C)

# Confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
conf_display = ConfusionMatrixDisplay.from_predictions(decisions, labels, display_labels=['1', '2', '3'],
                                                       colorbar=False)
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Number of Misclassified Samples:", N - correct_class_samples)

# Compute probability of error
prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

# Plot classification decisions vs true labels in 3D
fig_map = plt.figure(figsize=(10, 10))
ax_map = fig_map.add_subplot(111, projection='3d')

# Marker shapes and colors for correct vs incorrect classifications
marker_shapes = ['.', 'o', '^']
for r in L:  # Each decision option
    for c in L:  # Each true class
        ind_rc = np.argwhere((decisions == r) & (labels == c)).flatten()
        if len(ind_rc) > 0:
            ax_map.scatter(X[ind_rc, 0], X[ind_rc, 1], X[ind_rc, 2],
                           marker=marker_shapes[r],
                           color='g' if r == c else 'r',
                           label=f"D={r + 1} | L={c + 1}")

ax_map.set_xlabel(r"$x_1$")
ax_map.set_ylabel(r"$x_2$")
ax_map.set_zlabel(r"$x_3$")
plt.title("Classification Decisions: Marker Shape/Predictions, Color/True Labels")
plt.tight_layout()
plt.legend()
plt.show()
