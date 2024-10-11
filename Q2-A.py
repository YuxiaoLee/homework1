# Reference from Mark Zolotas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def generate_data_from_gmm(N, pdf_params):
    n = len(pdf_params['mu'][0])
    X = np.zeros([N, n])
    labels = np.zeros(N, dtype=int)

    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0) 

    L = np.arange(len(pdf_params['priors']))
    for l in L:
        indices = np.argwhere((thresholds[l] <= u) & (u < thresholds[l + 1])).flatten()
        Nl = len(indices)
        labels[indices] = l
        if l == 2:
            half_Nl = Nl // 2
            X[indices[:half_Nl], :] = multivariate_normal.rvs(mean=pdf_params['mu'][2][0], cov=pdf_params['Sigma'][2][0], size=half_Nl)
            X[indices[half_Nl:], :] = multivariate_normal.rvs(mean=pdf_params['mu'][2][1], cov=pdf_params['Sigma'][2][1], size=Nl - half_Nl)
        else:  
            X[indices, :] = multivariate_normal.rvs(mean=pdf_params['mu'][l], cov=pdf_params['Sigma'][l], size=Nl)

    return X, labels
N = 10000  
n = 3 
C = 3 
gmm_pdf = {}

gmm_pdf['priors'] = np.array([0.3, 0.3, 0.4])

gmm_pdf['mu'] = np.array([
    [0, 0, 1], 
    [2, 0, 0],  
    [[0, 2, 0], [2, 2, 0]]  
])

gmm_pdf['Sigma'] = np.array([
    [[1, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 1]], 
    [[1, 0.1, 0], [0.1, 1, 0.1], [0, 0.1, 1]],  
    [[[1, 0.3, 0], [0.3, 1, 0.3], [0, 0.3, 1]],  
     [[1, 0.2, 0], [0.2, 1, 0.2], [0, 0.2, 1]]]  
])

X, labels = generate_data_from_gmm(N, gmm_pdf)

L = np.arange(C)
N_per_l = np.array([sum(labels == l) for l in L])
print("Samples per class:", N_per_l)

fig = plt.figure(figsize=(10, 10))
ax_gmm = fig.add_subplot(111, projection='3d')

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

def perform_erm_classification(X, Lambda, gmm_pdf, C):
    priors = gmm_pdf['priors']
    mu = gmm_pdf['mu']
    Sigma = gmm_pdf['Sigma']

    decisions = np.zeros(X.shape[0], dtype=int)

    for i, x in enumerate(X):
        risks = np.zeros(C)
        for j in range(C):
            if j == 2:  
                p_x_given_class = 0.5 * multivariate_normal.pdf(x, mean=mu[2][0], cov=Sigma[2][0]) + \
                                  0.5 * multivariate_normal.pdf(x, mean=mu[2][1], cov=Sigma[2][1])
            else:
                p_x_given_class = multivariate_normal.pdf(x, mean=mu[j], cov=Sigma[j])
            for k in range(C):
                risks[k] += Lambda[k, j] * priors[j] * p_x_given_class
        decisions[i] = np.argmin(risks)

    return decisions

Lambda = np.ones((C, C)) - np.eye(C) 

decisions = perform_erm_classification(X, Lambda, gmm_pdf, C)

print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
conf_display = ConfusionMatrixDisplay.from_predictions(decisions, labels, display_labels=['1', '2', '3'],
                                                       colorbar=False)
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Number of Misclassified Samples:", N - correct_class_samples)

prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

fig_map = plt.figure(figsize=(10, 10))
ax_map = fig_map.add_subplot(111, projection='3d')

marker_shapes = ['.', 'o', '^']
for r in L:  
    for c in L:  
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
