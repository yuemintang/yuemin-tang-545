import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig
import time


# Problem 1

# Read data from DailyReturn.csv
data = pd.read_csv("DailyReturn.csv").iloc[:, 1:]
n_time = len(data)
n_stock = len(data.columns)


# Calculating exponentially weighted covariance matrix
def calc_ewmcov(mat, l=0.97):
    weights = np.zeros(n_time)
    means = np.mean(mat, axis=0)
    for i in range(n_time):
        weights[n_time - 1 - i] = ((1 - l) * pow(l, i))
    w_mat = np.diag(weights / sum(weights))
    cov = (mat - means).T @ w_mat @ (mat - means)
    return cov


# Calculate cumulative variance explained by each eigenvalue
def calc_cum_var(mat, n_stock):
    eigen_values, eigen_vectors = np.linalg.eig(mat)
    posv = eigen_values > 1e-8
    eigen_values = eigen_values[posv]
    eigen_vectors = eigen_vectors[:, posv]
    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    cumulative_variance = []
    for i in range(n_stock):
        cumulative_variance = np.cumsum(eigen_values[:i]) / np.sum(eigen_values)
    return cumulative_variance


# Vary λ ∈ (0, 1), use PCA and plot the cumulative variance for each λ chosen
l_list = [0.99, 0.97, 0.9, 0.75, 0.5, 0.2, 0.05]
cum_var_list = []
for l in l_list:
    covs = calc_ewmcov(data, l)
    for i in range(n_stock):
        cumulative_variance = calc_cum_var(covs, i)
        cum_var_list.append(cumulative_variance)
for cum_var in cum_var_list:
    plt.plot(np.real(cum_var))
    plt.xlabel("Principle Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.legend(l_list)
plt.savefig("Problem1.jpg", dpi=300)
plt.show()


# Problem 2

# Copy and implement the chol_psd() function from the course repository
def chol_psd(mat):
    n = len(mat)
    root = np.zeros_like(mat)

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = mat[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if 0.0 == root[j, j]:
            root[j + 1:, j] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (mat[i, j] - s) * ir
    return root


# Copy and implement the near_psd() function from the course repository
def near_psd(mat, epsilon=0.0):
    n = len(mat)

    invSD = None
    out = mat.copy()

    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1 / (np.square(vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out


# Implement Higham’s 2002 nearest psd correlation function
# The Frobenius Norm
def frobenius_norm(mat):
    sum = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            sum = sum + mat[i, j] * mat[i, j]
    return sum


# The first projection
def pu(mat):
    pu = mat.copy()
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if i == j:
                pu[i][j] = 1
    return pu


# The second projection
def ps(mat, weight):
    mat = np.sqrt(weight) @ mat @ np.sqrt(weight)
    vals, vecs = np.linalg.eigh(mat)
    vals[vals < 0] = 0
    ps = np.sqrt(weight) @ vecs @ np.diag(vals) @ vecs.T @ np.sqrt(weight)
    return ps


# Implement Higham'S psd
def Higham_psd(mat, weight, tol):
    if weight is None:
        weight = np.identity(len(mat))
    yk = mat.copy()
    y0 = mat.copy()
    delta_s = np.zeros_like(yk)
    gamma_0 = np.inf
    for k in range(1000):
        rk = yk - delta_s
        xk = ps(rk, weight)
        delta_s = xk - rk
        yk = pu(xk)
        gamma_k = frobenius_norm(yk - y0)
        if abs(gamma_k - gamma_0) < tol and min(np.linalg.eigvals(yk)) > -1e-8:
            break
        else:
            gamma_0 = gamma_k
    return yk


# Generate a non-psd correlation matrix that is 500x500
def sigma_generate(n):
    sigma = np.full((n, n), 0.9)
    for i in range(n):
        sigma[i, i] = 1.0
    sigma[0, 1] = 0.7357
    sigma[1, 0] = 0.7357
    return sigma


# Check if the matrix is PSD
def check_psd(mat):
    eigenvalues = np.linalg.eigh(mat)[0]
    if np.all(eigenvalues >= -1e-8):
        return "Matrix is PSD"
    else:
        return "Matrix is not PSD"


# Use near_psd() and Higham’s method to fix the matrix, confirm the matrix is now PSD
sigma = sigma_generate(500)
print("Original", check_psd(sigma))
print("Near_psd", check_psd(near_psd(sigma)))
print("Higham_psd", check_psd(Higham_psd(sigma, None, 1e-8)))


# Compare the results of both using the Frobenius Norm and run time
n_list = [100, 200, 300, 400, 500]
fnorm_near_list = []
fnorm_Higham_list = []
runtime_near_list = []
runtime_Higham_list = []

for n in n_list:
    sigma = sigma_generate(n)

    begin_near = time.time()
    sigma_near = near_psd(sigma)
    runtime_near_list.append(time.time() - begin_near)
    fnorm_near_list.append(frobenius_norm(sigma_near - sigma))

    begin_Higham = time.time()
    sigma_Higham = Higham_psd(sigma, None, 1e-8)
    runtime_Higham_list.append(time.time() - begin_Higham)
    fnorm_Higham_list.append(frobenius_norm(sigma_Higham - sigma))


# Plot the results of both using the Frobenius Norm and run time
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

axs[0].plot(n_list, fnorm_near_list, label="fnorm_near_psd")
axs[0].plot(n_list, fnorm_Higham_list, label="fnorm_Higham_psd")
axs[0].set_xlabel('N')
axs[0].set_ylabel('Frobenius Norm')
axs[0].legend()

axs[1].plot(n_list, runtime_near_list, label="runtime_near_psd")
axs[1].plot(n_list, runtime_Higham_list, label="runtime_Higham_psd")
axs[1].set_xlabel('N')
axs[1].set_ylabel('Run Time')
axs[1].legend()

plt.savefig("Problem2.jpg", dpi=300)
plt.show()


# Problem 3

# Multivariate normal simulation
def simulation(mat, n, explained=None):
    # Directly from a covariance matrix
    if explained is None:
        L = chol_psd(mat)
        Z = np.random.normal(size=(len(mat), n))
        X = (L @ Z + 0).T
        return X

    # Using PCA with an optional parameter for variance explained
    else:
        eigen_values, eigen_vectors = np.linalg.eigh(mat)
        posv = eigen_values > 1e-8
        eigen_values = eigen_values[posv]
        eigen_vectors = eigen_vectors[:, posv]
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        cum_var = np.cumsum(eigen_values) / np.sum(eigen_values)
        if explained == 1:
            explained = max(np.cumsum(eigen_values) / np.sum(eigen_values))
        idx = np.where(cum_var >= explained)[0][0] + 1
        eigen_vectors = eigen_vectors[:, :idx]
        eigen_values = eigen_values[:idx]
        Z = np.random.normal(size=(idx, n))
        B = eigen_vectors @ np.diag(np.sqrt(eigen_values))
        X = (B @ Z).T
        return X


# Generate covariance using Standard Pearson correlation and Standard Pearson variance
def p_corr_p_var(mat):
    cov = np.cov(mat, rowvar=False)
    return cov


# Generate covariance using Standard Pearson correlation and Exponentially weighted variance
def p_corr_ew_var(mat, l=0.97):
    cov = calc_ewmcov(mat, l)
    corr = np.corrcoef(mat.T)
    sd = np.sqrt(np.diag(cov))
    cov = np.outer(sd, sd) * corr
    return cov


# Generate covariance using Exponentially weighted correlation and Standard Pearson variance
def ew_corr_p_var(mat, l=0.97):
    cov = calc_ewmcov(mat, l)
    corr = np.diag(1 / (np.sqrt(np.diag(cov)))) @ cov @ np.diag(1 / (np.sqrt(np.diag(cov)))).T
    cov = corr * np.outer(np.std(mat), np.std(mat))
    return cov.to_numpy()


# Generate covariance using Exponentially weighted correlation and Exponentially weighted variance
def ew_corr_ew_var(mat):
    return calc_ewmcov(mat).to_numpy()


# Simulate 25,000 draws from each covariance matrix using Direct; PCA 100%; PCA 75%; PCA 50%
n = 25000
np.random.seed(0)
explained_list = [None, 1, 0.75, 0.5]


# Calculate the covariance of the simulated values.
mat_1 = p_corr_p_var(data)
mat_2 = p_corr_ew_var(data)
mat_3 = ew_corr_p_var(data)
mat_4 = ew_corr_ew_var(data)
mat_list = [mat_1, mat_2, mat_3, mat_4]


# Compare the simulated covariance to it’s input matrix using the Frobenius Norm and run time
mat_num = len(mat_list)
fnorm_result_list = np.empty((mat_num, 0)).tolist()
runtime_result_list = np.empty((mat_num, 0)).tolist()
for i in range(mat_num):
    for j in range(len(explained_list)):
        begin_t = time.time()
        x = simulation(mat_list[i], n, explained_list[j])
        used_t = time.time() - begin_t
        runtime_result_list[i].append(used_t)
        fnorm = frobenius_norm(np.cov(x, rowvar=False) - mat_list[i])
        fnorm_result_list[i].append(fnorm)


# Plot the simulated covariance to it’s input matrix using the Frobenius Norm and run time
mat_name = ["p_corr_p_var", "p_corr_ew_var", "ew_corr_p_var", "ew_corr_ew_var"]
method_name = ["Direct", "PCA 100%", "PCA 75%", "PCA 50%"]

fig, axs = plt.subplots(1, 2, figsize=(30, 40))
X_axis = np.arange(len(method_name))
width = 0.2
for i in range(mat_num):
    axs[0].bar(X_axis + width*i, fnorm_result_list[i], width, label=mat_name[i])
axs[0].set_xticks(X_axis)
axs[0].set_xticklabels(method_name)
axs[0].set_xlabel("Method")
axs[0].set_ylabel("F Norm")
axs[0].set_title("F Norm for each method")
axs[0].legend()

for i in range(mat_num):
    axs[1].bar(X_axis + width*i, runtime_result_list[i], width, label=mat_name[i])
axs[1].set_xticks(X_axis)
axs[1].set_xticklabels(method_name)
axs[1].set_xlabel("Method")
axs[1].set_ylabel("Run Time")
axs[1].set_title("Run time for each method")
axs[1].legend()

plt.savefig("Problem3.jpg")
plt.show()
