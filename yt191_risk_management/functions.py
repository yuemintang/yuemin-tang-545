import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy import stats
from scipy.optimize import minimize
from numpy.linalg import eig
from statsmodels.tsa.arima.model import ARIMA


# Calculating exponentially weighted covariance matrix
def calc_ewmcov(mat, l=0.97):
    weights = np.zeros(len(mat))
    means = np.mean(mat, axis=0)
    for i in range(len(mat)):
        weights[len(mat) - 1 - i] = ((1 - l) * pow(l, i))
    w_mat = np.diag(weights / sum(weights))
    cov = (mat - means).T @ w_mat @ (mat - means)
    return cov


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


# Implement a function similar to the “return_calculate()” in this week’s code.
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    vars_a = prices.columns
    nVars = len(vars_a)
    vars = [var for var in vars_a if var != dateColumn]
    if nVars == len(vars):
        raise ValueError("dateColumn: ", dateColumn, " not in DataFrame: ", vars)
    nVars = nVars - 1

    p = prices[vars].to_numpy()
    n = len(p)
    m = len(p[0])
    p2 = np.empty((n - 1, m))

    for i in range(n - 1):
        for j in range(m):
            p2[i, j] = p[i + 1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError("method: ", method, " must be in (\"LOG\",\"DISCRETE\")")

    dates = prices[dateColumn].iloc[1:n].to_numpy()
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars[i]] = p2[:, i]
    return out


# Calculate ES for distribution
def calculate_es(data, var):
    es = -np.mean(data[data <= -var])
    return es


# Calculate VaR for distribution
def calculate_var(data, alpha=0.05):
    var = -np.quantile(data, alpha)
    return var


# Calculate VaR using a normal distribution.
def normal_distribution(out, alpha=0.05, n=10000):
    dis = np.random.normal(np.mean(out), np.std(out), n)
    var = -np.quantile(dis, alpha)
    return dis, var


# Calculate VaR using a normal distribution with an Exponentially Weighted variance (λ = 0. 94).
def normal_distribution_ew(out, n=10000, alpha=0.05, l=0.94):
    cov = calc_ewmcov(out, l)
    dis = np.random.normal(np.mean(out), np.sqrt(cov).iloc[0, 0], n)
    var = -np.quantile(dis, alpha)
    return dis, var


# Calculate VaR using a MLE fitted T distribution.
def neg_LL(parameters, out):
    df, loc, scale = parameters
    neg_LL = -np.sum(stats.t.logpdf(out, df=df, loc=loc, scale=scale))
    return neg_LL


def mle_t_distribution(out, alpha=0.05, n=10000, fit=False):
    constraint = ({"type": "ineq", "fun": lambda x: x[0]})
    mle_t_model = minimize(neg_LL, np.array([10, np.mean(out), np.std(out)], dtype=object), args=out, constraints=constraint)
    df, loc, scale = mle_t_model.x[:3]
    if fit == True:
        return df, loc, scale
    else:
        dis = stats.t.rvs(df, loc=loc, scale=scale, size=n)
        var = -np.quantile(dis, alpha)
        return dis, var


# Calculate VaR using a fitted AR(1) model.
def ar1_simulation(out, alpha=0.05, n=10000):
    model = ARIMA(out, order=(1, 0, 0)).fit()
    sigma = np.std(model.resid)
    dis = np.empty(n)
    for i in range(n):
        dis[i] = model.params[0] * (out.values[-1]) + sigma * np.random.normal()
    var = -np.quantile(dis, alpha)
    return dis, var


# Calculate VaR using a Historic Simulation.
def historic_simulation(out, alpha=0.05):
    dis = out
    var = -np.quantile(out, alpha)
    return dis, var


# Calculate prices for the selected portfolio
def calc_price(portfolio, prices, code="ALL", delta=True):
    if code == "ALL":
        assets = portfolio
    else:
        assets = portfolio[portfolio["Portfolio"] == code]
    new_prices = pd.concat([prices["Date"], prices[assets["Stock"]]], axis=1)
    out = return_calculate(new_prices).iloc[:, 1:]
    if delta == True:
        pv = np.dot(prices[assets["Stock"]].iloc[-1], assets["Holding"])
        delta = assets["Holding"].values * prices[assets["Stock"]].iloc[-1].values / pv
        return pv, delta, assets, new_prices, out
    else:
        p_last = prices[assets["Stock"]].iloc[-1]
        return assets, p_last, out


# Using delta_normal model for returns and calculate VaR with lambda = 0.94.
def delta_normal(portfolio, prices, alpha=0.05, l=0.94, code="ALL"):
    pv, delta, assets, new_prices, out = calc_price(portfolio, prices, code)
    cov = calc_ewmcov(out, l)
    var = -pv * norm.ppf(alpha) * np.sqrt(delta.T @ cov @ delta)
    return var


# Using monte_carlo_normal model for returns and calculate VaR with lambda = 0.94.
def monte_carlo_normal(portfolio, prices, n=10000, alpha=0.05, l=0.94, code="ALL"):
    assets, p_last, out = calc_price(portfolio, prices, code, False)
    cov = calc_ewmcov(out, l)
    sim_prices = np.dot((simulation(cov, n, 1) + out.mean().values) * p_last.values, assets["Holding"])
    var = -np.quantile(sim_prices, alpha)
    return sim_prices, var


# Using historical_simulation model for returns and calculate VaR.
def historical_simulation(portfolio, prices, n=10000, alpha=0.05, code="ALL"):
    assets, p_last, out = calc_price(portfolio, prices, code, False)
    sim_prices = np.dot(out.sample(n, replace=True) * p_last.values, assets["Holding"])
    var = -np.quantile(sim_prices, alpha)
    return sim_prices, var


# Using generalized_t model for returns and calculate VaR.
def generalized_t(portfolio, prices, n=10000, code="ALL"):
    # Calculate the present value and returns for the portfolio
    pv, delta, assets, new_prices, out = calc_price(portfolio, prices, code)
    means = np.mean(out, axis=0)
    assets_ = out - means

    # Use generalized t model to fit for each asset
    assets_2 = assets_
    para_list = []
    for asset in assets_.columns:
        df, loc, scale = mle_t_distribution(assets_[asset], fit=True)
        para_list.append(mle_t_distribution(assets_[asset], fit=True))
        # Transform the observation vector Xi in to a uniform vector Ui using the CDF for xi
        assets_2[asset] = t.cdf(assets_[asset], df=df, loc=loc, scale=scale)

    # Transform the uniform vector Ui into a Standard Normal vector Zi
    assets_2 = pd.DataFrame(norm.ppf(assets_), index=assets_.index, columns=assets_.columns)
    # calculate the correlation matrix of Z
    mat = assets_2.corr(method='spearman')
    # Simulation NSim draws from the multivariate normal
    sim = pd.DataFrame(simulation(mat, n, 1), columns=out.columns)
    # Transform Z into a uniform variable using the standard normal CDF
    assets_3 = pd.DataFrame(norm.cdf(sim), index=sim.index, columns=sim.columns)

    # Transform Ui into the fitted distribution using the quantile of the fitted distribution
    assets_4 = pd.DataFrame()
    for i in range(len(sim.columns)):
        asset = sim.columns[i]
        assets_4[asset] = t.ppf(assets_3[asset], df=para_list[i][0], loc=para_list[i][1], scale=para_list[i][2])

    # Calculate the pv and VaR
    pv_t = np.dot(assets_4 * new_prices.iloc[-1].values[1:], assets["Holding"])
    var_t = calculate_var(pv_t)
    return pv_t, var_t
