import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import seaborn as sns


# Problem 1

# Calculate and compare the expected value and standard deviation of Pt for each of the 3 types of price returns
def classical_brownian_motion(mu, sigma, n, p0):
    rt = np.random.normal(mu, sigma, n)
    p = np.zeros(n)
    for i in range(n):
        p[i] = p0 + rt[i]
    mean_c = p0
    std_c = sigma
    print("For classical_brownian_motion")
    print("calculated mean:", mean_c, "\ncalculated std:", std_c)
    print("actual mean:", np.mean(p), "\nactual std:", np.std(p))
    return p


def arithmetic_return_system(mu, sigma, n, p0):
    rt = np.random.normal(mu, sigma, n)
    p = np.zeros(n)
    for i in range(n):
        p[i] = p0 * (1 + rt[i])
    mean_c = p0
    std_c = p0 * sigma
    print("For arithmetic_return_system")
    print("calculated mean:", mean_c, "\ncalculated std:", std_c)
    print("actual mean:", np.mean(p), "\nactual std:", np.std(p))
    return p


def geometric_brownian_motion(mu, sigma, n, p0):
    rt = np.random.normal(mu, sigma, n)
    p = np.zeros(n)
    for i in range(n):
        p[i] = p0 * np.exp(rt[i])
    mean_c = p0 * np.exp(1/2 * sigma * sigma)
    std_c = p0 * np.sqrt(np.exp(sigma * sigma) * (np.exp(sigma * sigma) - 1))
    print("For geometric_brownian_motion")
    print("calculated mean:", mean_c, "\ncalculated std:", std_c)
    print("actual mean:", np.mean(p), "\nactual std:", np.std(p))
    return p


# Simulate each return equation using  𝑟 ∼ 𝑁 (0, σ2)
np.random.seed(0)
p0 = 100
n = 10000
sigma = 0.3
mu = 0
p1 = classical_brownian_motion(mu, sigma, n, p0)
p2 = arithmetic_return_system(mu, sigma, n, p0)
p3 = geometric_brownian_motion(mu, sigma, n, p0)

fig, axs = plt.subplots(1, 3, figsize=(20, 10))
axs[0].hist(p1, bins=50, density=True)
sns.kdeplot(p1, ax=axs[0])
axs[0].set_title("classical_brownian_motion")
axs[1].hist(p2, bins=50, density=True)
sns.kdeplot(p2, ax=axs[1])
axs[1].set_title("arithmetic_return_system")
axs[2].hist(p3, bins=50, density=True)
sns.kdeplot(p3, ax=axs[2])
axs[2].set_title("geometric_brownian_motion")
plt.savefig("Problem1.jpg")
plt.show()


# Problem 2

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


# Use DailyPrices.csv. Calculate the arithmetic returns for all prices.
prices = pd.read_csv("DailyPrices.csv")
out = return_calculate(prices)
# Remove the mean from the series so that the mean(META)=0.
meta = out['META']-out['META'].mean()


# Calculate VaR using a normal distribution.
def normal_distribution(out, alpha=0.05, n=10000):
    dis = np.random.normal(np.mean(out), np.std(out), n)
    var = -np.quantile(dis, alpha)
    print("VaR for normal_distribution:", var)
    return dis, var


# Calculate VaR using a normal distribution with an Exponentially Weighted variance (λ = 0. 94).
def ew_cov(out, l=0.94):
    weights = np.zeros(len(out))
    means = np.mean(out, axis=0)
    for i in range(len(out)):
        weights[len(out) - 1 - i] = ((1 - l) * pow(l, i))
    w_mat = np.diag(weights / sum(weights))
    cov = (out - means).T @ w_mat @ (out - means)
    return cov


def normal_distribution_ew(out, n=10000, alpha=0.05, l=0.94):
    cov = ew_cov(out, l)
    dis = np.random.normal(np.mean(out), np.sqrt(cov), n)
    var = -np.quantile(dis, alpha)
    print("VaR for normal_distribution_ew:", var)
    return dis, var


# Calculate VaR using a MLE fitted T distribution.
def neg_LL(parameters, out):
    df, loc, scale = parameters
    neg_LL = -np.sum(stats.t.logpdf(out, df=df, loc=loc, scale=scale))
    return neg_LL


def mle_t_distribution(out, alpha=0.05, n=10000):
    constraint = ({"type": "ineq", "fun": lambda x: x[0]})
    mle_t_model = minimize(neg_LL, np.array([10, np.mean(out), np.std(out)]), args=out, constraints=constraint)
    df, loc, scale = mle_t_model.x[:3]
    dis = stats.t.rvs(df, loc=loc, scale=scale, size=n)
    var = -np.quantile(dis, alpha)
    print("VaR for mle_t_distribution:", var)
    return dis, var


# Calculate VaR using a fitted AR(1) model.
def ar1_simulation(out, alpha=0.05, n=10000):
    model = ARIMA(out, order=(1, 0, 0)).fit()
    sigma = np.std(model.resid)
    dis = np.empty(n)
    for i in range(n):
        dis[i] = model.params[0] * (out.values[-1]) + sigma * np.random.normal()
    var = -np.quantile(dis, alpha)
    print("VaR for ar1_simulation:", var)
    return dis, var


# Calculate VaR using a Historic Simulation.
def historic_simulation(out, alpha=0.05):
    dis = out
    var = -np.quantile(out, alpha)
    print("VaR for historic_simulation:", var)
    return dis, var


# Compare and plot the 5 values.
fig2, axs = plt.subplots(2, 3, figsize=(40, 20))
a, var_a = normal_distribution(meta)
axs[0, 0].hist(a, bins=50, density=True)
sns.kdeplot(a, ax=axs[0, 0])
axs[0, 0].axvline(-var_a, color="r")
axs[0, 0].set_title("normal_distribution")

b, var_b = normal_distribution_ew(meta)
axs[0, 1].hist(b, bins=50, density=True)
sns.kdeplot(b, ax=axs[0, 1])
axs[0, 1].axvline(-var_b, color="r")
axs[0, 1].set_title("normal_distribution_ew")

c, var_c = mle_t_distribution(meta)
axs[0, 2].hist(c, bins=50, density=True)
sns.kdeplot(c, ax=axs[0, 2])
axs[0, 2].axvline(-var_c, color="r")
axs[0, 2].set_title("mle_t_distribution")

d, var_d = ar1_simulation(meta)
axs[1, 0].hist(d, bins=50, density=True)
sns.kdeplot(d, ax=axs[1, 0])
axs[1, 0].axvline(-var_d, color="r")
axs[1, 0].set_title("ar1_simulation")

e, var_e = historic_simulation(meta)
axs[1, 1].hist(e, bins=50, density=True)
sns.kdeplot(e, ax=axs[1, 1])
axs[1, 1].axvline(-var_e, color="r")
axs[1, 1].set_title("historic_simulation")

plt.savefig("Problem2.jpg")
plt.show()


# Problem 3

# Using delta_normal model for returns and calculate VaR with lambda = 0.94.
def delta_normal(portfolio, prices, alpha=0.05, l=0.94, code="ALL"):
    if code == "ALL":
        assets = portfolio
    else:
        assets = portfolio[portfolio["Portfolio"] == code]
    pv = np.dot(prices[assets["Stock"]].iloc[-1], assets["Holding"])
    new_prices = pd.concat([prices["Date"], prices[assets["Stock"]]], axis=1)
    delta = assets["Holding"].values * prices[assets["Stock"]].iloc[-1].values / pv
    out = return_calculate(new_prices).iloc[:, 1:]
    cov = ew_cov(out, l)
    var = -pv * norm.ppf(alpha) * np.sqrt(delta.T @ cov @ delta)
    return var


# Simulation for Monte Carlo and Historic
def simulation(mat, n, explained=1):
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


# Using monte_carlo_normal model for returns and calculate VaR with lambda = 0.94.
def monte_carlo_normal(portfolio, prices, n = 10000, alpha=0.05, l=0.94, code="ALL"):
    if code == "ALL":
        assets = portfolio
    else:
        assets = portfolio[portfolio["Portfolio"] == code]
    p_last = prices[assets["Stock"]].iloc[-1]
    new_prices = pd.concat([prices["Date"], prices[assets["Stock"]]], axis=1)
    out = return_calculate(new_prices).iloc[:, 1:]
    cov = ew_cov(out, l)
    sim_prices = np.dot((simulation(cov, n) + out.mean().values) * p_last.values, assets["Holding"])
    var = -np.quantile(sim_prices, alpha)
    print("VaR for Monte Carlo Simulation of Portfolio "+code+":", var)
    return sim_prices, var


# Using historical_simulation model for returns and calculate VaR.
def historical_simulation(portfolio, prices, n=1000, alpha=0.05, code="ALL"):
    if code == "ALL":
        assets = portfolio
    else:
        assets = portfolio[portfolio["Portfolio"] == code]
    p_last = prices[assets["Stock"]].iloc[-1]
    new_prices = pd.concat([prices["Date"], prices[assets["Stock"]]], axis=1)
    out = return_calculate(new_prices).iloc[:, 1:]
    sim_prices = np.dot(out.sample(n, replace=True) * p_last.values, assets["Holding"])
    var = -np.quantile(sim_prices, alpha)
    print("VaR for Historical Simulation of Portfolio "+code+":", var)
    return sim_prices, var


# Using Portfolio.csv and DailyPrices.csv.
portfolio = pd.read_csv("portfolio.csv")
prices = pd.read_csv("DailyPrices.csv")
np.random.seed(0)


# calculate the VaR of each portfolio as well as your total VaR
a_d = delta_normal(portfolio, prices, 0.05, 0.94, "A")
b_d = delta_normal(portfolio, prices, 0.05, 0.94, "B")
c_d = delta_normal(portfolio, prices, 0.05, 0.94, "C")
all_d = delta_normal(portfolio, prices, 0.05, 0.94, "ALL")
print("VaR for Delta Normal of Portfolio A:", a_d)
print("VaR for Delta Normal of Portfolio B:", b_d)
print("VaR for Delta Normal of Portfolio C:", c_d)
print("VaR for Delta Normal of Portfolio ALL:", all_d)


# Choose monte_carlo_normal for returns and calculate VaR again.
a_mc, var_a_mc = monte_carlo_normal(portfolio, prices, 10000, 0.05, 0.94, "A")
b_mc, var_b_mc = monte_carlo_normal(portfolio, prices, 10000, 0.05, 0.94, "B")
c_mc, var_c_mc = monte_carlo_normal(portfolio, prices, 10000, 0.05, 0.94, "C")
all_mc, var_all_mc = monte_carlo_normal(portfolio, prices, 10000, 0.05, 0.94, "ALL")

# Plot the simulation and VaR for monte_carlo_normal
fig3, axs = plt.subplots(2, 2, figsize=(30, 30))

axs[0, 0].hist(a_mc, bins=50, density=True)
sns.kdeplot(a_mc, ax=axs[0, 0])
axs[0, 0].axvline(-var_a_mc, color="r")
axs[0, 0].set_title("Monte Carlo Simulation of Portfolio A")

axs[0, 1].hist(b_mc, bins=50, density=True)
sns.kdeplot(b_mc, ax=axs[0, 1])
axs[0, 1].axvline(-var_b_mc, color="r")
axs[0, 1].set_title("Monte Carlo Simulation of Portfolio B")

axs[1, 0].hist(c_mc, bins=50, density=True)
sns.kdeplot(c_mc, ax=axs[1, 0])
axs[1, 0].axvline(-var_c_mc, color="r")
axs[1, 0].set_title("Monte Carlo Simulation of Portfolio C")

axs[1, 1].hist(all_mc, bins=50, density=True)
sns.kdeplot(all_mc, ax=axs[1, 1])
axs[1, 1].axvline(-var_all_mc, color="r")
axs[1, 1].set_title("Monte Carlo Simulation of Portfolio All")

plt.savefig("Problem3-MC.jpg")
plt.show()


# Choose historical_simulation for returns and calculate VaR again.
a_h, var_a_h = historical_simulation(portfolio, prices, 10000, 0.05, "A")
b_h, var_b_h = historical_simulation(portfolio, prices, 10000, 0.05, "B")
c_h, var_c_h = historical_simulation(portfolio, prices, 10000, 0.05, "C")
all_h, var_all_h = historical_simulation(portfolio, prices, 10000, 0.05, "ALL")

# Plot the simulation and VaR for historical_simulation
fig4, axs = plt.subplots(2, 2, figsize=(30, 30))

axs[0, 0].hist(a_h, bins=50, density=True)
sns.kdeplot(a_h, ax=axs[0, 0])
axs[0, 0].axvline(-var_a_h, color="r")
axs[0, 0].set_title("Historical Simulation of Portfolio A")

axs[0, 1].hist(b_h, bins=50, density=True)
sns.kdeplot(b_h, ax=axs[0, 1])
axs[0, 1].axvline(-var_b_h, color="r")
axs[0, 1].set_title("Historical Simulation of Portfolio B")

axs[1, 0].hist(c_h, bins=50, density=True)
sns.kdeplot(c_h, ax=axs[1, 0])
axs[1, 0].axvline(-var_c_h, color="r")
axs[1, 0].set_title("Historical Simulation of Portfolio C")

axs[1, 1].hist(all_h, bins=50, density=True)
sns.kdeplot(all_h, ax=axs[1, 1])
axs[1, 1].axvline(-var_all_h, color="r")
axs[1, 1].set_title("Historical Simulation of Portfolio All")

plt.savefig("Problem3-H.jpg")
plt.show()


# Check the normal distribution of returns
skewness = out.iloc[:, 1:].skew()
kurtosis = out.iloc[:, 1:].kurtosis()
fig5, axs = plt.subplots(1, 2, figsize=(40, 20))
axs[0].hist(skewness, bins=50, density=True)
sns.kdeplot(skewness, ax=axs[0])
axs[0].set_title("Skewness for returns")
axs[1].hist(kurtosis, bins=50, density=True)
sns.kdeplot(kurtosis, ax=axs[1])
axs[1].set_title("Kurtosis for returns")
plt.savefig("Problem3-sk.jpg")
plt.show()
