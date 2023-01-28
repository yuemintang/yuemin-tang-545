import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm, kurtosis, skew, ttest_1samp
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import ArmaProcess


# Problem 1

# H0: the function is unbiased
# Significance level: 0.05
def test_biased(size, type):
    a = np.empty(1000)
    for i in range(1000):
        if type == "kurtosis":
            a[i] = kurtosis(norm(0, 1).rvs(size))
        elif type == "skewness":
            a[i] = skew(norm(0, 1).rvs(size))
    t, p_value = ttest_1samp(a, 0)
    print(type, "p_value is:", p_value)
    if p_value < 0.05:
        print("For", type, "with sample size", size, ": reject H0 that the function is unbiased.")
    else:
        print("For", type, "with sample size", size, ": fail to reject H0 that the function is unbiased.")


test_biased(100, "kurtosis")
test_biased(1000, "kurtosis")
test_biased(10000, "kurtosis")

test_biased(100, "skewness")
test_biased(1000, "skewness")
test_biased(10000, "skewness")


# Problem 2

# Read data and create x and y
data = pd.read_csv("problem2.csv")
x = data["x"]
y = data["y"]

# Fit the data using OLS
ones = [1] * len(x)
X = np.vstack((x, ones)).T
model = sm.OLS(y, X)
result = model.fit()
print(result.summary())

# Error vector and see if normally distributed
err_vec = result.resid
print("For the error vector:", "skewness =", skew(err_vec), "kurtosis =", kurtosis(err_vec))
plt.hist(result.resid, density=True)
x_axis = np.arange(-6, 6, 0.01)
plt.plot(x_axis, norm.pdf(x_axis, 0, 1))
# plt.savefig("Problem2.jpg")
plt.show()


# Fit the data using MLE given the assumption of normality
def MLE_Norm(parameters):
    # extract parameters
    const, beta, std_dev = parameters
    # predict the output
    pred = const + beta * x
    # Calculate the log-likelihood for normal distribution
    LL = np.sum(stats.norm.logpdf(y, pred, std_dev))
    # Return the negative log-likelihood
    return -1 * LL


mle_norm_model = minimize(MLE_Norm, np.array([1, 1, 1]))
print("Betas for MLE_Norm: ", mle_norm_model.x[0], mle_norm_model.x[1])
print("LL for MLE_Norm: ", -mle_norm_model.fun)


# Fit the MLE using the assumption of a T distribution of the errors
def MLE_T(parameters):
    const, beta, std_dev, scale = parameters
    pred = const + beta * x
    LL = np.sum(stats.t.logpdf(y-pred, std_dev, scale=scale))
    return -1 * LL


mle_t_model = minimize(MLE_T, np.array([1,1,1,1]))
print("Betas for MLE_T: ", mle_t_model.x[0], mle_t_model.x[1])
print("LL for MLE_T: ", -mle_t_model.fun)


def calc_r2(x, y, const, beta):
    y_pred = const + beta * x
    err = y - y_pred
    sst = sum((y - np.mean(y)) ** 2)
    sse = sum((err - np.mean(err)) ** 2)
    r2 = 1 - sse / sst
    return r2


# Compare which one is the best fit by calculating their R^2 and information criteria

# R^2
r2_norm = calc_r2(x, y, mle_norm_model.x[0], mle_norm_model.x[1])
print("R^2 for MLE_Norm: ", r2_norm)

r2_t = calc_r2(x, y, mle_t_model.x[0], mle_t_model.x[1])
print("R^2 for MLE_T: ", r2_t)

# Information criteria
aic_n = 2 * 2 + 2 * mle_norm_model.fun
bic_n = 2 * np.log(len(x)) + 2 * mle_norm_model.fun
print("AIC for MLE_Norm: ", aic_n)
print("BIC for MLE_Norm: ", bic_n)

aic_t = 2 * 2 + 2 * mle_t_model.fun
bic_t = 2 * np.log(len(x)) + 2 * mle_t_model.fun
print("AIC for MLE_T: ", aic_t)
print("BIC for MLE_T: ", bic_t)


# Problem 3

ma = np.array([1])
ar = np.array([1])
fig, axs = plt.subplots(6, 3, figsize=(40, 60))

# Simulate AR(1)
ar1 = np.array([1, 0.7])
ar1 = ArmaProcess(ar1, ma).generate_sample(nsample=1000)
axs[0, 0].plot(ar1)
axs[0, 0].set_title("AR1")
sm.graphics.tsa.plot_acf(ar1, lags=15, title="AR1 ACF", ax=axs[0, 1])
sm.graphics.tsa.plot_pacf(ar1, lags=15, title="AR1 PACF", ax=axs[0, 2], method='ywm')

# Simulate AR(2)
ar2 = np.array([1, 0.7, 0.5])
ar2 = ArmaProcess(ar2, ma).generate_sample(nsample=1000)
axs[1, 0].plot(ar2)
axs[1, 0].set_title("AR2")
sm.graphics.tsa.plot_acf(ar2, lags=15, title="AR2 ACF", ax=axs[1, 1])
sm.graphics.tsa.plot_pacf(ar2, lags=15, title="AR2 PACF", ax=axs[1, 2], method='ywm')

# Simulate AR(3)
ar3 = np.array([1, 0.7, 0.5, 0.3])
ar3 = ArmaProcess(ar3, ma).generate_sample(nsample=1000)
axs[2, 0].plot(ar3)
axs[2, 0].set_title("AR3")
sm.graphics.tsa.plot_acf(ar3, lags=15, title="AR3 ACF", ax=axs[2, 1])
sm.graphics.tsa.plot_pacf(ar3, lags=15, title="AR3 PACF", ax=axs[2, 2], method='ywm')

# Simulate MA(1)
ma1 = np.array([1, 0.7])
ma1 = ArmaProcess(ar, ma1).generate_sample(nsample=1000)
axs[3, 0].plot(ma1)
axs[3, 0].set_title("MA1")
sm.graphics.tsa.plot_acf(ma1, lags=15, title="MA1 ACF", ax=axs[3, 1])
sm.graphics.tsa.plot_pacf(ma1, lags=15, title="MA1 PACF", ax=axs[3, 2], method='ywm')

# Simulate MA(2)
ma2 = np.array([1, 0.7, 0.5])
ma2 = ArmaProcess(ar, ma2).generate_sample(nsample=1000)
axs[4, 0].plot(ma2)
axs[4, 0].set_title("MA2")
sm.graphics.tsa.plot_acf(ma2, lags=15, title="MA2 ACF", ax=axs[4, 1])
sm.graphics.tsa.plot_pacf(ma2, lags=15, title="MA2 PACF", ax=axs[4, 2], method='ywm')

# Simulate MA(3)
ma3 = np.array([1, 0.7, 0.5, 0.3])
ma3 = ArmaProcess(ar, ma3).generate_sample(nsample=1000)
axs[5, 0].plot(ma3)
axs[5, 0].set_title("MA3")
sm.graphics.tsa.plot_acf(ma3, lags=15, title="MA3 ACF", ax=axs[5, 1])
sm.graphics.tsa.plot_pacf(ma3, lags=15, title="MA3 PACF", ax=axs[5, 2], method='ywm')

# plt.savefig("Problem3.jpg")
plt.show()