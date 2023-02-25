import matplotlib.pyplot as plt
import seaborn as sns
from yt191_risk_management.functions import *


# Problem 1

# Use the data in problem1.csv.
data = pd.read_csv("problem1.csv")

# Fit a Normal Distribution to this data and calculate the VaR and ES
dis_n, var_n = normal_distribution(data)
print("VaR for normal_distribution:", var_n)
es_n = calculate_es(dis_n, var_n)
print("ES for normal_distribution:", es_n)

# Fit a Generalized T distribution to this data and calculate the VaR and ES
dis_t, var_t = mle_t_distribution(data)
print("VaR for mle_t_distribution:", var_t)
es_t = calculate_es(dis_t, var_t)
print("ES for mle_t_distribution:", es_t)

plt.hist(data, bins=50, density=True)

sns.kdeplot(dis_n, color="r", label="Normal")
plt.axvline(-var_n, color="r")
plt.axvline(-es_n, color="r", linestyle="--")

sns.kdeplot(dis_t, color="g", label="T")
plt.axvline(-var_t, color="g")
plt.axvline(-es_t, color="g", linestyle="--")

plt.title("Normal Distribution and Generalized T Distribution for the Data")
plt.legend()
plt.savefig("Problem1.jpg")
plt.show()


# Problem 2

# 1. Covariance estimation techniques.
prices = pd.read_csv("DailyPrices.csv")
returns = return_calculate(prices).iloc[:, 1:]
print("returns for DailyPrices.csv:\n", returns)

mat_1 = p_corr_p_var(returns)
print("covariance with  Standard Pearson correlation and Standard Pearson variance:\n", mat_1)

mat_2 = p_corr_ew_var(returns)
print("covariance using Standard Pearson correlation and Exponentially weighted variance:\n", mat_2)

mat_3 = ew_corr_p_var(returns)
print("covariance using Exponentially weighted correlation and Standard Pearson variance:\n", mat_3)

mat_4 = ew_corr_ew_var(returns)
print("covariance using Exponentially weighted correlation and Exponentially weighted variance:\n", mat_4)


# 2. Non PSD fixes for correlation matrices
sigma = sigma_generate(500)
print("Original", check_psd(sigma))
print("Near_psd", check_psd(near_psd(sigma)))
print("Higham_psd", check_psd(Higham_psd(sigma, None, 1e-8)))


# 3. Simulation Methods
x_d = simulation(mat_4, 25000, None)
print("Simulation directly from a covariance matrix:\n", x_d)

x_pca = simulation(mat_4, 25000, 0.75)
print("Simulation using PCA with an 0.75 variance explained:\n", x_pca)


# 4. VaR calculation methods (all discussed)
data = pd.read_csv("problem1.csv")
prices = pd.read_csv("DailyPrices.csv")
portfolio = pd.read_csv("portfolio.csv")

a, var_a = normal_distribution(data)
print("normal_distribution VaR:", var_a)

b, var_b = normal_distribution_ew(data)
print("normal_distribution_ew VaR:", var_b)

c, var_c = mle_t_distribution(data)
print("mle_t_distribution VaR:", var_c)

d, var_d = ar1_simulation(data)
print("ar1_simulation VaR:", var_d)

e, var_e = historic_simulation(data)
print("historic_simulation VaR:", var_e)

var_a_d = delta_normal(portfolio, prices, 0.05, 0.94, "A")
print("delta_normal VaR for portfolio A:", var_a_d)

a_mc, var_a_mc = monte_carlo_normal(portfolio, prices, 10000, 0.05, 0.94, "A")
print("monte_carlo_normal VaR for portfolio A:", var_a_mc)

a_h, var_a_h = historical_simulation(portfolio, prices, 10000, 0.05, "A")
print("historical_simulation VaR for portfolio A:", var_a_h)

a_t, var_a_t = generalized_t(portfolio, prices, 10000, "A")
print("generalized_t VaR for portfolio A:", var_a_t)


# 5. ES calculation (see Problem 1)


# Problem 3

# Using Portfolio.csv and DailyPrices.csv
prices = pd.read_csv("DailyPrices.csv")
portfolio = pd.read_csv("portfolio.csv")
fig, axs = plt.subplots(3, 4, figsize=(40, 30))
code_list = ["A", "B", "C", "ALL"]
np.random.seed(0)

for c in range(len(code_list)):
    # Fit a Generalized T model to each stock
    pv_t, var_t = generalized_t(portfolio, prices, 10000, code_list[c])
    es_t = calculate_es(pv_t, var_t)
    # calculate the VaR and ES of each portfolio as well as total VaR and ES
    print("VaR of Generalized T Model for Portfolio "+code_list[c]+":", var_t)
    print("ES of Generalized T Model for Portfolio "+code_list[c]+":", es_t)

    axs[0, c].hist(pv_t, bins=50, density=True)
    sns.kdeplot(pv_t, ax=axs[0, c])
    axs[0, c].axvline(-var_t, color="r", label="VaR")
    axs[0, c].axvline(-es_t, color="g", label="ES")
    axs[0, c].set_title("Generalized T Model for Portfolio "+code_list[c])

    # Compare the results from this to VaR using Delta Normal
    var_d = delta_normal(portfolio, prices, 0.05, 0.94,  code_list[c])
    print("VaR of Delta Normal for Portfolio " + code_list[c] + ":", var_d)

    # Compare the results from this to VaR using Monte Carlo Simulation
    pv_mc, var_mc = monte_carlo_normal(portfolio, prices, 10000, 0.05, 0.94, code_list[c])
    print("VaR of Monte Carlo Simulation for Portfolio " + code_list[c] + ":", var_mc)

    axs[1, c].hist(pv_mc, bins=50, density=True)
    sns.kdeplot(pv_mc, ax=axs[1, c])
    axs[1, c].axvline(-var_mc, color="r", label="VaR")
    axs[1, c].set_title("Monte Carlo Simulation for Portfolio " + code_list[c])

    # Compare the results from this to VaR using Historical Simulation
    pv_h, var_h = historical_simulation(portfolio, prices, 10000, 0.05, code_list[c])
    print("VaR of Historical Simulation for Portfolio " + code_list[c] + ":", var_h)

    axs[2, c].hist(pv_h, bins=50, density=True)
    sns.kdeplot(pv_h, ax=axs[2, c])
    axs[2, c].axvline(-var_h, color="r", label="VaR")
    axs[2, c].set_title("Historical Simulation for Portfolio " + code_list[c])

axs[0, 3].legend(loc="upper right")
plt.savefig("Problem3.jpg")
plt.show()
