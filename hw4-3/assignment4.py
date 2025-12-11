# %% [markdown]
# # Assignment 4
# 
# 
# ## Instructions
# 
# Please complete this Jupyter notebook and then convert it to a `.py` file called `assignment4.py`. Upload this file to Gradescope, and await feedback. 
# 
# For manual grading, you will need to upload additional files to separate Gradescope portals.
# 
# You may submit as many times as you want up until the deadline. Only your latest submission counts toward your grade.
# 
# Some tests are hidden and some are visible. The outcome of the visible checks will be displayed to you immediately after you submit to Gradescope. The hidden test outcomes will be revealed after final scores are published. 
# 
# This means that an important part of any strategy is to **start early** and **lock in all the visible test points**. After that, brainstorm what the hidden checks could be and collaborate with your teammates.
# 

# %% [markdown]
# ### Problem 1: Dirichlet-Multinomial
# 
# If $y = (y_1, \ldots, y_k)$ is a length $k$ count vector (e.g. $(1,3,4,0,2)$ and $k=5$), we could let it have a multinomial distribution. 
# $$
# L(y \mid \theta) \propto \prod_{i=1}^k \theta_i^{y_i}
# $$
# 
# We could also put a $\text{Dirichlet}(\alpha_1, \ldots, \alpha_k)$ prior on the unknown $\theta$:
# $$
# \pi(\theta) = \text{Dirichlet}(\alpha_1, \ldots, \alpha_k) \propto \prod_{i=1}^k \theta_i^{\alpha_i - 1}.
# $$
# 
# In lecture we showed that the resulting posterior is 
# $$
# \theta \mid y \sim \text{Dirichlet}(\alpha_1 + y_1, \ldots, \alpha_k + y_k)
# $$
# 
# Now suppose that we have the same prior, but we have $N > 1$ observed vectors. Derive the posterior in this situation. Show all your work!
# 
# Hint: it is customary to let the row index correspond to the observation number; i.e.
# 
# $$
# y = 
# \begin{bmatrix}
# y_1 \\
# y_2 \\
# \vdots \\
# y_N
# \end{bmatrix}
# =
# \begin{bmatrix}
# y_{1,1} & y_{1,2} & \cdots & y_{1,k} \\
# \vdots  & \vdots & \ddots & \vdots \\
# y_{N,1} & y_{N,2} & \cdots & y_{N,k} \\
# \end{bmatrix}
# $$
# 
# Hint 2: we assume each row/observation is independent and identically distributed, so
# 
# $$
# L(y_1, \ldots, y_N \mid \theta) = \prod_{r=1}^N L(y_r \mid \theta) 
# $$

# %% [markdown]
# ### Problem 2: Roulette!
# 
# 
# 
# ![roulette.jpg](attachment:f7e1207c-2e7d-4712-9d1a-2306af682ee3.jpg)
# 
# 
# 
# 

# %% [markdown]
# In the game of (American) Roulette, you guess where the ball will land after the wheel is spun. If all spaces are equally-likely, the house has an advantage no matter what you do. However, if you can spot a "bias" in the wheel, you might have an advantage!
# 
# In this example, the parameter $\theta = (\theta_1, \ldots, \theta_{36}, \theta_{37}, \theta_{38})$ represents the probabilities of each of the 38 possible ball-landing locations. $\theta_{37}$, $\theta_{38}$ will represent the "0" and "00" outcomes, respectively.
# 
# Suppose you have a $\text{Dirichlet}(\alpha_1, \ldots, \alpha_{38})$ prior:
# $$
# \pi(\theta) \propto \prod_{j=1}^{38} \theta_j^{\alpha_j-1}.
# $$
# 
# And you have $N$ vector-valued observations--one vector of counts for each day. We can use a **multinomial** likelihood for this data:
# $$
# L(y \mid \theta) = \prod_{i=1}^{100} L(y_i \mid \theta),
# $$
# 
# where
# $$
# L(y_i \mid \theta) \propto \prod_{j=1}^{38} \theta_j^{y_{ij}}
# $$

# %% [markdown]
# 1.
# 
# Pick a prior by specifying the hyperparameters $\alpha_1, \ldots, \alpha_{38}$. Assign a Numpy array to the variable named `prior_hyperparams`

# %%
import numpy as np
import pandas as pd

# %%
prior_hyperparams = np.ones(38)

# %% [markdown]
# 2. 
# 
# Consider the following (fake) data set `my_data` below. Construct the observation vectors. This data array should have day on the rows, and location on the columns. As a result, your final answer should have shape `(100,38)`
# 
# For example, the day three row vector consists of the 38 count integers:
# 
# $$
# y_3 = \begin{bmatrix}\text{(num 1s on day 3)}, & \cdots ,& \text{(num 38s on day 3)} \end{bmatrix}
# $$
# 
# Assign your answer to the numpy array `y_data`.
# 
# Hint: `.pivot_table()` is nice for this.

# %%
# Define the possible numbers on the roulette wheel
numbers = np.arange(0, 38)  
colors = ['green'] + ['red', 'black'] * 18  + ['green']

# simulate some data
num_days = 100
num_spins_per_day = 100
num_rows = num_days*num_spins_per_day
my_data = pd.DataFrame({'number':np.random.choice(numbers, num_rows)})
my_data['color'] = my_data.number.apply( lambda num : colors[num])
my_data['day'] = np.repeat(np.arange(1,(num_spins_per_day+1)),num_days)


# %%
y_data = np.array(my_data.pivot_table(index='day', columns='number', values='color', aggfunc='count').fillna(0))

# %% [markdown]
# 3.
# 
# Calculate the Dirichlet posterior hyperparameters. Assign a 1-d Numpy array to the variable named `post_hyperparams`

# %%
post_hyperparams = y_data.sum(axis=0) + prior_hyperparams

# %%
post_hyperparams

# %% [markdown]
# 4.
# 
# If you can only bet one one number, which number are you picking? Possible answer choices include $1, \ldots, 36,37,38$ where the last two numbers follow the aforementioned convention. 
# 
# Assign your answer to `best_bet`
# 
# Hint: be careful...Python uses 0-based indexing but Roulette starts counting at $1$.
# 

# %%
best_bet = post_hyperparams.argmax()+1
best_bet

# %% [markdown]
# 5.
# 
# 
# If the true model is that every outcome is equally likely, will you make money in the long run?
# 
# In other words, suppose that your prior is **super informative,** and simulate from the prior predictive distribution. For each spin of the wheel, calculate your profit (or loss). Use fixed bet sizes.
# 
# Visualize your simulated profit and losses appropriately, and justify your answer with these visualizations.

# %%
from scipy.stats import multinomial
#import matplotlib.pyplot as plt

# %%
uniform_prior = np.ones(38)/38

# %%
num_of_spins = 10000

outcomes = [np.argmax(multinomial.rvs(1, uniform_prior))+1 for _ in range(num_of_spins)] # generate 10000 outcomes pulling from a uniform prior

bet_on_number = best_bet
bet_amount = 1
payout = 35

profit_history = []
current_profit = 0

for outcome in outcomes:
    if outcome == bet_on_number:
        current_profit += bet_amount*payout
    else:
        current_profit -= bet_amount
    profit_history.append(current_profit)



# %% [markdown]
# plt.plot(range(num_of_spins), profit_history)
# plt.axhline(0, color='black', linestyle='-', linewidth=1.5, label='Break-Even Point')
# plt.title('Simulated Profit/Loss from Betting on Roulette', fontsize=16)
# plt.xlabel('Number of Spins', fontsize=12)
# plt.ylabel('Total Profit ($)', fontsize=12)

# %% [markdown]
# ### Problem 3: Stocks
# 
# In example 2 of module 4, we showed that if one has multivariate normal observations, then the **Normal Inverse Wishart** prior is the conjugate prior.
# 
# In this example, we'll be using stocks data again, but with a **daily sampling frequency.**

# %% [markdown]
# 1.
# 
# Download the data `stocks.csv` and assign it to a `pandas` `DataFrame` called `adj_prices`. Be sure to set the date as the index.
# 
# Calculate percent returns (scaled by $100$) and call the resulting `DataFrame` `rets`. After understanding where they come from, be sure to remove any `NaN`s.

# %%
adj_prices = pd.read_csv('stocks.csv', index_col=0)
rets = adj_prices.pct_change().dropna()*100

# %% [markdown]
# 2.
# 
# Write a function called `sim_data` that can simulate from either the prior predictive or the posterior predictive distribution.
# 
# Here's a template based on how I did it. You may use all of it or just some of it. However, do be sure to keep the same function signature :) Every input or output is either a `float` or a 2-d Numpy array.

# %%
from scipy.stats import invwishart
from scipy.stats import multivariate_normal

def sim_data(nu0, Lambda0, mu0, kappa0, num_sims):
    sigma_samples = invwishart(nu0, Lambda0).rvs(num_sims)
    mu_samples = np.array(
        [
            multivariate_normal.rvs(mean = mu0,cov =  S/kappa0) for S in sigma_samples
        ]
    )

    theta_samples = zip(mu_samples, sigma_samples)
    fake_y = np.array([multivariate_normal.rvs(mean=mu, cov=S) for mu, S in theta_samples])
    return fake_y

# %% [markdown]
# 3.
# 
# Pick a prior by assigning specific values to `nu0`, `Lambda0`, `mu0` and `kappa0`. Use your `sim_data()` function to choose wisely.

# %%
nu0 = 10
Lambda0 = np.array([[1/4, 0], [0, 1/4]])
mu0 = np.array([0, 0])
kappa0 = 10

fake_y = sim_data(nu0, Lambda0, mu0, kappa0, 1000)

# %% [markdown]
# 4.
# 
# Calculate the posterior NIW distribution. Assign the distribution's hyperparameters to `nu_n`, `kappa_n`, `mu_n`, and `Lambda_n`

# %%
n = len(rets)
ybar = rets.mean().values
ybarminusmu = (ybar-mu0).reshape(2,1)
S =(rets.transpose() @ rets).values

# %%
nu_n = nu0+n
kappa_n = kappa0 +n
mu_n = mu0*kappa0/(kappa0+n) + ybar*n/(kappa0+n)
Lambda_n = (Lambda0+S+
            ybarminusmu @ ybarminusmu.transpose()*
            kappa0*n/(kappa0+n))

# %% [markdown]
# 5.
# 
# Simulate $1239$ observations from the posterior predictive distribution. You can use `sim_data()` for this, too. Assign your array to the variable `post_pred_sims`
# 
# 
# 
# Does this model fit lower-frequency stock returns better? Discuss.
# 
# 

# %%
# uncomment after you have a working implementation of sim_data()!
post_pred_sims = sim_data(nu_n, Lambda_n, mu_n, kappa_n, 1239)


# %% [markdown]
# 6. 
# 
# Now that you have a posterior for $\mu$ and $\Sigma$, that means you have an understanding of the distribution of future returns. Let's find the "optimal" portfolio weights! 
# 
# 
# Portfolio weights $w = w_1, \ldots, w_k$ are proportions of your wealth that you invest in each security. We want to find the vector that that **maximizes "risk-adjusted return"**
# 
# $$
# \underbrace{\mathbf{w}^\intercal \mathbb{E}_{\text{posterior}}[y]}_{\text{reward}} - \frac{1}{2} \gamma \underbrace{\mathbf{w}^\intercal \mathbb{V}_{\text{posterior}}[y] \mathbf{w}}_{\text{risk}}
# $$ 
# 
# subject to the constraint that you have a finite amount of money:
# 
# $$
# \sum_i w_i = 1
# $$
# 
# $\gamma > 0$ is a user-chosen risk-aversion parameter. Describe your choice of it, and run the function below to get your optimal portfolio weights. Discuss your results. Some examples of questions to consider are:
# 
#  - Do my weights indeed sum to $1$?
#  - What are some problems with choosing $\gamma$ very close to $0$?
#  - What are some problems with choosing $\gamma$ very large?
#  - What do I think most peoples' $\gamma$ are?
#  - Knowing what I know now, would I go back and change around the prior to get a posterior that gives me a better answer here? 

# %%
def get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma, s = 1):
    k = len(mu_n)
    post_mean = mu_n
    post_var = Lambda_n / (nu_n - k - 1)*(1 + 1/kappa_n)
    V_inv = np.linalg.inv(post_var)
    ones = np.repeat(1, k)
    q1 = ones.transpose() @ V_inv @ post_mean
    q2 = ones.transpose() @ V_inv @ ones
    le_fraction = (q1 - gamma*s)/q2
    return V_inv @ (post_mean - le_fraction * ones) / gamma
    
best_weights = get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma=.05, s = 1)


# %%
gamma_values = np.arange(0.05, 1.05, 0.05)

all_weights = []

for gamma in gamma_values:
    weights = get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma, s = 1)
    all_weights.append(weights)

weights_array = np.array(all_weights)

# %% [markdown]
# plt.plot(gamma_values, weights_array[:, 0], marker='o', linestyle='-', label='Weight of Asset 1')
# plt.plot(gamma_values, weights_array[:, 1], marker='o', linestyle='-', label='Weight of Asset 2')
# plt.title('Optimal Portfolio Weights vs. Risk Aversion (gamma)', fontsize=16)
# plt.xlabel('Gamma (Risk Aversion)', fontsize=12)
# plt.ylabel('Optimal Weight', fontsize=12)
# plt.legend()
# plt.grid(True)

# %%



