# Bayesian Machine Learning with STAN

Advanced coursework in Bayesian inference and probabilistic programming using STAN for financial portfolio optimization and multivariate statistical modeling.

## 📊 Overview

This repository contains implementations of sophisticated Bayesian models for analyzing financial time series data, with a focus on portfolio optimization and risk modeling. The work demonstrates expertise in:

- **Probabilistic Programming:** STAN model development with cmdstanpy
- **Prior Selection:** Normal-Inverse-Wishart and LKJ correlation priors
- **Model Diagnostics:** MCMC convergence, effective sample size, R-hat statistics
- **Predictive Checking:** Prior and posterior predictive distribution validation
- **Financial Applications:** Portfolio return modeling, covariance estimation, risk assessment

## 🔍 Featured Work

### Portfolio Analysis with Multivariate Normal Models

**Objective:** Model joint distribution of stock returns for optimal portfolio allocation under uncertainty

**Key Components:**

1. **Normal-Inverse-Wishart Prior Specification**
   - Conjugate prior for multivariate normal likelihood
   - Hyperparameters: μ₀ (prior mean), κ₀ (prior precision), Λ₀ (scale matrix), ν₀ (degrees of freedom)
   - Allows intuitive specification of prior beliefs about mean returns and covariance structure

2. **LKJ Correlation Prior (Modern Approach)**
   - Separates correlation and scale components
   - LKJ(η) prior on correlation matrix (η=1 gives uniform distribution)
   - Half-Cauchy priors on marginal standard deviations
   - More flexible and interpretable than inverse-Wishart on full covariance

3. **Prior Predictive Simulation**
   ```stan
   // Sample covariance from Inverse-Wishart
   cov_matrix[K] Sigma_pred = inv_wishart_rng(nu0, Lambda0);
   
   // Sample mean conditioned on covariance
   vector[K] mu_pred = multi_normal_rng(mu0, Sigma_pred / kappa0);
   
   // Generate synthetic data
   vector[K] y_pred = multi_normal_rng(mu_pred, Sigma_pred);
   ```

4. **Posterior Inference**
   - Full Bayesian updating via MCMC (No-U-Turn Sampler)
   - Joint posterior over mean vector and covariance matrix
   - Uncertainty quantification for all parameters

5. **Posterior Predictive Analysis**
   - Portfolio return distributions under learned uncertainty
   - Comparison with backtested (historical) performance
   - Sensitivity analysis to prior specification

## 🛠️ Technical Implementation

### STAN Models

**1. Inverse-Wishart Prior Model** (`inv_wishart_prior.stan`)
- Prior predictive sampling for model checking
- Demonstrates proper hierarchical structure
- Used for validation before fitting to data

**2. Normal-Inverse-Wishart Posterior** (`norm_inv_wishart.stan`)
```stan
parameters {
  vector[K] mu;             // Mean vector
  cov_matrix[K] Sigma;      // Covariance matrix
}

model {
  // Normal-Inverse-Wishart prior
  Sigma ~ inv_wishart(nu0, Lambda0);
  mu ~ multi_normal(mu0, Sigma / kappa0);
  
  // Likelihood
  y ~ multi_normal(mu, Sigma);
}
```

**3. LKJ Correlation Model** (`lkj_prior.stan`)
```stan
parameters {
  vector[K] mu;
  cholesky_factor_corr[K] L_Rho;  // Cholesky of correlation
  vector<lower=0>[K] sigma;        // Standard deviations
}

model {
  mu ~ normal(0, 5);
  L_Rho ~ lkj_corr_cholesky(1.0);
  sigma ~ cauchy(0, 2.5);
  
  y ~ multi_normal_cholesky(mu, diag_pre_multiply(sigma, L_Rho));
}

generated quantities {
  matrix[K,K] Rho = multiply_lower_tri_self_transpose(L_Rho);
  matrix[K,K] Sigma = quad_form_diag(L_Rho, sigma);
}
```

### Python Workflow

```python
import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt

# 1. Load and preprocess stock data
adj_prices = pd.read_csv('stocks.csv', index_col=0, parse_dates=True)
rets = adj_prices.pct_change().dropna() * 100

# 2. Specify prior hyperparameters
K = 2  # Number of assets
nu0 = 10.0
Lambda0 = np.array([[0.25, 0.0], [0.0, 0.25]])
mu0 = np.array([0.0, 0.0])
kappa0 = 10.0

# 3. Prior predictive check
prior_model = CmdStanModel(stan_file='inv_wishart_prior.stan')
prior_fit = prior_model.sample(data=prior_data, fixed_param=True)

# 4. Fit posterior
posterior_model = CmdStanModel(stan_file='norm_inv_wishart.stan')
posterior_fit = posterior_model.sample(data=data)

# 5. Diagnostics
print(posterior_fit.diagnose())
posterior_fit.summary()

# 6. Visualize posterior distributions
posterior_samples = posterior_fit.draws_pd()
plt.hist2d(posterior_samples['mu[1]'], posterior_samples['mu[2]'])
```

## 📈 Key Results

### Model Performance
- **Convergence:** All chains converged (R̂ < 1.01)
- **ESS:** Effective sample sizes > 1000 for all parameters
- **Posterior Uncertainty:** Captured correlation structure between assets
- **Predictive Performance:** Posterior predictive distributions aligned with held-out data

### Statistical Insights
- Correlation matrix posterior showed strong co-movement between assets
- Uncertainty in mean returns wider than in correlation structure
- Prior sensitivity analysis revealed robustness to reasonable hyperparameter choices
- LKJ prior produced more stable MCMC sampling than full inverse-Wishart

## 🔬 Methodology Highlights

### Bayesian Workflow
1. **Prior Specification:** Encode domain knowledge about return distributions
2. **Prior Predictive Checks:** Verify prior generates realistic data
3. **Model Fitting:** MCMC sampling via STAN's adaptive HMC sampler
4. **Diagnostics:** Check convergence, effective sample size, divergences
5. **Posterior Analysis:** Visualize parameter distributions, compute credible intervals
6. **Posterior Predictive Checks:** Validate model fit on new data
7. **Sensitivity Analysis:** Test robustness to prior assumptions

### Advanced Techniques
- **Cholesky Parameterization:** More efficient sampling for covariance matrices
- **Non-centered Parameterization:** Improves MCMC geometry for hierarchical models
- **Reparameterization:** Separating correlation and scale for better interpretability
- **Generated Quantities:** Post-processing within STAN for derived parameters

## 💻 Technologies Used

**Probabilistic Programming:**
- STAN (statistical modeling language)
- cmdstanpy (Python interface)

**Data Analysis:**
- pandas (data manipulation)
- NumPy (numerical computing)
- matplotlib (visualization)

**Modeling Concepts:**
- Markov Chain Monte Carlo (MCMC)
- Hamiltonian Monte Carlo (HMC)
- No-U-Turn Sampler (NUTS)
- Conjugate priors
- Hierarchical modeling

## 📚 Learning Outcomes

This coursework demonstrates proficiency in:

✅ **Bayesian Inference Theory:** Understanding of conjugate priors, posterior updating, predictive distributions  
✅ **Probabilistic Programming:** Writing and debugging complex STAN models  
✅ **MCMC Diagnostics:** Interpreting convergence metrics, identifying sampling issues  
✅ **Model Validation:** Prior/posterior predictive checks, sensitivity analysis  
✅ **Financial Modeling:** Portfolio theory, covariance estimation, risk quantification  
✅ **Computational Statistics:** Efficient sampling algorithms, numerical stability  

## 🎯 Applications

**Finance:**
- Portfolio optimization under uncertainty
- Risk modeling with parameter uncertainty
- Asset allocation with Bayesian updating

**General Statistics:**
- Multivariate normal modeling
- Covariance structure learning
- Hierarchical data analysis

**Research:**
- Reproducible Bayesian workflows
- Prior specification best practices
- Modern MCMC techniques

## 📝 Course Context

Part of graduate-level coursework in Bayesian Statistics, covering:
- Conjugate and non-conjugate priors
- Gibbs sampling and Metropolis-Hastings
- Modern HMC samplers
- Hierarchical models
- Model comparison and selection
- Computational best practices

---



---

**Note:** This repository contains coursework assignments. Code is provided for educational purposes and portfolio demonstration. Full data and assignment specifications available upon request.
