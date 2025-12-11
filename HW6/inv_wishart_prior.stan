
// Samples from the prior predictive distribution to check prior assumptions.

data {
  int<lower=1> K;           // Number of dimensions (index funds)

  // --- Normal-Inverse-Wishart Prior Parameters ---
  real<lower=0> kappa0;
  vector[K] mu0;
  real<lower=K-1> nu0;
  cov_matrix[K] Lambda0;
}

// No parameters to fit, so this block is empty.
parameters {}

// No posterior to sample, so this block is empty.
model {}

generated quantities {
  // 1. Draw a covariance matrix from the Inverse-Wishart prior
  cov_matrix[K] Sigma_pred = inv_wishart_rng(nu0, Lambda0);
  
  // 2. Draw a mean vector from the Normal prior, conditioned on the sampled Sigma
  vector[K] mu_pred = multi_normal_rng(mu0, Sigma_pred / kappa0);
  
  // 3. Draw a 'predicted' data point from the likelihood using the parameters from steps 1 & 2
  vector[K] y_pred = multi_normal_rng(mu_pred, Sigma_pred);
  
}