
// A more modern approach using an LKJ prior
data {
  int<lower=1> K;       // Number of dimensions
  int<lower=0> N;       // Number of data points
  array[N] vector[K] y;       // Data
}

parameters {
  vector[K] mu;                     // Mean vector
  cholesky_factor_corr[K] L_Rho;    // Cholesky factor of the correlation matrix
  vector<lower=0>[K] sigma;         // Vector of standard deviations
}

model {
  // Priors
  mu ~ normal(0, 5);
  
  // LKJ prior on the Cholesky factor of the correlation matrix.
  // eta = 1.0 gives a uniform distribution over all valid correlation matrices.
  L_Rho ~ lkj_corr_cholesky(1.0);
  
  // Prior on standard deviations. Half-Cauchy is a good weakly informative choice.
  sigma ~ cauchy(0, 2.5);

  // Likelihood
  // We construct the covariance matrix from sigma and L_Rho
  matrix[K, K] Sigma = quad_form_diag(L_Rho, sigma);
  y ~ multi_normal_cholesky(mu, diag_pre_multiply(sigma, L_Rho));
}

generated quantities {
    matrix[K, K] Rho = multiply_lower_tri_self_transpose(L_Rho);
    matrix[K, K] Sigma = quad_form_diag(L_Rho, sigma);
}