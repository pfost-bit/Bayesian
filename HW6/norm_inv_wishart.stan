data {

  int<lower=1> K;           // Number of dimensions (index funds)
  int<lower=0> N;           // Number of data points
  array[N] vector[K] y;         // Data: N observations of K-dimensional vectors

  //  Normal-Inverse-Wishart Prior Parameters 
  real<lower=0> kappa0;         // Scaling factor for the prior on the mean
  vector[K] mu0;                // The prior mean
  real<lower=K-1> nu0;          // Degrees of freedom for the Inverse-Wishart prior
  cov_matrix[K] Lambda0;        // Scale matrix for the Inverse-Wishart prior (must be positive definite)
}

parameters {
  vector[K] mu;             // Mean vector of the data
  cov_matrix[K] Sigma;      // Covariance matrix of the data
}

model {
  //  Normal-Inverse-Wishart Prior 
  
  // The Inverse-Wishart prior on the covariance matrix Sigma.
  Sigma ~ inv_wishart(nu0, Lambda0);

  // The prior on mu, conditioned on Sigma. 
  mu ~ multi_normal(mu0, Sigma / kappa0);

  //  Likelihood 
  // The data is modeled as a multivariate normal distribution.
  y ~ multi_normal(mu, Sigma);
}