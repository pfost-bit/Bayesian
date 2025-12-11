data {
  int<lower=0> J;         // number of rat groups
  array[J] int<lower=0> n;      // sample size for each group
  array[J] int<lower=0> y;      // number of successes for each group
}

parameters {
  real<lower=0> alpha;    // alpha parameter for the Beta distribution
  real<lower=0> beta;     // beta parameter for the Beta distribution
  array[J] real<lower=0, upper=1> z; // success probabilities for each group
}

model {
  // Priors
  alpha ~ gamma(1, 1);    // Gamma prior for alpha
  beta ~ gamma(1, 1);     // Gamma prior for beta
  
  // Complete-Data Likelihood
  for (j in 1:J) {
    z[j] ~ beta(alpha, beta);  // hierarchical model
    y[j] ~ binomial(n[j], z[j]); // binomial likelihood
  }
}
