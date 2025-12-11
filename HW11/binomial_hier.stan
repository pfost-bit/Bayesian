data {
  int<lower=0> J;         // number of rocketes
  array[J] int<lower=0> n;      // array for the number of launches of each rocket
  array[J] int<lower=0> y;      // array for the number of failures of each rocket
}

parameters {
  real<lower=0> alpha;    // alpha parameter for the Beta distribution
  real<lower=0> beta;     // beta parameter for the Beta distribution
  array[J] real<lower=0, upper=1> z; // probabilities of failure for each rocket
}

model {
  // Priors
  alpha ~ gamma(1, 1);    // Weakly informative prior on alpha
  beta ~ gamma(1, 1);     // Weakly informative prior on beta

  // Complete-Data Likelihood
  for (j in 1:J) {
    z[j] ~ beta(alpha, beta);  // hierarchical model
    y[j] ~ binomial(n[j], z[j]); // binomial likelihood
  }
}

generated quantities{
    real mean_p_fail = beta_rng(alpha, beta);
}
