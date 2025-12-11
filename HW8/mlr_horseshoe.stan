data {
  int N; // number of observations
  int K; // number of predictors
  vector[N] y; // dependent variable
  matrix[N, K] X; // matrix of predictors
}

parameters {
  real alpha;            // Intercept
  vector[K] beta;        // Coefficients for predictors
  real<lower=0> sigma;   // Standard deviation instead of variance

  // additional parameters for horseshoe prior
  vector<lower=0>[K] lambda;  // Local shrinkage parameters
  real<lower=0> tau;          // Global shrinkage parameter
}

model {
  // Prior for the intercept
  alpha ~ normal(0, 10);

  // Horseshoe prior on beta
  beta ~ normal(0, tau * lambda); // part 1 of horseshoe prior
  lambda ~ cauchy(0, 1); // part 2 of the horseshoe prior
  tau ~ cauchy(0, 1); // highly debated!

  // Prior for sigma
  sigma ~ lognormal(0, 1); // Adjusted the scale for practical purposes

  // Likelihood
  vector[N] mu = alpha + X * beta;
  target += normal_lpdf(y | mu, sigma);
}
