data {
  int K; // number of output categories
  int N; // number of rows of data
  int D; // number of predictors
  array[N] int y;
  matrix[N, D] x;
}
parameters {
  matrix[D, K] beta;
}
model {
  matrix[N, K] x_beta = x * beta;

  to_vector(beta) ~ normal(0, 5);

  for (i in 1:N) {
    y[i] ~ categorical_logit(x_beta[i]');
  }
}