import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Rename the model instance to avoid naming conflict
transformer_model = EmbeddingTransformer(
    num_layers=num_layers,
    model_dim=model_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    output_dim=output_dim,
    sequence_length=sequence_length
)

# Modified probabilistic model
def bayesian_model(y_obs):
    # Priors (using training data statistics)
    x = numpyro.sample('x', dist.Normal(mu, sd))  # Using original data statistics
    # Neural network forward pass (using trained parameters)
    pred = transformer_model.apply(state.params, x[None, :]).squeeze(axis=0)
    # Observation noise prior
    sigma = numpyro.sample('sigma', dist.HalfNormal(0.5))
    # Likelihood
    numpyro.sample('obs', dist.Normal(pred, sigma), obs=y_obs)

def run_hmc_inference(y_target, num_samples=2000, num_warmup=1000):
    nuts_kernel = NUTS(bayesian_model, init_strategy=numpyro.infer.init_to_sample)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
    # Run inference
    mcmc.run(random.PRNGKey(0), y_obs=y_target)
    return mcmc.get_samples()

