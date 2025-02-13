import jax, jax.random as random, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro


def infer_inputs(y_test, body_fn, input_dim=5, num_samples=100, num_warmup=100, num_chains=1):
    def model(y_obs):
        X = numpyro.sample("X", dist.Uniform(jnp.zeros(input_dim)-3, jnp.zeros(input_dim)+3))
        predictions = body_fn(X.reshape(1,-1))
        predictions = predictions.squeeze()
        numpyro.sample("y", dist.Normal(predictions, 0.1), obs=y_obs)
    
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    rng_key = random.PRNGKey(42)
    mcmc.run(rng_key, y_obs=y_test)
    
    samples = mcmc.get_samples()
    return samples["X"]


