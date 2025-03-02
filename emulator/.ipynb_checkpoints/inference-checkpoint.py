import jax, jax.random as random, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.distributions import TransformedDistribution, LogNormal
import numpyro.distributions as dist
import numpyro


def infer_inputs(y_test, body_fn, X_mean, X_std, y_mean, y_std,
                 num_samples=1000, num_warmup=500, num_chains=1):
    @jax.jit
    def loss_fn(X):
        pred = body_fn(X.reshape(1,-1)).squeeze()
        return jnp.mean((pred - y_test)**2)
    
    X_init = jax.scipy.optimize.minimize(
        loss_fn, jnp.zeros(5), method='BFGS'
    ).x

    def model(y_obs):
        X = numpyro.sample("features", dist.Normal(jnp.zeros(5), jnp.ones(5)))
        pred = body_fn(X.reshape(1,-1)).squeeze()
        noise_scale = numpyro.sample("noise", dist.HalfNormal(0.1))
        numpyro.sample(
            "obs",
            dist.StudentT(4.0, pred, noise_scale),
            obs=y_test
        )
		
    nuts_kernel = NUTS(
        model,
        target_accept_prob=0.9,
        max_tree_depth=12,
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    
    # Initialize near MAP estimate
    init_params = {
        "mass": jnp.repeat(X_init[0], num_chains),
        "luminosity": jnp.repeat(X_init[1], num_chains),
        "teff": jnp.repeat(X_init[2], num_chains),
        "Z": jnp.repeat(X_init[3], num_chains),
        "X": jnp.repeat(X_init[4], num_chains),
    }
    
    mcmc.run(
        random.PRNGKey(611),
        y_obs=y_test,
        init_params=init_params
    )
    
    return mcmc.get_samples()    
