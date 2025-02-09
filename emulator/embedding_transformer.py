import jax, jax.numpy as jnp, jax.random as random
from models import EmbeddingTransformer
from flax.training import train_state
import matplotlib.pyplot as plt
from dataloader import *
from utils import *
import optax, tqdm


# fix precision loss on a100
jax.config.update("jax_default_matmul_precision", "high")



@jax.jit
def test_loss_fn(params):
    predictions = model.apply(params, X_test)
    loss = mse_loss(predictions, y_test) + smoothness_loss(predictions)
    return loss


def create_train_state(rng, model, learning_rate, input_dim):
    params = model.init(rng, jnp.ones((batch_size, input_dim)))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=num_epochs * len(X_train) // batch_size
    )
    tx = optax.chain(
       optax.adaptive_grad_clip(1.0),
       optax.adamw(learning_rate=schedule)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = model.apply(params, batch['inputs'])
        smoothing = 0.3 * smoothness_loss(predictions)
        loss = mse_loss(predictions, batch['targets'])
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Hyperparameters
num_layers = 4
model_dim = 512
num_heads = 8
ff_dim = 1024 
batch_size = 1024
learning_rate = 1e-3
num_epochs = 2000
sequence_length = 100

# fetch dataset
X, y = create_dataset(sequence_length=sequence_length)
key1, key2 = random.split(random.PRNGKey(42))
X_train, y_train, X_test, y_test = train_test_split(key1, X, y, split=0.2, shuffle=True)
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
std_y = y_train.std()
y_train = y_train/std_y
y_test = y_test/std_y
X_train = (X_train - mu)/sd
X_test = (X_test - mu)/sd
input_dim  = X_train.shape[1]
output_dim = y_train.shape[1]

# init the model & train state
model = EmbeddingTransformer(
    num_layers=num_layers,
    model_dim=model_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    output_dim=output_dim,
    sequence_length=sequence_length
)
state = create_train_state(key2, model, learning_rate, input_dim=input_dim)


loss = test_loss = 0
with tqdm.trange(num_epochs) as t:
    for epoch in t:
        epoch_loss = []
        for batch in batch_generator(X_train, y_train, batch_size):
            state, loss = train_step(state, batch)
            epoch_loss.append(loss) 
        test_loss = test_loss_fn(state.params)
        t.set_postfix_str(f"train: {jnp.mean(jnp.array(epoch_loss)):.4f}, test: {test_loss:.4f}", refresh=False)


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
# 3. Run inference on a test case
test_idx = 0  # Example index
samples = run_hmc_inference(y_test[test_idx])
# 4. Analyze results
print("\nInference results:")
print(f"True X (original scale): {X_test[test_idx] * sd + mu}")
print(f"Inferred X mean: {samples['x'].mean(axis=0)}")
print(f"Observation noise: {samples['sigma'].mean():.4f}")
# 5. Plot posterior predictive distribution
ppc_samples = jax.vmap(lambda x: transformer_model.apply(state.params, x[None]))(samples['x'])
ppc_mean = ppc_samples.mean(axis=0)
ppc_std = ppc_samples.std(axis=0)
plt.figure(figsize=(10, 4))
plt.plot(y_test[test_idx], 'k-', label='Observed')
plt.plot(ppc_mean, 'r--', label='Posterior mean')
plt.fill_between(np.arange(sequence_length),
                 ppc_mean - 2*ppc_std,
                 ppc_mean + 2*ppc_std,
                 alpha=0.3, color='red')
plt.title("Posterior Predictive Distribution")
plt.legend()
plt.savefig('post-pred.pdf', bbox_inches='tight')
#from utils import save_params
#import os
#save_dir = os.path.abspath("./checkpoints")
#save_params(state, save_dir)
#predictions = jnp.asarray(model.apply(state.params, X_train[:11]))*std_y

#for i in range(10):
#    plt.plot(jnp.linspace(0,1,sequence_length),predictions[i])
#    plt.plot(jnp.linspace(0,1,sequence_length),y_train[i]*std_y)
#    plt.savefig(f'../figs/train_preds_{i}.pdf', bbox_inches='tight')
#    plt.clf()
#predictions = jnp.asarray(model.apply(state.params, X_test[:11]))*std_y
#for i in range(10):
#    plt.plot(jnp.linspace(0,1,sequence_length),predictions[i])
#    plt.plot(jnp.linspace(0,1,sequence_length),y_test[i]*std_y)
#    plt.savefig(f'../figs/test_preds_{i}.pdf', bbox_inches='tight')
#    plt.clf()
