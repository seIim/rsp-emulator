import jax, jax.numpy as jnp, jax.random as random
from flax.training import train_state
import matplotlib.pyplot as plt
from models import Transformer
from dataloader import *
import pandas as pd
from utils import *
import optax, tqdm


@jax.jit
def test_loss_fn(params):
    predictions = model.apply(params, X_test)
    loss = mse_loss(predictions, y_test)
    return loss


def create_train_state(rng, model, learning_rate, input_dim):
    params = model.init(rng, jnp.ones((batch_size, input_dim)))
    
    tx = optax.chain(
       #optax.adaptive_grad_clip(1.0),
       optax.adamw(learning_rate)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def smoothness_loss(pred):
    # Penalize large differences between consecutive predictions
    diffs = pred[:, 1:] - pred[:, :-1]
    return jnp.mean(diffs**2)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = model.apply(params, batch['inputs'])
        loss = mse_loss(predictions, batch['targets']) + 0.3 * smoothness_loss(predictions)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


X, y = create_dataset()
key1, key2 = random.split(random.PRNGKey(611))
X_train, y_train, X_test, y_test = train_test_split(key1, X, y, split=0.2, shuffle=True)
input_dim  = X_train.shape[1] # features
output_dim = y_train.shape[1]

# Hyperparameters
num_layers = 4
model_dim = 512
num_heads = 8
ff_dim = 512
batch_size = 512
learning_rate = 0.001
num_epochs = 1
sequence_length = 100

model = Transformer(num_layers, model_dim, num_heads, ff_dim, output_dim)
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
