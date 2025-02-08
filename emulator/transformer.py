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
        loss = mse_loss(predictions, batch['targets']) + 1.0 * smoothness_loss(predictions)
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

import orbax.checkpoint as orbax
from flax.training import orbax_utils
import os
save_dir = os.path.abspath("./checkpoints/")
def save_params(train_state, save_dir):
    options = orbax.CheckpointManagerOptions(max_to_keep=1)  # Keep only the last 3 checkpoints
    checkpoint_manager = orbax.CheckpointManager(save_dir, orbax.PyTreeCheckpointer(), options)
    state_to_save = {'params': train_state.params}
    checkpoint_manager.save(
        step=0,
        items=state_to_save,
        save_kwargs={'save_args': orbax_utils.save_args_from_target(state_to_save)})

def load_params(save_dir):
    checkpoint_manager = orbax.CheckpointManager(save_dir, orbax.PyTreeCheckpointer())
    step = checkpoint_manager.latest_step()
    if step is None:
        raise ValueError(f"No checkpoints found in {save_dir}")
    restored = checkpoint_manager.restore(step)
    params = restored['params']
    return params

save_params(state, save_dir)
predictions = jnp.asarray(model.apply(state.params, X_train[:11]))
predictions_0 = predictions

for i in range(10):
    plt.plot(jnp.linspace(0,1,sequence_length),predictions[i])
    plt.plot(jnp.linspace(0,1,sequence_length),y_train[i])
    plt.savefig(f'../figs/train_preds_{i}.pdf', bbox_inches='tight')
    plt.show()

params = load_params(save_dir)
predictions = jnp.asarray(model.apply(params, X_train[:11]))
print(jnp.sum(predictions - predictions_0))
