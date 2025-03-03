import jax
import jax.numpy as jnp
import jax.random as random
from flax.training import train_state
import optax
import numpy as np
import pandas as pd
import os
import sys
from functools import partial
from tqdm import tqdm
project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from emulator.layers import FiLMEmbeddingTransformer
from emulator.dataloader import *
from emulator.utils import *


jax.config.update("jax_default_matmul_precision", "float32")
sequence_length = 100

X = pd.read_csv('../../data/rsp.bettergridsetc/features.dat', sep=r'\s+')
y = jnp.load('../../data/rsp.bettergridsetc/targets.npy')
features = ['RSP_Z', 'RSP_X', 'RSP_mass', 'RSP_L', 'RSP_Teff', 'RSP_alfa']
X = jnp.array(X[features].values)
key1, key2 = random.split(random.PRNGKey(42))
X_train, y_train, X_test, y_test = train_test_split(key1, X, y, split=0.2,
                                                    shuffle=True)
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
std_y = y_train.std()
y_train = y_train/std_y
y_test = y_test/std_y
X_train = (X_train - mu)/sd
X_test = (X_test - mu)/sd
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]


def spectral_loss(pred, target):
    pred_fft = jnp.fft.rfft(pred, axis=1)
    target_fft = jnp.fft.rfft(target, axis=1)
    return jnp.mean(jnp.abs(pred_fft - target_fft)**2)


def gradient_penalty(model, params, inputs):
    def loss_fn(inputs):
        pred = model.apply(params, inputs)
        return jnp.mean(pred**2)
    grads = jax.grad(loss_fn)(inputs)
    return jnp.mean(jnp.concatenate([jnp.ravel(g)**2 for g in jax.tree.leaves(grads)]))


def create_train_state(rng, model, X_sample, learning_rate=3e-4):
    params = model.init(rng, X_sample[:1])
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=1000,
        decay_steps=10000
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule)
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, key, model):
    def loss_fn(params):
        pred = model.apply(params, batch['X'])
        mse = jnp.mean((pred - batch['y'])**2)
        spec = spectral_loss(pred, batch['y'])
        gp = gradient_penalty(model, params, batch['X'])
        return 0.5*mse + 0.3*spec + 0.2*gp

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def main():
    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 1000
    SEED = 42

    test_data = {'X': X_test, 'y': y_test}

    key = jax.random.PRNGKey(SEED)
    model = FiLMEmbeddingTransformer(model_dim=512, ff_dim=1024)
    state = create_train_state(key, model, X_train)

    for epoch in tqdm(range(NUM_EPOCHS)):
        key, subkey = jax.random.split(key)
        # This shuffles the dataset
        perm = jax.random.permutation(subkey, len(X_train))
        epoch_loss = []

        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = perm[i:i+BATCH_SIZE]
            batch = {'X': X_train[batch_idx], 'y': y_train[batch_idx]}
            state, loss = train_step(state, batch, key, model)
            epoch_loss.append(loss)

        # Evaluation
        test_pred = model.apply(state.params, test_data['X'])
        test_loss = jnp.mean((test_pred - test_data['y'])**2)
        tqdm.write(f"Epoch {epoch+1}: Train Loss {np.mean(epoch_loss):.4f}, \
                Test MSE {test_loss:.4f}")
    save_model('new_arch_res', state)


if __name__ == "__main__":
    main()
