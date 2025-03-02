import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, orbax_utils
from flax.linen import SpectralNorm as spectral_norm
import optax
import numpy as np
import pandas as pd
from utils import *
from dataloader import *
import os
from functools import partial
from tqdm import tqdm

jax.config.update("jax_default_matmul_precision", "float32")
sequence_length=100

# X, y = create_dataset(sequence_length=sequence_length)
# key1, key2 = random.split(random.PRNGKey(42))
# X_train, y_train, X_test, y_test = train_test_split(key1, X, y, split=0.2, shuffle=True)
# mu = X_train.mean(axis=0)
# sd = X_train.std(axis=0)
# std_y = y_train.std()
# y_train = y_train/std_y
# y_test = y_test/std_y
# X_train = (X_train - mu)/sd
# X_test = (X_test - mu)/sd
# input_dim  = X_train.shape[1]
# output_dim = y_train.shape[1]
# OTHER DATASET
X = pd.read_csv('../data/rsp.bettergridsetc/features.dat', sep=r'\s+')
y = jnp.load('../data/rsp.bettergridsetc/targets.npy')
features = ['RSP_Z','RSP_X','RSP_mass','RSP_L','RSP_Teff','RSP_alfa']
X = jnp.array(X[features].values)
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
class FiLMGenerator(nn.Module):
    model_dim: int
    @nn.compact
    def __call__(self, x):
        gamma = nn.Dense(self.model_dim)(x)
        beta = nn.Dense(self.model_dim)(x)
        return gamma, beta

class PhaseAwareTransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    ff_dim: int
    
    @nn.compact
    def __call__(self, x, phase_embed):
        x = jnp.concatenate([x, phase_embed], axis=-1)
        x = nn.Dense(self.model_dim)(x)
        
        attn = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.model_dim)(x)
        x = x + attn

        x = x + nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.ff_dim),
            nn.gelu,
            nn.Dense(self.model_dim)
        ])(x)
        
        return x

class ImprovedTransformer(nn.Module):
    num_layers: int = 4
    model_dim: int = 128
    num_heads: int = 8
    ff_dim: int = 256
    sequence_length: int = 100

    def setup(self):
        self.static_encoder = nn.Sequential([
            nn.Dense(256), nn.gelu, nn.Dense(self.model_dim)
        ])
        self.phase_encoder = nn.Dense(self.model_dim)
        self.film_generators = [FiLMGenerator(self.model_dim) for _ in range(self.num_layers)]
        self.blocks = [PhaseAwareTransformerBlock(self.model_dim, self.num_heads, self.ff_dim)
                       for _ in range(self.num_layers)]
        self.output_proj = nn.Dense(1)

    def __call__(self, static_inputs):
        batch_size = static_inputs.shape[0]
        
        static_embed = self.static_encoder(static_inputs)
        
        phases = jnp.linspace(0, 1, self.sequence_length)
        phase_embed = self.phase_encoder(phases[None, :, None])
        phase_embed = jnp.repeat(phase_embed, batch_size, axis=0)
        
        x = jnp.zeros((batch_size, self.sequence_length, self.model_dim))
        
        for i in range(self.num_layers):
            gamma, beta = self.film_generators[i](static_embed)
            x = self.blocks[i](x, phase_embed)
            x = gamma[:, None, :] * x + beta[:, None, :]  # FiLM modulation
            
        return self.output_proj(x).squeeze(-1)

def spectral_loss(pred, target):
    pred_fft = jnp.fft.rfft(pred, axis=1)
    target_fft = jnp.fft.rfft(target, axis=1)
    return jnp.mean(jnp.abs(pred_fft - target_fft)**2)

def gradient_penalty(model, params, inputs):
    def loss_fn(inputs):
        pred = model.apply(params, inputs)
        return jnp.mean(pred**2)
    grads = jax.grad(loss_fn)(inputs)
    return jnp.mean(jnp.concatenate([jnp.ravel(g)**2 for g in jax.tree_leaves(grads)]))

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
    DATA_PATH = "../data/rsp.rrab.dat"
    BATCH_SIZE = 128
    NUM_EPOCHS = 1000
    SEED = 42
    
    train_data = {'X': X_train, 'y': y_train}
    test_data = {'X': X_test, 'y': y_test}

    # Initialize model and training state
    key = jax.random.PRNGKey(SEED)
    model = ImprovedTransformer(model_dim=512,ff_dim=1024)
    state = create_train_state(key, model, X_train)

    # Training loop
    best_test_loss = float('inf')
    for epoch in tqdm(range(NUM_EPOCHS)):
        key, subkey = jax.random.split(key)
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
        tqdm.write(f"Epoch {epoch+1}: Train Loss {np.mean(epoch_loss):.4f}, Test MSE {test_loss:.4f}")
    # save_model('new_arch_res', state)

if __name__ == "__main__":
    main()

