import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import jax.random as random
import optax

class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    ff_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Layer normalization
        x_norm = nn.LayerNorm()(x)
        # Multi-head self-attention
        attention_output = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.model_dim)(x_norm)
        # Add and norm
        x = x + attention_output
        # Layer normalization
        x_norm = nn.LayerNorm()(x)
        # Feed-forward network
        ff_output = nn.Dense(self.ff_dim)(x_norm)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dense(self.model_dim)(ff_output)
        # Add and norm
        return x + ff_output

class Transformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    ff_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Initial dense layer to project input to model_dim
        x = nn.Dense(self.model_dim)(x)
        # Apply transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(self.model_dim, self.num_heads, self.ff_dim)(x)
        # Layer normalization
        x = nn.LayerNorm()(x)
        # Output dense layer
        x = nn.Dense(self.output_dim)(x)
        return x

# Hyperparameters
num_layers = 4
model_dim = 128
num_heads = 8
ff_dim = 512
output_dim = 6
batch_size = 32
learning_rate = 0.001
num_epochs = 1000

# Initialize the model and optimizer
key1, key2 = random.split(random.PRNGKey(611))
model = Transformer(num_layers, model_dim, num_heads, ff_dim, output_dim)
params = model.init(key1, jnp.ones((batch_size, 5)))

# Define the optimizer
tx = optax.adam(learning_rate)

# Create the training state
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((batch_size, 5)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

state = create_train_state(random.PRNGKey(0), model, learning_rate)

# Define the MSE loss function
def mse_loss(predicted, target):
    return jnp.mean(jnp.square(predicted - target))

# Define a training step function
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = model.apply(params, batch['inputs'])
        loss = mse_loss(predictions, batch['targets'])
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

import pandas as pd
df = pd.read_csv('../data/rsp.rrab.dat', sep=r'\s+')
df['V_A2'] = df['V_R21']*df['V_A1']
df['V_A3'] = df['V_R31']*df['V_A1']
df['V_phi2'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P21']).values), 2*jnp.pi)
df['V_phi3'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P31']).values), 2*jnp.pi)

X = ['Z', 'X', 'M', 'L', 'Teff']
#y = ['Period']
y = ['V_A1', 'V_phi1', 'V_A2', 'V_phi2', 'V_A3', 'V_phi3']
# ['V_A3', 'V_phi3']
X = jnp.array(df[X])
y = jnp.array(df[y])
# scale inputs
X = (X - jnp.mean(X,axis=0))/jnp.std(X,axis=0)
y = (y - jnp.mean(y,axis=0))/jnp.std(y,axis=0)
data = jnp.concatenate([X, y], axis=1)
from utils import train_test_split
train, test = train_test_split(key1, data, 0.3, shuffle=True)
X_train, y_train = train[:,:X.shape[1]], train[:,X.shape[1]:]
X_test,  y_test  = test[:,:X.shape[1]], test[:,X.shape[1]:]

def get_batch():
    inputs = X_train
    targets = y_train
    return {'inputs': inputs, 'targets': targets}
def test_loss_fn(params):
        predictions = model.apply(params, X_test)
        loss = mse_loss(predictions, y_test)
        return loss
import tqdm
with tqdm.trange(num_epochs) as t:
    for epoch in t:
        state, loss = train_step(state, get_batch())
        test_loss = test_loss_fn(state.params)
        t.set_postfix_str("train: {:.4f}, test: {:.4f}".format(loss, test_loss), refresh=False) 

