import jax, jax.numpy as jnp, jax.random as random
from flax.training import train_state
import matplotlib.pyplot as plt
from dataloader import *
import flax.linen as nn
import pandas as pd
from utils import *
import optax, tqdm


class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    ff_dim: int
    
    @nn.compact
    def __call__(self, x):
        # Self-attention with pre-LayerNorm
        x_norm = nn.LayerNorm()(x)
        attention_output = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            deterministic=False
        )(x_norm)
        x = x + attention_output
        
        # Feed-forward with pre-LayerNorm
        x_norm = nn.LayerNorm()(x)
        ff_output = nn.Dense(self.ff_dim)(x_norm)
        ff_output = nn.gelu(ff_output)  # Smoother activation
        ff_output = nn.Dense(self.model_dim)(ff_output)
        return x + ff_output

class Transformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    ff_dim: int
    output_dim: int
    sequence_length: int = 100  # Added sequence length parameter

    def setup(self):
        # Learnable positional encoding
        self.pos_encoding = self.param('pos_encoding',
                                      nn.initializers.normal(stddev=0.02),
                                      (1, self.sequence_length, self.model_dim))
        # Project static features to sequence space
        self.feature_proj = nn.Dense(self.model_dim)
        # Final output projection
        self.output_proj = nn.Dense(1)  # Predict single value per time step

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        
        # Project input features to model dimension
        x = self.feature_proj(x)  # (batch_size, model_dim)
        
        # Expand to sequence length and add positional encoding
        x = jnp.repeat(x[:, jnp.newaxis, :], self.sequence_length, axis=1)  # (batch, seq, dim)
        x = x + self.pos_encoding  # Broadcast positional encoding
        
        # Process through transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(self.model_dim, self.num_heads, self.ff_dim)(x)
        
        # Final projection to output dimension
        return self.output_proj(x).squeeze(-1)  # (batch_size, sequence_length)


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


def smoothness_loss(pred):
    # Penalize large differences between consecutive predictions
    diffs = pred[:, 1:] - pred[:, :-1]
    return jnp.mean(diffs**2)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = model.apply(params, batch['inputs'])
        loss = mse_loss(predictions, batch['targets']) + 0.1 * smoothness_loss(predictions)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def test_loss_fn(params):
    predictions = model.apply(params, X_test)
    loss = mse_loss(predictions, y_test) + smoothness_loss(predictions)
    return loss


# Hyperparameters
num_layers = 4
model_dim = 512
num_heads = 8
ff_dim = 1024 
batch_size = 1024
learning_rate = 0.005
num_epochs = 1000
sequence_length = 100

# fetch dataset
X, y = create_dataset(sequence_length=sequence_length)
key1, key2 = random.split(random.PRNGKey(42))
X_train, y_train, X_test, y_test = train_test_split(key1, X, y, split=0.2, shuffle=True)
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
X_train = (X_train - mu)/sd
X_test = (X_test - mu)/sd
input_dim  = X_train.shape[1]
output_dim = y_train.shape[1]

# init the model & train state
model = Transformer(
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

predictions = jnp.asarray(model.apply(state.params, X_train[:11]))
for i in range(10):
    plt.plot(jnp.linspace(0,1,sequence_length),predictions[i])
    plt.plot(jnp.linspace(0,1,sequence_length),y_train[i])
    plt.savefig(f'../figs/train_preds_{i}.pdf', bbox_inches='tight')
    plt.clf()
predictions = jnp.asarray(model.apply(state.params, X_test[:11]))
for i in range(10):
    plt.plot(jnp.linspace(0,1,sequence_length),predictions[i])
    plt.plot(jnp.linspace(0,1,sequence_length),y_test[i])
    plt.savefig(f'../figs/test_preds_{i}.pdf', bbox_inches='tight')
    plt.clf()
