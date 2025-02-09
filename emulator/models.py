import jax.numpy as jnp
import flax.linen as nn
from flax import nnx
import jax


class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.3, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  @nnx.jit
  def __call__(self, x: jax.Array):
    x =  self.linear1(x)
    x = nnx.leaky_relu(x)
    x = self.linear2(x)
    return x

class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    ff_dim: int
    
    @nn.compact
    def __call__(self, x):
        x_norm = nn.LayerNorm()(x)
        attention_output = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.model_dim)(x_norm)
        x = x + attention_output
        x_norm = nn.LayerNorm()(x)
        ff_output = nn.Dense(self.ff_dim)(x_norm)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dense(self.model_dim)(ff_output)
        return x + ff_output

class Transformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    ff_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.model_dim)(x)
        for _ in range(self.num_layers):
            x = TransformerBlock(self.model_dim, self.num_heads, self.ff_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.output_dim)(x)
        return x



class EmbeddingTransformer(nn.Module):
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
