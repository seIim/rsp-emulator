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
        x = self.linear1(x)
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
        attention_output = nn.SelfAttention(
                    num_heads=self.num_heads,
                    qkv_features=self.model_dim)(x_norm)
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
            x = TransformerBlock(self.model_dim,
                                 self.num_heads,
                                 self.ff_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class EmbeddingTransformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    ff_dim: int
    output_dim: int
    sequence_length: int = 100

    def setup(self):
        self.pos_encoding = self.param('pos_encoding',
                                       nn.initializers.normal(stddev=0.02),
                                       (1, self.sequence_length,
                                        self.model_dim))
        # Project static features to sequence space
        self.feature_proj = nn.Dense(self.model_dim)
        self.output_proj = nn.Dense(1)

    @nn.compact
    def __call__(self, x):
        x = self.feature_proj(x)
        x = jnp.repeat(x[:, jnp.newaxis, :], self.sequence_length, axis=1)
        x = x + self.pos_encoding
        for _ in range(self.num_layers):
            x = TransformerBlock(self.model_dim,
                                 self.num_heads,
                                 self.ff_dim)(x)
        return self.output_proj(x).squeeze(-1)  # (batch_size, sequence_length)


class FiLMGenerator(nn.Module):
    model_dim: int

    @nn.compact
    def __call__(self, x):
        gamma = nn.Dense(self.model_dim)(x)
        beta = nn.Dense(self.model_dim)(x)
        return gamma, beta


class EmbeddingTransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    ff_dim: int

    @nn.compact
    def __call__(self, x, phase_embed):
        x = jnp.concatenate([x, phase_embed], axis=-1)
        x = nn.Dense(self.model_dim)(x)

        attn = nn.SelfAttention(num_heads=self.num_heads,
                                qkv_features=self.model_dim)(x)
        x = x + attn

        x = x + nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.ff_dim),
            nn.gelu,
            nn.Dense(self.model_dim)
        ])(x)

        return x


class FiLMEmbeddingTransformer(nn.Module):
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
        self.blocks = [EmbeddingTransformerBlock(self.model_dim, self.num_heads, self.ff_dim)
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
