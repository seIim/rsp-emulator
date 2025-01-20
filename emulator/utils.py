import jax, jax.numpy as jnp
from flax import nnx

@nnx.jit
def train_step(model, optimizer, x, y):
  def loss_fn(model):
    y_pred = model(x)
    return jnp.sqrt(jnp.mean((y_pred - y) ** 2))

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)

  return loss

def train_test_split(key, data, split, shuffle=False):
    if shuffle:
        data = jax.random.permutation(key, data, axis=1)
    N = data.shape[0] 
    idx = int(N*split)
    train, test = data[idx:], data[:idx]
    return train,test

def main():
    x = jnp.arange(0, 100)
    key = jax.random.PRNGKey(611)
    train, test = train_test_split(key, x, 0.1)
    print(train, test)
