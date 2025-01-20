import jax, jax.numpy as jnp
from flax import nnx
import optax


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
#    x = self.bn(x)
#    x = self.dropout(x)
    x = self.linear2(x)
    return x

    
def test():
    model = MLP(2, 16, 10, rngs=nnx.Rngs(0))
    x, y = jnp.ones((5, 2)), jnp.ones((5, 10))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))
    loss = train_step(model, optimizer, x, y)
    
    print(f'{loss = }')
    
    print(f'{optimizer.step.value = }')
