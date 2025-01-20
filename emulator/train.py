import jax.numpy as jnp
import pandas as pd
from flax import nnx
import optax
import models
from utils import *

df = pd.read_csv('../data/rsp.rrab.dat', sep=r'\s+')
df['V_A2'] = df['V_R21']*df['V_A1']
df['V_A3'] = df['V_R31']*df['V_A1']
df['V_phi2'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P21']).values), 2*jnp.pi)
df['V_phi3'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P31']).values), 2*jnp.pi)

X = ['Z', 'X', 'M', 'L', 'Teff']
#y = ['Period']
y = ['V_A1', 'V_phi1', 'V_A2', 'V_phi2', 'Period']
# ['V_A3', 'V_phi3']
X = jnp.array(df[X])
y = jnp.array(df[y])
# scale inputs
X = (X - jnp.mean(X,axis=0))/jnp.std(X,axis=0)
y = (y - jnp.mean(y,axis=0))/jnp.std(y,axis=0)
nn = models.MLP(din=X.shape[1], dout=y.shape[1], dmid=1, rngs=nnx.Rngs(611))
opt = optax.chain(
#       optax.adaptive_grad_clip(1.0),
       optax.adam(1.7e-3)
)
optimizer = nnx.Optimizer(nn, opt)
import tqdm
test_loss = 0
loss = 0

@nnx.jit
def test_loss_fn(model, test_x, test_y):
   y_pred = model(test_x)
   return jnp.mean((y_pred - test_y) ** 2)

key = jax.random.PRNGKey(611)
data = jnp.concatenate([X, y], axis=1)
train, test = train_test_split(key, data, 0.3, shuffle=True)
X_train, y_train = train[:,:X.shape[1]], train[:,X.shape[1]:]
X_test,  y_test  = test[:,:X.shape[1]], test[:,X.shape[1]:]
with tqdm.trange(1, 10_000 + 1) as t:
    for i in t:
        loss = train_step(nn, optimizer=optimizer, x=X_train, y=y_train)
        test_loss = test_loss_fn(nn, X_test, y_test)
        t.set_postfix_str("train: {:.4f}, test: {:.4f}".format(loss, test_loss), refresh=False) 

