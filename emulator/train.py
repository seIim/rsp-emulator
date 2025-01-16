import jax.numpy as jnp
import pandas as pd
from flax import nnx
import optax
import model

df = pd.read_csv('../data/rsp.rrab.dat', sep=r'\s+')
df['V_A2'] = df['V_R21']*df['V_A1']
df['V_A3'] = df['V_R31']*df['V_A1']
df['V_phi2'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P21']).values), 2*jnp.pi)
df['V_phi3'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P31']).values), 2*jnp.pi)

X = ['Z', 'X', 'M', 'L', 'Teff']
y = ['V_A1', 'V_phi1', 'V_A2', 'V_phi2', 'V_A3', 'V_phi3']

X = jnp.array(df[X])
y = jnp.array(df[y])
X = (X - jnp.mean(X,axis=0))/jnp.std(X,axis=0)
y = (y - jnp.mean(y,axis=0))/jnp.std(y,axis=0)
nn = model.MLP(din=X.shape[1], dout=y.shape[1], dmid=100, rngs=nnx.Rngs(42))
optimizer = nnx.Optimizer(nn, optax.adam(1e-3))
#loss = model.train_step(nn, optimizer=optimizer, x=X, y=y)
import tqdm
loss = 0
for _ in tqdm.tqdm(range(10_000)):
    loss = model.train_step(nn, optimizer=optimizer, x=X, y=y)

print(loss)
