import jax, jax.numpy as jnp, jax.random as random
import pandas as pd


def batch_generator(X: jax.Array, y: jax.Array, batch_size: int):
    """
    Util function for minibatch training with jax.
    Arguments:
        X: Features/inputs of shape (samples, features).
        y: Targets/outputs of shape (samples,features).
        batch_size: Size of minibatches.
    """
    num_batches = X.shape[0] // batch_size
    indices = jnp.arange(X.shape[0])
    for i in range(num_batches):
        batch_indices = indices[i*batch_size:(i+1)*batch_size]
        yield {'inputs': X[batch_indices], 'targets': y[batch_indices]}


def generate_sine_series(y, x):
    V_A1, V_phi1, V_A2, V_phi2, V_A3, V_phi3, Period = y
    t = x * Period
    omega = 2 * jnp.pi / Period
    sine_series = (V_A1 * jnp.sin(omega * t + V_phi1) + 
                   V_A2 * jnp.sin(omega * t + V_phi2) + 
                   V_A3 * jnp.sin(omega * t + V_phi3))
    return sine_series


def create_dataset(sequence_length=100):
    df = pd.read_csv('../data/rsp.rrab.dat', sep=r'\s+')
    df['V_A2'] = df['V_R21']*df['V_A1']
    df['V_A3'] = df['V_R31']*df['V_A1']
    df['V_phi2'] = jnp.mod(jnp.array((2*df['V_phi1'] + df['V_P21']).values), 2*jnp.pi)
    df['V_phi3'] = jnp.mod(jnp.array((3*df['V_phi1'] + df['V_P31']).values), 2*jnp.pi)
    X = ['Z', 'X', 'M', 'L', 'Teff']
    y = ['V_A1', 'V_phi1', 'V_A2', 'V_phi2', 'V_A3', 'V_phi3', 'Period']
    X = jnp.array(df[X])
    y = jnp.array(df[y])
    y = jnp.array([generate_sine_series(y, jnp.linspace(0,1,sequence_length)) for y in y])
    return X,y
