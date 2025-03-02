import jax, jax.numpy as jnp, jax.random as random
import new
from dataloader import *
from utils import *


jax.config.update("jax_default_matmul_precision", "high")
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
sequence_length=100
# fetch dataset
X, y = create_dataset(sequence_length=sequence_length)
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
print(X_test.shape)
model = new.ImprovedTransformer()
s = load_model('new_arch_res')
from functools import partial
body_fn = jax.jit(partial(model.apply, s['state']['params']))
from inference import infer_inputs
s = infer_inputs(y_test[100], body_fn, mu, sd, 0, std_y)
print(s.shape)
print(s.mean(axis=0))
print(X_test[100])
