import jax
import jax.numpy as jnp
import jax.random as random
from flax.training import train_state
import optax
import tqdm
import sys
import os
project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# local imports
from emulator.layers import Transformer
from emulator.dataloader import *
from emulator.utils import *

# fix precision loss on a100
jax.config.update("jax_default_matmul_precision", "high")
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

@jax.jit
def test_loss_fn(params):
    predictions = model.apply(params, X_test)
    loss = mse_loss(predictions, y_test)
    return loss


def create_train_state(rng, model, learning_rate, input_dim):
    params = model.init(rng, jnp.ones((batch_size, input_dim)))

    tx = optax.chain(
       optax.adaptive_grad_clip(1.0),
       optax.adamw(learning_rate)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def smoothness_loss(pred):
    # Penalize large differences between consecutive predictions
    diffs = pred[:, 1:] - pred[:, :-1]
    return jnp.mean(diffs**2)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = model.apply(params, batch['inputs'])
        loss = mse_loss(predictions, batch['targets']) + 0.3 * smoothness_loss(predictions)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


X, y = create_dataset()
key1, key2 = random.split(random.PRNGKey(611))
X_train, y_train, X_test, y_test = train_test_split(key1, X, y, split=0.2, shuffle=True)
input_dim  = X_train.shape[1] # features
output_dim = y_train.shape[1]

# Hyperparameters
num_layers = 4
model_dim = 512
num_heads = 8
ff_dim = 512
batch_size = 512
learning_rate = 0.001
num_epochs = 100
sequence_length = 100

model = Transformer(num_layers, model_dim, num_heads, ff_dim, output_dim)
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


save_model('default_transformer', state)
#s = load_model('ckpt')
#from functools import partial
#body_fn = jax.jit(partial(model.apply, state.params))
#from inference import infer_inputs
#s = infer_inputs(y_test[10], body_fn)
#print(s.shape)
