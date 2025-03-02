import jax
import jax.numpy as jnp
import jax.random as random
from flax.training import train_state
from functools import partial
import optax
import tqdm
import sys
import os
project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# local imports
from emulator.layers import EmbeddingTransformer
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
    loss = mse_loss(predictions, y_test) + smoothness_loss(predictions)
    return loss


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
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = model.apply(params, batch['inputs'])
        smoothing = 0.3 * smoothness_loss(predictions)
        loss = mse_loss(predictions, batch['targets']) + smoothing
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Hyperparameters
num_layers = 1
model_dim = 512
num_heads = 8
ff_dim = 1024
batch_size = 1024
learning_rate = 1e-3
num_epochs = 2000
sequence_length = 100

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
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# init the model & train state
model = EmbeddingTransformer(
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
        t.set_postfix_str(f"train: {jnp.mean(jnp.array(epoch_loss)):.4f}, \
                          test: {test_loss:.4f}", refresh=False)

hyperparams = {'num_layers': num_layers,
               'model_dim': model_dim,
               'num_heads': num_heads,
               'ff_dim': ff_dim,
               'output_dim': output_dim
               }

save_model('embedding_transformer', state)
s = load_model('ckpt')
body_fn = jax.jit(partial(model.apply, state.params))
#from inference import infer_inputs
#s = infer_inputs(y_test[0], body_fn, num_warmup=2000, num_samples=500)
#print(s.mean(axis=0))
#print(s.std(axis=0))
#print(X_test[0])
#from utils import save_params
#import os
#save_dir = os.path.abspath("./checkpoints")
#save_params(state, save_dir)

#predictions = jnp.asarray(model.apply(state.params, X_train[:11]))*std_y
#for i in range(10):
#    plt.plot(jnp.linspace(0,1,sequence_length),predictions[i])
#    plt.plot(jnp.linspace(0,1,sequence_length),y_train[i]*std_y)
#    plt.savefig(f'../figs/train_preds_{i}.pdf', bbox_inches='tight')
#    plt.clf()
#predictions = jnp.asarray(model.apply(state.params, X_test[:11]))*std_y
#for i in range(10):
#    plt.plot(jnp.linspace(0,1,sequence_length),predictions[i])
#    plt.plot(jnp.linspace(0,1,sequence_length),y_test[i]*std_y)
#    plt.savefig(f'../figs/test_preds_{i}.pdf', bbox_inches='tight')
#    plt.clf()
