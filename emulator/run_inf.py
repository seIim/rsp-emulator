import jax, jax.numpy as jnp, jax.random as random
from models import EmbeddingTransformer
from dataloader import *
from utils import *


# Hyperparameters
num_layers = 4
model_dim = 512
num_heads = 8
ff_dim = 1024 
batch_size = 256
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
input_dim  = X_train.shape[1]
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

hyperparams = {'num_layers': num_layers,
                'model_dim': model_dim,
                'num_heads': num_heads,
                'ff_dim': ff_dim, 
                'output_dim': output_dim
                }

s = load_model('fin_a100')
from functools import partial
body_fn = jax.jit(partial(model.apply, s['state']['params']))
from inference import infer_inputs
s = infer_inputs(y_test[0], body_fn, num_warmup=2000, num_samples=500)
print(s.mean(axis=0))
print(s.std(axis=0))
print(X_test[0])

