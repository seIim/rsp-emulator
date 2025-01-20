from xgboost import XGBRegressor
import xgboost
import numpy as jnp
import pandas as pd

df = pd.read_csv('../data/rsp.rrab.dat', sep=r'\s+')
df['V_A2'] = df['V_R21']*df['V_A1']
df['V_A3'] = df['V_R31']*df['V_A1']
df['V_phi2'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P21']).values), 2*jnp.pi)
df['V_phi3'] = jnp.mod(jnp.array((df['V_phi1']*df['V_P31']).values), 2*jnp.pi)

X = ['Z', 'X', 'M', 'L', 'Teff']
y = ['V_A1', 'V_phi1', 'V_A2', 'V_phi2', 'V_A3', 'V_phi3']

X = jnp.array(df[X])
y = jnp.array(df[y])
# scale inputs
X = (X - jnp.mean(X,axis=0))/jnp.std(X,axis=0)
y = (y - jnp.mean(y,axis=0))/jnp.std(y,axis=0)
from sklearn.model_selection import train_test_split
xgb_params = {'lambda': 0.24624952377064363, 'alpha': 0.007745847934609679, 
              'colsample_bytree': 0.6, 'subsample': 0.6, 'learning_rate': 0.016, 
              'max_depth': 13, 'random_state': 611, 'min_child_weight': 1,
              'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)
xgb = XGBRegressor(n_estimators=10000)
xgb.fit(train_X, train_y, verbose=100, eval_set=[(test_X, test_y)])

