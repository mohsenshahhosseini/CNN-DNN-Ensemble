import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, train_test_split
import warnings
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from eli5.sklearn import PermutationImportance
from sklearn.metrics import r2_score
import joblib

warnings.filterwarnings('ignore')

test_year = 2019
scenario = '_final'

random_state = 1369
pd.set_option('display.max_columns', 500)
np.random.seed(random_state)

# loading yield file
print('\n $$ Loading data files... \n')
Yield = pd.read_csv('yield_data.csv')
Yield = Yield[['Year', 'State ANSI', 'County ANSI', 'State', 'County', 'Value']]
Yield.columns = ['year', 'state', 'county', 'State', 'County', 'Yield']
# remove rows with na
Yield.dropna(inplace=True)
Yield.county = Yield.county.astype('int')


# loading progress data
progress = pd.read_csv('progress_data.csv')
progress = progress[['Year', 'Period', 'State ANSI', 'Value']]
progress.columns = ['year', 'week', 'State', 'Progress']
# renaming the columns
for i in range(12, 30):
    progress.loc[progress.week == 'WEEK #'+str(i), 'week'] = i
# unmelting the dataframe
progress = progress.pivot(index=['year','State'], columns=['week'])
# flattening the column names
progress.columns = progress.columns.get_level_values(1)
# resetting index
progress = progress.reset_index()
# renaming the columns
progress.columns = ['year', 'state', *['Progress_'+str(i) for i in range(12,30)]]
# imputing na values
for i in progress.columns[2:11]:
    progress.loc[:,str(i)].fillna(0, inplace=True)
for i in progress.columns[11:]:
    progress.loc[:,str(i)].fillna(100, inplace=True)
# removing columns with all 100%
progress.drop(columns=['Progress_28','Progress_29'], inplace=True)
# adding rows for ND before the year 2000
listOfRows = []
for i in range(1980, 2000):
    listOfRows.append(pd.Series([i, 38, *[np.nan]*16], index=progress.columns))
progress = progress.append(listOfRows, ignore_index=True)
# imputing ND na values with average of two states close to ND (MI, SD)
for i in range(1980,2000):
    for p in range(12,28):
        progress.loc[(progress.year==i)&(progress.state==38),'Progress_'+str(p)] = \
            0.5*(progress.loc[(progress.year==i)&(progress.state==27),'Progress_'+str(p)].values +
                 progress.loc[(progress.year==i)&(progress.state==46),'Progress_'+str(p)].values)
# merging progress data with the yield
Yield = pd.merge(Yield, progress, on=['year','state'], how='left')


# loading soil file
soil = pd.read_csv('soil_data.csv')
keys = pd.read_csv('soil_data_key.csv')
soil = pd.concat([soil, keys[['State', 'County']]], axis=1)
soil['State'] = soil['State'].str.upper()
soil['County'] = soil['County'].str.upper()

# merging soil data with the yield
data1 = pd.merge(Yield, soil, on=['State', 'County'])
data1.drop(columns=['State','County'], inplace=True)

# loading weather data
weather = pd.read_parquet('main_weather_final.parquet')
weather = weather[(weather.year >= 1980)&(weather.year <= test_year)]
weather.state = weather.state.astype('int')
weather.county = weather.county.astype('int')
weather.year = weather.year.astype('int')

# Removing weather data after harvesting and before next planting date
idx = list(['state', 'county', 'year']) + \
      list(weather.loc[:,'prcp_16':'prcp_43'].columns) + \
      list(weather.loc[:,'tmax_16':'tmax_43'].columns) + \
      list(weather.loc[:,'tmin_16':'tmin_43'].columns) + \
      list(weather.loc[:,'srad_16':'srad_43'].columns) + \
      list(weather.loc[:,'gddf_16':'gddf_43'].columns)
weather = weather[idx]

# Constructing aggregated weather features
for f in ['srad', 'prcp', 'gddf']:
    weather[str(f)+'_Q2'] = weather.loc[:, str(f)+'_16':str(f)+'_26'].sum(axis=1)
    weather[str(f)+'_Q3'] = weather.loc[:, str(f)+'_27':str(f)+'_39'].sum(axis=1)
    weather[str(f)+'_Q4'] = weather.loc[:, str(f)+'_40':str(f)+'_43'].sum(axis=1)
    weather[str(f)+'_GS'] = weather.loc[:, str(f)+'_16':str(f)+'_43'].sum(axis=1)
for f in ['tmax', 'tmin']:
    weather[str(f)+'_Q2'] = weather.loc[:, str(f)+'_16':str(f)+'_26'].mean(axis=1)
    weather[str(f)+'_Q3'] = weather.loc[:, str(f)+'_27':str(f)+'_39'].mean(axis=1)
    weather[str(f)+'_Q4'] = weather.loc[:, str(f)+'_40':str(f)+'_43'].mean(axis=1)
    weather[str(f)+'_GS'] = weather.loc[:, str(f)+'_16':str(f)+'_43'].mean(axis=1)



##  -----------------  data preprocessing ----------------- ##

print('\n $$ Data preprocessing... \n')

# Feature construction (trend)
data1['yield_trend'] = 0
for s in data1.state.unique():
    for c in data1[data1.state==s].county.unique():
        y1 = pd.DataFrame(data1.Yield[(data1.year<test_year) & (data1.state == s) & (data1.county == c)])
        x1 = pd.DataFrame(data1.year[(data1.year<test_year) & (data1.state == s) & (data1.county == c)])
        regressor = LinearRegression()
        regressor.fit(x1, y1)
        data1.loc[(data1.year<test_year)&(data1.state==s)&(data1.county==c),'yield_trend'] = regressor.predict(x1)
        for i in [2019]:
            if len(data1.year[(data1.year==i)&(data1.state==s)&(data1.county==c)].unique()) != 0:
                data1.loc[(data1.year==i)&(data1.state==s)&(data1.county==c),'yield_trend'] = regressor.predict(pd.DataFrame([i]))
cols = data1.columns.tolist()
cols = cols[-1:]+cols[:-1]
data1 = data1[cols]

data = pd.merge(data1, weather , on=['year','state','county'])

# removing very small yields
data = data[data.Yield>10]
data = data.reset_index(drop=True)
data = data.rename(columns = {'year':'Year'})

for s in data.state.unique():
    print('State ' + Yield.State[Yield.state==s].unique()[0] + ' Average: ', data.Yield[(data.state==s) & (data.Year==2019)].mean())

# train and test splits
train = data[data.Year < test_year].reset_index(drop=True)
test = data[data.Year == test_year].reset_index(drop=True)

# Scaling the variables
columns_to_scale = data.drop(columns=['Yield','Year','state','county']).columns.values
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train[columns_to_scale])
scaled_train = pd.DataFrame(scaled_train, columns=columns_to_scale)
train = pd.concat([train.Yield, train.Year, scaled_train], axis=1)
scaled_test = scaler.transform(test[columns_to_scale])
scaled_test = pd.DataFrame(scaled_test, columns=columns_to_scale)
test = pd.concat([test.Yield, test.Year, scaled_test], axis=1)

# splits continued...
x_test = test.drop(columns=['Yield'])
y_test = test.Yield
x_training = train[(train.Year<test_year)].drop(columns=['Yield'])
y_training = train.loc[(train.Year<test_year),'Yield']
bins = np.linspace(start=0, stop=y_training.max(), num=5)
y_training_binned = np.digitize(y_training, bins, right=True)
X, x_valid, Y, y_valid = train_test_split(x_training, y_training, test_size=0.2,
                                          random_state=random_state, stratify=y_training_binned)

X.drop(columns='Year', inplace=True)
x_test.drop(columns='Year', inplace=True)
x_valid.drop(columns='Year', inplace=True)
x_training.drop(columns='Year', inplace=True)

X.reset_index(inplace=True, drop=True)
Y.reset_index(inplace=True, drop=True)
x_valid.reset_index(inplace=True, drop=True)
y_valid.reset_index(inplace=True, drop=True)



 ## ---------------- Bayesian Search ---------------- ##


max_evals = 10

def objective_LASSO(params):
    LASSO_df_B = pd.DataFrame()
    L1_B = Lasso()
    LASSO_B = L1_B.fit(X, Y)
    LASSO_df_B = pd.DataFrame(LASSO_B.predict(x_valid))
    loss_LASSO = mse(y_valid, LASSO_df_B)
    return {'loss': loss_LASSO, 'params': params, 'status': STATUS_OK}

space_LASSO = {'alpha': hp.uniform('alpha', 10**-5, 1)}
tpe_algorithm = tpe.suggest
trials_LASSO = Trials()
best_LASSO = fmin(fn=objective_LASSO, space=space_LASSO, algo=tpe.suggest,
                  max_evals=max_evals, trials=trials_LASSO, rstate=np.random.RandomState(random_state))
LASSO_param_B = pd.DataFrame({'alpha': []})
for i in range(max_evals):
    LASSO_param_B.alpha[i] = trials_LASSO.results[i]['params']['alpha']
LASSO_param_B = pd.DataFrame(LASSO_param_B.alpha)



def objective_XGB(params):
    XGB_df_B = pd.DataFrame()
    X1_B = XGBRegressor(objective='reg:squarederror', **params)
    XGB_B = X1_B.fit(X, Y)
    XGB_df_B = pd.DataFrame(XGB_B.predict(x_valid))
    loss_XGB = mse(y_valid, XGB_df_B)
    return {'loss': loss_XGB, 'params': params, 'status': STATUS_OK}

space_XGB = {'gamma': hp.uniform('gamma', 0, 1),
             'learning_rate': hp.uniform('learning_rate', 0.001, 0.5),
             'n_estimators': hp.choice('n_estimators', [100, 300, 500]),
             'max_depth': hp.choice('max_depth', [int(x) for x in np.arange(3, 10, 1)])}
tpe_algorithm = tpe.suggest
trials_XGB = Trials()
best_XGB = fmin(fn=objective_XGB, space=space_XGB, algo=tpe.suggest,
                max_evals=max_evals, trials=trials_XGB, rstate=np.random.RandomState(random_state))
XGB_param_B = pd.DataFrame({'gamma': [], 'learning_rate': [], 'n_estimators': [], 'max_depth': []})
for i in range(max_evals):
    XGB_param_B.gamma[i] = trials_XGB.results[i]['params']['gamma']
    XGB_param_B.learning_rate[i] = trials_XGB.results[i]['params']['learning_rate']
    XGB_param_B.n_estimators[i] = trials_XGB.results[i]['params']['n_estimators']
    XGB_param_B.max_depth[i] = trials_XGB.results[i]['params']['max_depth']
XGB_param_B = pd.DataFrame({'gamma': XGB_param_B.gamma,
                            'learning_rate': XGB_param_B.learning_rate,
                            'n_estimators': XGB_param_B.n_estimators,
                            'max_depth': XGB_param_B.max_depth})


def objective_LGB(params):
    LGB_df_B = pd.DataFrame()
    G1_B = LGBMRegressor(objective='regression', **params)
    LGB_B = G1_B.fit(X, Y)
    LGB_df_B = pd.DataFrame(LGB_B.predict(x_valid))
    loss_LGB = mse(y_valid, LGB_df_B)
    return {'loss': loss_LGB, 'params': params, 'status': STATUS_OK}

space_LGB = {'num_leaves': hp.choice('num_leaves', [int(x) for x in np.arange(5, 20, 2)]),
             'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
             'n_estimators': hp.choice('n_estimators', [100, 300, 500])}
tpe_algorithm = tpe.suggest
trials_LGB = Trials()
best_LGB = fmin(fn=objective_LGB, space=space_LGB, algo=tpe.suggest,
                max_evals=max_evals, trials=trials_LGB, rstate=np.random.RandomState(random_state))
LGB_param_B = pd.DataFrame({'num_leaves': [], 'learning_rate': [], 'n_estimators': []})
for i in range(max_evals):
    LGB_param_B.num_leaves[i] = trials_LGB.results[i]['params']['num_leaves']
    LGB_param_B.learning_rate[i] = trials_LGB.results[i]['params']['learning_rate']
    LGB_param_B.n_estimators[i] = trials_LGB.results[i]['params']['n_estimators']
LGB_param_B = pd.DataFrame({'num_leaves': LGB_param_B.num_leaves,
                            'learning_rate': LGB_param_B.learning_rate,
                            'n_estimators': LGB_param_B.n_estimators})


def objective_RF(params):
    RF_df_B = pd.DataFrame()
    R1_B = RandomForestRegressor(**params)
    RF_B = R1_B.fit(X, Y)
    RF_df_B = pd.DataFrame(RF_B.predict(x_valid))
    loss_RF = mse(y_valid, RF_df_B)
    return {'loss': loss_RF, 'params': params, 'status': STATUS_OK}

space_RF = {'n_estimators': hp.choice('n_estimators', [100, 300, 500]),
            'max_depth': hp.choice('max_depth', [int(x) for x in np.arange(5, 21, 5)])}
tpe_algorithm = tpe.suggest
trials_RF = Trials()
best_RF = fmin(fn=objective_RF, space=space_RF, algo=tpe.suggest,
               max_evals=max_evals, trials=trials_RF, rstate=np.random.RandomState(random_state))
RF_param_B = pd.DataFrame({'n_estimators': [], 'max_depth': []})
for i in range(max_evals):
    RF_param_B.n_estimators[i] = trials_RF.results[i]['params']['n_estimators']
    RF_param_B.max_depth[i] = trials_RF.results[i]['params']['max_depth']
RF_param_B = pd.DataFrame({'n_estimators': RF_param_B.n_estimators,
                           'max_depth': RF_param_B.max_depth})



## ---------------- Building models ---------------- ##
L2 = Lasso(alpha=trials_LASSO.best_trial['result']['params']['alpha'], random_state=random_state)
LASSO = L2.fit(X, Y)
LASSO_preds_test2 = LASSO.predict(x_test)
pd.DataFrame(LASSO_preds_test2).to_csv('LASSO_preds_test_' + str(test_year) + scenario+'.csv')
LASSO_mse_test2 = mse(y_test, LASSO_preds_test2)
LASSO_rmse_test2 = np.sqrt(LASSO_mse_test2)
LASSO_r2_test2 = r2_score(y_test, LASSO_preds_test2)
LASSO_preds_train = LASSO.predict(X)
pd.DataFrame(LASSO_preds_train).to_csv('LASSO_preds_train_' + str(test_year) + scenario+'.csv')
LASSO_rmse_train = np.sqrt(mse(Y, LASSO_preds_train))
LASSO_r2_train = r2_score(Y, LASSO_preds_train)
LASSO_df2 = pd.DataFrame(L2.predict(x_valid))
LASSO_df2 = LASSO_df2.reset_index(drop=True)
LASSO_mse2 = mse(y_valid, LASSO_df2)


### ---------- XGB ------------ ###
X2 = XGBRegressor(objective='reg:squarederror',
                  gamma=trials_XGB.best_trial['result']['params']['gamma'],
                  learning_rate=trials_XGB.best_trial['result']['params']['learning_rate'],
                  n_estimators=int(trials_XGB.best_trial['result']['params']['n_estimators']),
                  max_depth=int(trials_XGB.best_trial['result']['params']['max_depth']), random_state=random_state)
XGB = X2.fit(X, Y)
XGB_preds_test2 = XGB.predict(x_test)
pd.DataFrame(XGB_preds_test2).to_csv('XGB_preds_test_' + str(test_year) + scenario+'.csv')
XGB_mse_test2 = mse(y_test, XGB_preds_test2)
XGB_rmse_test2 = np.sqrt(XGB_mse_test2)
XGB_r2_test2 = r2_score(y_test, XGB_preds_test2)
joblib.dump(XGB, 'XGB_saved_'+str(test_year)+scenario+'.sav')
XGB_preds_train = XGB.predict(X)
pd.DataFrame(XGB_preds_train).to_csv('XGB_preds_train_' + str(test_year) + scenario+'.csv')
XGB_rmse_train = np.sqrt(mse(Y, XGB_preds_train))
XGB_r2_train = r2_score(Y, XGB_preds_train)
XGB_df2 = pd.DataFrame(X2.predict(x_valid))
XGB_df2 = XGB_df2.reset_index(drop=True)
XGB_mse2 = mse(y_valid, XGB_df2)


### ---------- LGB ------------ ###
G2 = LGBMRegressor(objective='regression', random_state=random_state,
                   num_leaves=int(trials_LGB.best_trial['result']['params']['num_leaves']),
                   learning_rate=trials_LGB.best_trial['result']['params']['learning_rate'],
                   n_estimators=int(trials_LGB.best_trial['result']['params']['n_estimators']))
LGB = G2.fit(X, Y)
LGB_preds_test2 = LGB.predict(x_test)
pd.DataFrame(LGB_preds_test2).to_csv('LGB_preds_test_' + str(test_year) + scenario+'.csv')
LGB_mse_test2 = mse(y_test, LGB_preds_test2)
LGB_rmse_test2 = np.sqrt(LGB_mse_test2)
LGB_r2_test2 = r2_score(y_test, LGB_preds_test2)
joblib.dump(LGB, 'LGB_saved_'+str(test_year)+scenario+'.sav')
LGB_preds_train = LGB.predict(X)
pd.DataFrame(LGB_preds_train).to_csv('LGB_preds_train_' + str(test_year) + scenario+'.csv')
LGB_rmse_train = np.sqrt(mse(Y, LGB_preds_train))
LGB_r2_train = r2_score(Y, LGB_preds_train)
LGB_df2 = pd.DataFrame(G2.predict(x_valid))
LGB_df2 = LGB_df2.reset_index(drop=True)
LGB_mse2 = mse(y_valid, LGB_df2)


### ---------- RF ------------ ###
R2 = RandomForestRegressor(max_depth=int(trials_RF.best_trial['result']['params']['max_depth']),
                           n_estimators=int(trials_RF.best_trial['result']['params']['n_estimators']), random_state=random_state)
RF = R2.fit(X, Y)
RF_preds_test2 = RF.predict(x_test)
pd.DataFrame(RF_preds_test2).to_csv('RF_preds_test_' + str(test_year) + scenario+'.csv')
RF_mse_test2 = mse(y_test, RF_preds_test2)
RF_rmse_test2 = np.sqrt(RF_mse_test2)
RF_r2_test2 = r2_score(y_test, RF_preds_test2)
joblib.dump(RF, 'RF_saved_'+str(test_year)+scenario+'.sav')
RF_preds_train = RF.predict(X)
pd.DataFrame(RF_preds_train).to_csv('RF_preds_train_' + str(test_year) + scenario+'.csv')
RF_rmse_train = np.sqrt(mse(Y, RF_preds_train))
RF_r2_train = r2_score(Y, RF_preds_train)
RF_df2 = pd.DataFrame(R2.predict(x_valid))
RF_df2 = RF_df2.reset_index(drop=True)
RF_mse2 = mse(y_valid, RF_df2)


### ---------- LR ------------ ###
lm2 = LinearRegression()
LR = lm2.fit(X, Y)
LR_preds_test2 = LR.predict(x_test)
pd.DataFrame(LR_preds_test2).to_csv('LR_preds_test_' + str(test_year) + scenario+'.csv')
LR_mse_test2 = mse(y_test, LR_preds_test2)
LR_rmse_test2 = np.sqrt(LR_mse_test2)
LR_r2_test2 = r2_score(y_test, LR_preds_test2)
joblib.dump(LR, 'LR_saved_'+str(test_year)+scenario+'.sav')
LR_preds_train = LR.predict(X)
pd.DataFrame(LR_preds_train).to_csv('LR_preds_train_' + str(test_year) + scenario+'.csv')
LR_rmse_train = np.sqrt(mse(Y, LR_preds_train))
LR_r2_train = r2_score(Y, LR_preds_train)
LR_df2 = pd.DataFrame(LR.predict(x_valid))
LR_df2 = LR_df2.reset_index(drop=True)
LR_mse2 = mse(y_valid, LR_df2)


## ---------------- Optimizing Ensembles ---------------- ##

def objective2(y):
    return mse(y_valid,
               (y[0]*LASSO_df2 + y[1]*XGB_df2 + y[2]*LGB_df2 + y[3]*RF_df2 + y[4]*LR_df2))

def constraint12(y):
    return y[0] + y[1] + y[2] + y[3] + y[4] - 1.0
def constraint22(y):
    return LASSO_mse2 - objective2(y)
def constraint32(y):
    return XGB_mse2 - objective2(y)
def constraint42(y):
    return LGB_mse2 - objective2(y)
def constraint52(y):
    return RF_mse2 - objective2(y)
def constraint62(y):
    return LR_mse2 - objective2(y)


y0 = np.zeros(5)
y0[0] = 1 / 5
y0[1] = 1 / 5
y0[2] = 1 / 5
y0[3] = 1 / 5
y0[4] = 1 / 5

b = (0, 1.0)
bnds2 = (b, b, b, b, b)
con12 = {'type': 'eq', 'fun': constraint12}
con22 = {'type': 'ineq', 'fun': constraint22}
con32 = {'type': 'ineq', 'fun': constraint32}
con42 = {'type': 'ineq', 'fun': constraint42}
con52 = {'type': 'ineq', 'fun': constraint52}
con62 = {'type': 'ineq', 'fun': constraint62}

cons2 = [con12, con22, con32, con42, con52, con62]

solution2 = minimize(objective2, y0, method='SLSQP',
                    options={'disp': True, 'maxiter': 3000, 'eps': 1e-3}, bounds=bnds2,
                    constraints=cons2)
y = solution2.x
pd.DataFrame(y).to_csv('others_opt_weights_'+str(test_year)+scenario+'.csv')

cowe_preds_test = y[0]*LASSO_preds_test2 + y[1]*XGB_preds_test2 + y[2]*LGB_preds_test2 + y[3]*RF_preds_test2 + y[4]*LR_preds_test2
cowe_mse_test = mse(y_test, cowe_preds_test)
cowe_rmse_test = np.sqrt(cowe_mse_test)
cowe_r2_test = r2_score(y_test, cowe_preds_test)
pd.DataFrame(cowe_preds_test).to_csv('cowe_preds_test_' + str(test_year) + scenario+'.csv')

cls_preds_test = y0[0]*LASSO_preds_test2 + y0[1]*XGB_preds_test2 + y0[2]*LGB_preds_test2 + y0[3]*RF_preds_test2 + y0[4]*LR_preds_test2
cls_mse_test = mse(y_test, cls_preds_test)
cls_rmse_test = np.sqrt(cls_mse_test)
cls_r2_test = r2_score(y_test, cls_preds_test)
pd.DataFrame(cls_preds_test).to_csv('cls_preds_test_' + str(test_year) + scenario+'.csv')



## -------------------------------- STACKING -------------------------------- ##

predsDF2 = pd.DataFrame()
predsDF2['LASSO'] = LASSO_df2[0]
predsDF2['XGB']= XGB_df2[0]
predsDF2['LGB'] = LGB_df2[0]
predsDF2['RF'] = RF_df2[0]
predsDF2['LR'] = LR_df2[0]
predsDF2['Y'] = y_valid.reset_index(drop=True)
x_stacked2 = predsDF2.drop(columns='Y', axis=1)
y_stacked2 = predsDF2['Y']
testPreds2 = pd.DataFrame([LASSO_preds_test2, XGB_preds_test2, LGB_preds_test2, RF_preds_test2, LR_preds_test2]).T
testPreds2.columns = ['LASSO', 'XGB', 'LGB', 'RF', 'LR']


stck_reg2 = LinearRegression()
stck_reg2.fit(x_stacked2, y_stacked2)
stck_reg_preds_test2 = stck_reg2.predict(testPreds2)
stck_reg_mse_test2 = mse(y_test, stck_reg_preds_test2)
stck_reg_rmse_test2 = np.sqrt(stck_reg_mse_test2)
stck_reg_r2_test2 = r2_score(y_test, stck_reg_preds_test2)
pd.DataFrame(stck_reg_preds_test2).to_csv('stck_reg_preds_test_' + str(test_year) + scenario+'.csv')
joblib.dump(stck_reg2, 'others_stckreg_saved_'+str(test_year)+scenario+'.sav')

stck_lasso2 = Lasso()
stck_lasso2.fit(x_stacked2, y_stacked2)
stck_lasso_preds_test2 = stck_lasso2.predict(testPreds2)
stck_lasso_mse_test2 = mse(y_test, stck_lasso_preds_test2)
stck_lasso_rmse_test2 = np.sqrt(stck_lasso_mse_test2)
stck_lasso_r2_test2 = r2_score(y_test, stck_lasso_preds_test2)
pd.DataFrame(stck_lasso_preds_test2).to_csv('stck_lasso_preds_test_' + str(test_year) + scenario+'.csv')
joblib.dump(stck_lasso2, 'others_stcklasso_saved_'+str(test_year)+scenario+'.sav')

stck_rf2 = RandomForestRegressor()
stck_rf2.fit(x_stacked2, y_stacked2)
stck_rf_preds_test2 = stck_rf2.predict(testPreds2)
stck_rf_mse_test2 = mse(y_test, stck_rf_preds_test2)
stck_rf_rmse_test2 = np.sqrt(stck_rf_mse_test2)
stck_rf_r2_test2 = r2_score(y_test, stck_rf_preds_test2)
pd.DataFrame(stck_rf_preds_test2).to_csv('stck_rf_preds_test_' + str(test_year) + scenario+'.csv')
joblib.dump(stck_rf2, 'others_stckrf_saved_'+str(test_year)+scenario+'.sav')

stck_lgb2 = LGBMRegressor()
stck_lgb2.fit(x_stacked2, y_stacked2)
stck_lgb_preds_test2 = stck_lgb2.predict(testPreds2)
stck_lgb_mse_test2 = mse(y_test, stck_lgb_preds_test2)
stck_lgb_rmse_test2 = np.sqrt(stck_lgb_mse_test2)
stck_lgb_r2_test2 = r2_score(y_test, stck_lgb_preds_test2)
pd.DataFrame(stck_lgb_preds_test2).to_csv('stck_lgb_preds_test_' + str(test_year) + scenario+'.csv')
joblib.dump(stck_lgb2, 'others_stcklgb_saved_'+str(test_year)+scenario+'.sav')



## -------------------------- RESULTS -------------------------- ##


test_results = pd.DataFrame(data={'model':['RMSE'],'LASSO':[LASSO_rmse_test2], 'XGB':[XGB_rmse_test2], 'LGB':[LGB_rmse_test2],
                                  'RF': [RF_rmse_test2], 'LR': [LR_rmse_test2],
                                  'COWE': [cowe_rmse_test], 'Classical': [cls_rmse_test],
                                  'stck_reg': [stck_reg_rmse_test2], 'stck_lasso': [stck_lasso_rmse_test2],
                                  'stck_rf': [stck_rf_rmse_test2], 'stck_lgb': [stck_lgb_rmse_test2]})

train_results = pd.DataFrame(data={'model':['RMSE'],'LASSO':[LASSO_rmse_train], 'XGB':[XGB_rmse_train],
                                   'LGB':[LGB_rmse_train], 'RF': [RF_rmse_train], 'LR': [LR_rmse_train]})

test_r2 = pd.DataFrame(data={'model':['R2'],'LASSO':[LASSO_r2_test2], 'XGB':[XGB_r2_test2], 'LGB':[LGB_r2_test2],
                             'RF': [RF_r2_test2], 'LR': [LR_r2_test2],
                             'COWE': [cowe_r2_test], 'Classical': [cls_r2_test],
                             'stck_reg': [stck_reg_r2_test2], 'stck_lasso': [stck_lasso_r2_test2],
                             'stck_rf': [stck_rf_r2_test2], 'stck_lgb': [stck_lgb_r2_test2]})

train_r2 = pd.DataFrame(data={'model':['R2'],'LASSO':[LASSO_r2_train], 'XGB':[XGB_r2_train],
                              'LGB':[LGB_r2_train], 'RF': [RF_r2_train], 'LR': [LR_r2_train]})

test_results.to_csv('test_' + str(test_year) + scenario+'.csv')
train_results.to_csv('train_' + str(test_year) + scenario+'.csv')
test_r2.to_csv('R2_test_' + str(test_year) + scenario+'.csv')
train_r2.to_csv('R2_train_' + str(test_year) + scenario+'.csv')
