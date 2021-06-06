import numpy as np
import pandas as pd
import random
from random import randrange
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, train_test_split
import warnings
from keras.models import *
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
import joblib


warnings.filterwarnings('ignore')

test_year = 2019
scenario = '_final_fixedscaling'

random_state = 1369
pd.set_option('display.max_columns', 500)
np.random.seed(random_state)

# loading yield file
print('\n $$ Loading data files... \n')
# os.chdir("/Users/mohsen/Box/8. Fall 20/Ensemble Deep Learning/data/")
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

# aggregating soil features to 3 levels
# for v in range(1,11):
#     soil['S_var'+str(v)+'_0:45'] = soil['S_var' + str(v) + '_depth1'] * 5/45 + \
#                                    soil['S_var' + str(v) + '_depth2'] * 5/45 + \
#                                    soil['S_var' + str(v) + '_depth3'] * 5/45 + \
#                                    soil['S_var' + str(v) + '_depth4'] * 15/45 + \
#                                    soil['S_var' + str(v) + '_depth5'] * 15/45
#     soil['S_var'+str(v)+'_45:80'] = soil['S_var' + str(v) + '_depth6'] * 15/35 + \
#                                     soil['S_var' + str(v) + '_depth7'] * 20/35
#     soil['S_var'+str(v)+'_80:150'] = soil['S_var' + str(v) + '_depth8'] * 20/70 + \
#                                      soil['S_var' + str(v) + '_depth9'] * 20/70 + \
#                                      soil['S_var' + str(v) + '_depth10'] * 30/70
# soil.drop(soil.loc[:,'S_var1_depth1':'S_var10_depth10'], axis=1, inplace=True)
keys = pd.read_csv('soil_data_key.csv')
soil = pd.concat([soil, keys[['State', 'County']]], axis=1)
soil['State'] = soil['State'].str.upper()
soil['County'] = soil['County'].str.upper()

# merging soil data with the yield
data1 = pd.merge(Yield, soil, on=['State', 'County'])
data1.drop(columns=['State','County'], inplace=True)

# loading weather data
# os.chdir('/Users/mohsen/Box/8. Fall 20/Ensemble Deep Learning/weather')
weather = pd.read_parquet('main_weather_final.parquet')
weather = weather[(weather.year >= 1980)&(weather.year <= test_year)]
weather.state = weather.state.astype('int')
weather.county = weather.county.astype('int')
weather.year = weather.year.astype('int')
# weather.drop(weather.loc[:,'gddf_1':'gddf_52'], axis=1, inplace=True)

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



## -------------------------- Ensemble DNN -------------------------- ##


def CNN_W(input_train, kernel_size, stride_conv, stride_pool, filter1, filter2, filter3, pool_size):
    # expanding input dimensions
    X = np.expand_dims(input_train, axis=2)
    # define inputs
    input = Input(shape=(X.shape[1], 1))
    x = Conv1D(kernel_size=2*kernel_size, filters=filter1, strides=stride_conv, activation='relu')(input)
    x = AveragePooling1D(pool_size=pool_size, strides=stride_pool)(x)
    x = Conv1D(kernel_size=kernel_size, filters=filter2, strides=stride_conv, activation='relu')(x)
    x = AveragePooling1D(pool_size=pool_size, strides=stride_pool)(x)
    x = Conv1D(kernel_size=kernel_size, filters=filter3, strides=stride_conv, activation='relu')(x)
    x = AveragePooling1D(pool_size=pool_size, strides=stride_pool)(x)
    x = Flatten()(x)
    x = Model(inputs=input, outputs=x)
    return X, input, x.output


def CNN_S(input_train, kernel_size, stride_conv, stride_pool, filter1, filter2, filter3, pool_size):
    # expanding input dimensions
    X = np.expand_dims(input_train, axis=2)
    # define inputs
    input = Input(shape=(X.shape[1], 1))
    x = Conv1D(kernel_size=kernel_size, filters=filter1, strides=stride_conv, activation='relu')(input)
    x = AveragePooling1D(pool_size=pool_size, strides=stride_pool)(x)
    x = Conv1D(kernel_size=kernel_size, filters=filter2, strides=stride_conv, activation='relu')(x)
    x = AveragePooling1D(pool_size=pool_size, strides=stride_pool)(x)
    x = Conv1D(kernel_size=1, filters=filter3, strides=stride_conv, activation='relu')(x)
    x = Flatten()(x)
    x = Model(inputs=input, outputs=x)
    return X, input, x.output


def CNN_DNN(input_train, input_test, kernel_size, stride_conv, stride_pool,
            filter1, filter2, filter3, pool_size,
            FC1_l1, FC1_l2, FC1_l3, FC2_l1, FC2_l2, FC2_l3, w_fc_l, s_fc_l,
            dropout, lr, epochs, batch_size, y_train):

    X_w_prcp = input_train.iloc[:, input_train.columns.str.startswith('prcp')]
    X_w_tmax = input_train.iloc[:, input_train.columns.str.startswith('tmax')]
    X_w_tmin = input_train.iloc[:, input_train.columns.str.startswith('tmin')]
    X_w_srad = input_train.iloc[:, input_train.columns.str.startswith('srad')]
    X_w_gddf = input_train.iloc[:, input_train.columns.str.startswith('gddf')]
    X_s_var1 = input_train.iloc[:, input_train.columns.str.startswith('S_var1_')]
    X_s_var2 = input_train.iloc[:, input_train.columns.str.startswith('S_var2_')]
    X_s_var3 = input_train.iloc[:, input_train.columns.str.startswith('S_var3_')]
    X_s_var4 = input_train.iloc[:, input_train.columns.str.startswith('S_var4_')]
    X_s_var5 = input_train.iloc[:, input_train.columns.str.startswith('S_var5_')]
    X_s_var6 = input_train.iloc[:, input_train.columns.str.startswith('S_var6_')]
    X_s_var7 = input_train.iloc[:, input_train.columns.str.startswith('S_var7_')]
    X_s_var8 = input_train.iloc[:, input_train.columns.str.startswith('S_var8_')]
    X_s_var9 = input_train.iloc[:, input_train.columns.str.startswith('S_var9_')]
    X_s_var10 = input_train.iloc[:, input_train.columns.str.startswith('S_var10_')]

    X_test_w_prcp = input_test.iloc[:, input_test.columns.str.startswith('prcp')]
    X_test_w_tmax = input_test.iloc[:, input_test.columns.str.startswith('tmax')]
    X_test_w_tmin = input_test.iloc[:, input_test.columns.str.startswith('tmin')]
    X_test_w_srad = input_test.iloc[:, input_test.columns.str.startswith('srad')]
    X_test_w_gddf = input_test.iloc[:, input_test.columns.str.startswith('gddf')]
    X_test_s_var1 = input_test.iloc[:, input_test.columns.str.startswith('S_var1_')]
    X_test_s_var2 = input_test.iloc[:, input_test.columns.str.startswith('S_var2_')]
    X_test_s_var3 = input_test.iloc[:, input_test.columns.str.startswith('S_var3_')]
    X_test_s_var4 = input_test.iloc[:, input_test.columns.str.startswith('S_var4_')]
    X_test_s_var5 = input_test.iloc[:, input_test.columns.str.startswith('S_var5_')]
    X_test_s_var6 = input_test.iloc[:, input_test.columns.str.startswith('S_var6_')]
    X_test_s_var7 = input_test.iloc[:, input_test.columns.str.startswith('S_var7_')]
    X_test_s_var8 = input_test.iloc[:, input_test.columns.str.startswith('S_var8_')]
    X_test_s_var9 = input_test.iloc[:, input_test.columns.str.startswith('S_var9_')]
    X_test_s_var10 = input_test.iloc[:, input_test.columns.str.startswith('S_var10_')]

    X_w_prcp, w_prcp_input, w_prcp_out = CNN_W(X_w_prcp, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_w_tmax, w_tmax_input, w_tmax_out = CNN_W(X_w_tmax, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_w_tmin, w_tmin_input, w_tmin_out = CNN_W(X_w_tmin, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_w_srad, w_srad_input, w_srad_out = CNN_W(X_w_srad, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_w_gddf, w_gddf_input, w_gddf_out = CNN_W(X_w_gddf, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var1, s_var1_input, s_var1_out = CNN_S(X_s_var1, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var2, s_var2_input, s_var2_out = CNN_S(X_s_var2, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var3, s_var3_input, s_var3_out = CNN_S(X_s_var3, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var4, s_var4_input, s_var4_out = CNN_S(X_s_var4, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var5, s_var5_input, s_var5_out = CNN_S(X_s_var5, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var6, s_var6_input, s_var6_out = CNN_S(X_s_var6, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var7, s_var7_input, s_var7_out = CNN_S(X_s_var7, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var8, s_var8_input, s_var8_out = CNN_S(X_s_var8, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var9, s_var9_input, s_var9_out = CNN_S(X_s_var9, kernel_size, stride_conv, stride_pool,
                                               filter1, filter2, filter3, pool_size)
    X_s_var10, s_var10_input, s_var10_out = CNN_S(X_s_var10, kernel_size, stride_conv, stride_pool,
                                                  filter1, filter2, filter3, pool_size)

    w_combined = concatenate([w_prcp_out, w_tmax_out, w_tmin_out, w_srad_out, w_gddf_out])
    s_combined = concatenate([s_var1_out, s_var2_out, s_var3_out, s_var4_out, s_var5_out,
                              s_var6_out, s_var7_out, s_var8_out, s_var9_out, s_var10_out])

    w_fc = Dense(w_fc_l, activation='relu')(w_combined)
    w_fc = Flatten()(w_fc)
    w_fc = Model(inputs=[w_prcp_input, w_gddf_input, w_srad_input, w_tmin_input, w_tmax_input], outputs=w_fc)

    s_fc = Dense(s_fc_l, activation='relu')(s_combined)
    s_fc = Flatten()(s_fc)
    s_fc = Model(inputs=[s_var1_input, s_var2_input, s_var3_input, s_var4_input, s_var5_input,
                         s_var6_input, s_var7_input, s_var8_input, s_var9_input, s_var10_input], outputs=s_fc)

    # define inputs
    X_o = np.expand_dims(input_train.iloc[:, :17], axis=2)
    X_test_o = np.expand_dims(input_test.iloc[:, :17], axis=2)
    input_o = Input(shape=(X_o.shape[1], 1))

    # third branch (others FC)
    o_fc = Dense(FC1_l1, activation='relu')(input_o)
    o_fc = Dense(FC1_l2, activation='relu')(o_fc)
    o_fc = Dense(FC1_l3, activation='relu')(o_fc)
    o_fc = Flatten()(o_fc)
    o_fc = Model(inputs=input_o, outputs=o_fc)

    # concatenate
    combined = concatenate([w_fc.output, s_fc.output, o_fc.output])

    # final fully connected network
    q = Dense(FC2_l1, activation='relu')(combined)
    q = Dropout(dropout)(q)
    q = Dense(FC2_l2, activation='relu')(q)
    q = Dropout(dropout)(q)
    q = Dense(FC2_l3, activation='linear')(q)

    # defining the model, optimizer, and compiling
    model = Model(inputs=[w_prcp_input, w_tmax_input, w_tmin_input, w_srad_input, w_gddf_input,
                          s_var1_input, s_var2_input, s_var3_input, s_var4_input, s_var5_input,
                          s_var6_input, s_var7_input, s_var8_input, s_var9_input, s_var10_input, input_o], outputs=q)
    opt = Adam(lr=lr)
    model.compile(loss='mse', optimizer=opt)

    xs = [X_w_prcp, X_w_tmax, X_w_tmin, X_w_srad, X_w_gddf,
          X_s_var1, X_s_var2, X_s_var3, X_s_var4, X_s_var5,
          X_s_var6, X_s_var7, X_s_var8, X_s_var9, X_s_var10, X_o]
    valids = [X_test_w_prcp, X_test_w_tmax, X_test_w_tmin, X_test_w_srad, X_test_w_gddf,
              X_test_s_var1, X_test_s_var2, X_test_s_var3, X_test_s_var4, X_test_s_var5,
              X_test_s_var6, X_test_s_var7, X_test_s_var8, X_test_s_var9, X_test_s_var10, X_test_o]

    # training the model
    history = model.fit(x=xs, validation_data=(valids, y_test),
                        y=y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(valids)
    preds_train = model.predict(xs)
    rmse_train = np.sqrt(model.evaluate(xs, y_train))

    return model, preds, rmse_train, preds_train, history



### ---------- MODEL 1 ---------- ###

print('\n $$ Training Model 1... \n')

model1, preds1, rmse_train1, preds_train1, history1 = CNN_DNN(input_train=X, input_test=x_test, kernel_size=3,
                                                              stride_conv=1, stride_pool=2, y_train=Y, w_fc_l=60, s_fc_l=40,
                                                              filter1=2, filter2=2, filter3=2, pool_size=2,
                                                              FC1_l1=64, FC1_l2=32, FC1_l3=16, FC2_l1=128, FC2_l2=64, FC2_l3=1,
                                                              dropout=0.5, lr=0.0001, epochs=1000, batch_size=16)
rmse_test1 = np.sqrt(mse(y_test, preds1))
r2_test1 = r2_score(y_test, preds1)
r2_train1 = r2_score(Y, preds_train1)
model1_saved = model1.save('het_model1_saved_'+str(test_year)+scenario+'.h5')
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('het_model1_plot_'+str(test_year)+scenario+'.png')
plt.close()


### ---------- MODEL 2 ---------- ###

print('\n $$ Training Model 2... \n')

model2, preds2, rmse_train2, preds_train2, history2 = CNN_DNN(input_train=X, input_test=x_test, kernel_size=3,
                                                              stride_conv=1, stride_pool=2, y_train=Y, w_fc_l=60, s_fc_l=40,
                                                              filter1=3, filter2=3, filter3=3, pool_size=2,
                                                              FC1_l1=64, FC1_l2=32, FC1_l3=16, FC2_l1=128, FC2_l2=64, FC2_l3=1,
                                                              dropout=0.5, lr=0.0001, epochs=1000, batch_size=16)
rmse_test2 = np.sqrt(mse(y_test, preds2))
r2_test2 = r2_score(y_test, preds2)
r2_train2 = r2_score(Y, preds_train2)
model2_saved = model2.save('het_model2_saved_'+str(test_year)+scenario+'.h5')
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('het_model2_plot_'+str(test_year)+scenario+'.png')
plt.close()


### ---------- MODEL 3 ---------- ###

print('\n $$ Training Model 3... \n')

model3, preds3, rmse_train3, preds_train3, history3 = CNN_DNN(input_train=X, input_test=x_test, kernel_size=3,
                                                              stride_conv=1, stride_pool=2, y_train=Y, w_fc_l=60, s_fc_l=40,
                                                              filter1=4, filter2=4, filter3=4, pool_size=2,
                                                              FC1_l1=64, FC1_l2=32, FC1_l3=16, FC2_l1=128, FC2_l2=64, FC2_l3=1,
                                                              dropout=0.5, lr=0.0001, epochs=1000, batch_size=16)
rmse_test3 = np.sqrt(mse(y_test, preds3))
r2_test3 = r2_score(y_test, preds3)
r2_train3 = r2_score(Y, preds_train3)
model3_saved = model3.save('het_model3_saved_'+str(test_year)+scenario+'.h5')
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('het_model3_plot_'+str(test_year)+scenario+'.png')
plt.close()


### ---------- MODEL 4 ---------- ###

print('\n $$ Training Model 4... \n')

model4, preds4, rmse_train4, preds_train4, history4 = CNN_DNN(input_train=X, input_test=x_test, kernel_size=3,
                                                              stride_conv=1, stride_pool=2, y_train=Y, w_fc_l=60, s_fc_l=40,
                                                              filter1=5, filter2=5, filter3=5, pool_size=2,
                                                              FC1_l1=64, FC1_l2=32, FC1_l3=16, FC2_l1=128, FC2_l2=64, FC2_l3=1,
                                                              dropout=0.5, lr=0.0001, epochs=1000, batch_size=16)
rmse_test4 = np.sqrt(mse(y_test, preds4))
r2_test4 = r2_score(y_test, preds4)
r2_train4 = r2_score(Y, preds_train4)
model4_saved = model4.save('het_model4_saved_'+str(test_year)+scenario+'.h5')
plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('het_model4_plot_'+str(test_year)+scenario+'.png')
plt.close()


### ---------- MODEL 5 ---------- ###

print('\n $$ Training Model 5... \n')

model5, preds5, rmse_train5, preds_train5, history5 = CNN_DNN(input_train=X, input_test=x_test, kernel_size=3,
                                                              stride_conv=1, stride_pool=2, y_train=Y, w_fc_l=60, s_fc_l=40,
                                                              filter1=6, filter2=6, filter3=6, pool_size=2,
                                                              FC1_l1=64, FC1_l2=32, FC1_l3=16, FC2_l1=128, FC2_l2=64, FC2_l3=1,
                                                              dropout=0.5, lr=0.0001, epochs=1000, batch_size=16)
rmse_test5 = np.sqrt(mse(y_test, preds5))
r2_test5 = r2_score(y_test, preds5)
r2_train5 = r2_score(Y, preds_train5)
model5_saved = model5.save('het_model5_saved_'+str(test_year)+scenario+'.h5')
plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('het_model5_plot_'+str(test_year)+scenario+'.png')
plt.close()


## -------------------------- OPTIMIZED ENSEMBLE -------------------------- ##

print('\n $$ Creating Ensembles... \n')

## OOB predictions

valid_X_w_prcp = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('prcp')], axis=2)
valid_X_w_tmax = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('tmax')], axis=2)
valid_X_w_tmin = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('tmin')], axis=2)
valid_X_w_srad = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('srad')], axis=2)
valid_X_w_gddf = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('gddf')], axis=2)
valid_X_s_var1 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var1_')], axis=2)
valid_X_s_var2 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var2_')], axis=2)
valid_X_s_var3 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var3_')], axis=2)
valid_X_s_var4 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var4_')], axis=2)
valid_X_s_var5 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var5_')], axis=2)
valid_X_s_var6 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var6_')], axis=2)
valid_X_s_var7 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var7_')], axis=2)
valid_X_s_var8 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var8_')], axis=2)
valid_X_s_var9 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var9_')], axis=2)
valid_X_s_var10 = np.expand_dims(x_valid.iloc[:, x_valid.columns.str.startswith('S_var10_')], axis=2)
valid_X_o = np.expand_dims(x_valid.iloc[:, :17], axis=2)
valids = [valid_X_w_prcp, valid_X_w_tmax, valid_X_w_tmin, valid_X_w_srad, valid_X_w_gddf,
          valid_X_s_var1, valid_X_s_var2, valid_X_s_var3, valid_X_s_var4, valid_X_s_var5, valid_X_s_var6,
          valid_X_s_var7, valid_X_s_var8, valid_X_s_var9, valid_X_s_var10, valid_X_o]
Model1_oob = model1.predict(valids)
Model2_oob = model2.predict(valids)
Model3_oob = model3.predict(valids)
Model4_oob = model4.predict(valids)
Model5_oob = model5.predict(valids)

# Optimizing weights

def objective2(y):
    return mse(y_valid, (y[0]*Model1_oob + y[1]*Model2_oob + y[2]*Model3_oob + y[3]*Model4_oob + y[4]*Model5_oob))

def constraint12(y):
    return y[0] + y[1] + y[2] + y[3] + y[4] - 1.0

y0 = np.zeros(5)
y0[0] = 1 / 5
y0[1] = 1 / 5
y0[2] = 1 / 5
y0[3] = 1 / 5
y0[4] = 1 / 5

b = (0, 1.0)
bnds2 = (b, b, b, b, b)
con12 = {'type': 'eq', 'fun': constraint12}

cons2 = [con12]

solution2 = minimize(objective2, y0, method='SLSQP',
                    options={'disp': True, 'maxiter': 3000, 'eps': 1e-3}, bounds=bnds2,
                    constraints=cons2)
y = solution2.x
pd.DataFrame(y).to_csv('het_opt_weights_'+str(test_year)+scenario+'.csv')

opt_preds_test = y[0]*preds1 + y[1]*preds2 + y[2]*preds3 + y[3]*preds4 + y[4]*preds5
opt_rmse_test = np.sqrt(mse(y_test, opt_preds_test))
opt_r2_test = r2_score(y_test, opt_preds_test)
pd.DataFrame(opt_preds_test).to_csv('het_gem_preds_'+str(test_year)+scenario+'.csv')

opt_preds_train = y[0]*preds_train1 + y[1]*preds_train2 + y[2]*preds_train3 + y[3]*preds_train4 + y[4]*preds_train5
opt_rmse_train = np.sqrt(mse(Y, opt_preds_train))
opt_r2_train = r2_score(Y, opt_preds_train)
pd.DataFrame(opt_preds_train).to_csv('het_gem_preds_train_'+str(test_year)+scenario+'.csv')


avg_preds_test = y0[0]*preds1 + y0[1]*preds2 + y0[2]*preds3 + y0[3]*preds4 + y0[4]*preds5
avg_rmse_test = np.sqrt(mse(y_test, avg_preds_test))
avg_r2_test = r2_score(y_test, avg_preds_test)
pd.DataFrame(avg_preds_test).to_csv('het_bem_preds_'+str(test_year)+scenario+'.csv')

avg_preds_train = y0[0]*preds_train1 + y0[1]*preds_train2 + y0[2]*preds_train3 + y0[3]*preds_train4 + y0[4]*preds_train5
avg_rmse_train = np.sqrt(mse(Y, avg_preds_train))
avg_r2_train = r2_score(Y, avg_preds_train)
pd.DataFrame(avg_preds_train).to_csv('het_bem_preds_train_'+str(test_year)+scenario+'.csv')



## -------------------------------- STACKING -------------------------------- ##

predsDF = pd.DataFrame()
predsDF['model1'] = pd.DataFrame(Model1_oob)[0]
predsDF['model2'] = pd.DataFrame(Model2_oob)[0]
predsDF['model3'] = pd.DataFrame(Model3_oob)[0]
predsDF['model4'] = pd.DataFrame(Model4_oob)[0]
predsDF['model5'] = pd.DataFrame(Model5_oob)[0]
predsDF['Y'] = y_valid
x_stacked = predsDF.drop(columns='Y', axis=1)
y_stacked = predsDF['Y']
testPreds = pd.DataFrame([pd.DataFrame(preds1)[0], pd.DataFrame(preds2)[0], pd.DataFrame(preds3)[0],
                          pd.DataFrame(preds4)[0], pd.DataFrame(preds5)[0]]).T
trainPreds = pd.DataFrame([pd.DataFrame(preds_train1)[0], pd.DataFrame(preds_train2)[0], pd.DataFrame(preds_train3)[0],
                           pd.DataFrame(preds_train4)[0], pd.DataFrame(preds_train5)[0]]).T
testPreds.columns = ['model1', 'model2', 'model3', 'model4', 'model5']
trainPreds.columns = ['model1', 'model2', 'model3', 'model4', 'model5']

stck_reg = LinearRegression()
stck_reg.fit(x_stacked, y_stacked)
stck_reg_preds_test = stck_reg.predict(testPreds)
stck_reg_mse_test = mse(y_test, stck_reg_preds_test)
stck_reg_rmse_test = np.sqrt(stck_reg_mse_test)
stck_reg_r2_test = r2_score(y_test, stck_reg_preds_test)
pd.DataFrame(stck_reg_preds_test).to_csv('het_stckreg_preds_'+str(test_year)+scenario+'.csv')
joblib.dump(stck_reg, 'het_stckreg_saved_'+str(test_year)+scenario+'.sav')
stck_reg_preds_train = stck_reg.predict(trainPreds)
stck_reg_rmse_train = np.sqrt(mse(Y, stck_reg_preds_train))
stck_reg_r2_train = r2_score(Y, stck_reg_preds_train)

stck_lasso = Lasso()
stck_lasso.fit(x_stacked, y_stacked)
stck_lasso_preds_test = stck_lasso.predict(testPreds)
stck_lasso_mse_test = mse(y_test, stck_lasso_preds_test)
stck_lasso_rmse_test = np.sqrt(stck_lasso_mse_test)
stck_lasso_r2_test = r2_score(y_test, stck_lasso_preds_test)
pd.DataFrame(stck_lasso_preds_test).to_csv('het_stcklasso_preds_'+str(test_year)+scenario+'.csv')
joblib.dump(stck_lasso, 'het_stcklasso_saved_'+str(test_year)+scenario+'.sav')
stck_lasso_preds_train = stck_lasso.predict(trainPreds)
stck_lasso_rmse_train = np.sqrt(mse(Y, stck_lasso_preds_train))
stck_lasso_r2_train = r2_score(Y, stck_lasso_preds_train)

stck_rf = RandomForestRegressor()
stck_rf.fit(x_stacked, y_stacked)
stck_rf_preds_test = stck_rf.predict(testPreds)
stck_rf_mse_test = mse(y_test, stck_rf_preds_test)
stck_rf_rmse_test = np.sqrt(stck_rf_mse_test)
stck_rf_r2_test = r2_score(y_test, stck_rf_preds_test)
pd.DataFrame(stck_rf_preds_test).to_csv('het_stckrf_preds_'+str(test_year)+scenario+'.csv')
joblib.dump(stck_rf, 'het_stckrf_saved_'+str(test_year)+scenario+'.sav')
stck_rf_preds_train = stck_rf.predict(trainPreds)
stck_rf_rmse_train = np.sqrt(mse(Y, stck_rf_preds_train))
stck_rf_r2_train = r2_score(Y, stck_rf_preds_train)

stck_lgb = LGBMRegressor()
stck_lgb.fit(x_stacked, y_stacked)
stck_lgb_preds_test = stck_lgb.predict(testPreds)
stck_lgb_mse_test = mse(y_test, stck_lgb_preds_test)
stck_lgb_rmse_test = np.sqrt(stck_lgb_mse_test)
stck_lgb_r2_test = r2_score(y_test, stck_lgb_preds_test)
pd.DataFrame(stck_lgb_preds_test).to_csv('het_stcklgb_preds_'+str(test_year)+scenario+'.csv')
joblib.dump(stck_lgb, 'het_stcklgb_saved_'+str(test_year)+scenario+'.sav')
stck_lgb_preds_train = stck_lgb.predict(trainPreds)
stck_lgb_rmse_train = np.sqrt(mse(Y, stck_lgb_preds_train))
stck_lgb_r2_train = r2_score(Y, stck_lgb_preds_train)



## -------------------------- RESULTS -------------------------- ##

results_test = pd.DataFrame({'models': ['NN1','NN2','NN3','NN4','NN5', 'BEM', 'GEM',
                                        'stck_reg', 'stck_lasso', 'stck_rf', 'stck_lgb'],
                             'rmse': [rmse_test1, rmse_test2, rmse_test3, rmse_test4, rmse_test5,
                                      avg_rmse_test, opt_rmse_test, stck_reg_rmse_test,
                                      stck_lasso_rmse_test, stck_rf_rmse_test, stck_lgb_rmse_test]})
results_train = pd.DataFrame({'models': ['NN1','NN2','NN3','NN4','NN5', 'BEM', 'GEM',
                                         'stck_reg', 'stck_lasso', 'stck_rf', 'stck_lgb'],
                              'rmse': [rmse_train1, rmse_train2, rmse_train3, rmse_train4, rmse_train5,
                                       avg_rmse_train, opt_rmse_train, stck_reg_rmse_train,
                                       stck_lasso_rmse_train, stck_rf_rmse_train, stck_lgb_rmse_train]})
R2_test = pd.DataFrame({'models': ['NN1','NN2','NN3','NN4','NN5', 'BEM', 'GEM',
                                   'stck_reg', 'stck_lasso', 'stck_rf', 'stck_lgb'],
                        'R2': [r2_test1, r2_test2, r2_test3, r2_test4, r2_test5,
                               avg_r2_test, opt_r2_test, stck_reg_r2_test,
                               stck_lasso_r2_test, stck_rf_r2_test, stck_lgb_r2_test]})
R2_train = pd.DataFrame({'models': ['NN1','NN2','NN3','NN4','NN5', 'BEM', 'GEM',
                                    'stck_reg', 'stck_lasso', 'stck_rf', 'stck_lgb'],
                         'R2': [r2_train1, r2_train2, r2_train3, r2_train4, r2_train5,
                                avg_r2_train, opt_r2_train, stck_reg_r2_train,
                                stck_lasso_r2_train, stck_rf_r2_train, stck_lgb_r2_train]})

pd.DataFrame(preds1).to_csv('het_model1_preds_'+str(test_year)+scenario+'.csv')
pd.DataFrame(preds2).to_csv('het_model2_preds_'+str(test_year)+scenario+'.csv')
pd.DataFrame(preds3).to_csv('het_model3_preds_'+str(test_year)+scenario+'.csv')
pd.DataFrame(preds4).to_csv('het_model4_preds_'+str(test_year)+scenario+'.csv')
pd.DataFrame(preds5).to_csv('het_model5_preds_'+str(test_year)+scenario+'.csv')

results_test.to_csv('het_results_test_'+str(test_year)+scenario+'.csv')
results_train.to_csv('het_results_train_'+str(test_year)+scenario+'.csv')
R2_test.to_csv('het_R2_test_'+str(test_year)+scenario+'.csv')
R2_train.to_csv('het_R2_train_'+str(test_year)+scenario+'.csv')