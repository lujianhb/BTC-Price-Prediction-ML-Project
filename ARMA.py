import pandas as pd
import matplotlib.pyplot as plt
import requests
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from data import get_data

df = get_data()
# Creating a copy for making small changes
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['Mean'].shift(-1)
dataset_for_prediction = dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# normalizing the exogeneous variables
from sklearn.preprocessing import MinMaxScaler

sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0: 'Low', 1: 'High', 2: 'Open', 3: 'Close', 4: 'Volume', 5: 'Mean'}, inplace=True)
print("Normalized X")
print(X.head())

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output = pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y = scaler_output
y.rename(columns={0: 'BTC Price next day'}, inplace=True)
y.index = dataset_for_prediction.index
print("Normalized y")
print(y.head())

# train-test split (cannot shuffle in case of time series)
train_size = int(len(df) * 0.9)
test_size = int(len(df)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

# Init the best ARMA model
from statsmodels.tsa.arima_model import ARMA

model = ARMA(
    train_y,
    exog=train_X,
    order=(0, 1, 1)
)

# training the model
results = model.fit()

# get predictions
predictions = results.predict(start=train_size, end=train_size + test_size - 2, exog=test_X)

# setting up for plots
act = pd.DataFrame(scaler_output.iloc[train_size:, 0])
predictions = pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index = test_X.index
predictions['Actual'] = act['BTC Price next day']
predictions.rename(columns={0: 'Pred', 'predicted_mean': 'Pred'}, inplace=True)

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# prediction plots
plt.figure(figsize=(20, 10))
# plt.plot(predictions.index, testActual, label='Actual', color='blue')
# plt.plot(predictions.index, testPredict, label='Pred', color='red')
testPredict = testPredict[1:] / testActual[:-1] - 1
testActual = testActual[1:] / testActual[:-1] - 1
plt.subplot(2, 1, 1)
plt.plot(testActual, label='Actual', color='blue')
plt.plot(testPredict, label='Pred', color='red')
plt.plot(testPredict - testActual, label='Pred-Actual', color='green')
plt.legend()
plt.subplot(2, 1, 2)
profit = testPredict / abs(testPredict) * testActual
plt.plot(np.cumprod(profit + 1))
plt.legend()
plt.show()

# print RMSE
from statsmodels.tools.eval_measures import rmse

print("RMSE:", rmse(testActual, testPredict))
