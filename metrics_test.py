#%%
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
#Define a variable to keep track of the total RMSE
total_rmse = 0
time_series_data = pd.read_csv("ts_cv.csv", index_col = "DATE", parse_dates = ["DATE"])
#time_series_data = np.asarray(time_series_data.dtypes(""))
type(time_series_data)
time_series_data.info()
## prove that those two are the same
dataset = pd.read_csv("loocv_data.csv")
dataset['DATE'] = pd.to_datetime(dataset['DATE'],infer_datetime_format=True) #convert from string to datetime
indexedDataset = dataset.set_index(['DATE'])
indexedDataset.info()
#make sure the dataset is a series type one
def test_for_arima(dataset = None, p = 0, d = 0, q = 0, P = 0, D = 0, Q =0, S=0, steps = 0,n_splits = 0 , test_size = 0):
    _ = input("input dataset,(p,d,q),(P,D,Q),steps,n_splits and test_size")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits = n_splits, test_size = test_size)
    mape_array = []
    for _, (train_index, test_index) in enumerate(tscv.split(dataset)):
        model = ARIMA(dataset.iloc[train_index,], order=(p, d, q), seasonal_order=(P, D, Q, S))
        results = model.fit()
        forecast_results = results.forecast(steps=steps).to_frame()
        predict_array = forecast_results.values
        true_array = dataset.iloc[test_index,].values
        ape = np.abs((predict_array - true_array) / predict_array) * 100
        mape = np.mean(ape)
        mape_array.append(mape)
    return np.mean(mape_array)
#%%
test_for_arima(indexedDataset,2,1,0,1,0,0,12,4,2,4)
#%%
# # pd.Series(time_series_data, index = dates)
# # time_series_data.info()
# #%%
#%%
# Define the ARIMA order (p, d, q) and seasonal order (P, D, Q, s)
p, d, q = 2, 1, 0
P, D, Q, s = 1, 0, 0, 12
# Create an ARIMA model with seasonal order
model = ARIMA(indexedDataset, order=(p, d, q), seasonal_order=(P, D, Q, s))
model.fit().summary()
#%%
dataset
#%%time series cross validation

# # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
# # y = np.array([1, 2, 3, 4, 5, 6])
# # tscv = TimeSeriesSplit()
# # print(tscv)
# # for i, (train_index, test_index) in enumerate(tscv.split(X)):
# #     print(f"Fold {i}:")
# #     print(f"  Train: index={train_index}")
# #     print(f"  Test:  index={test_index}")
# #     model = ARIMA(indexedDataset, order=(2, 1, 0), seasonal_order=(1, 0, 0, 12))
# #     model.fit()
# #%%
# # Fix test_size to 2 with 12 samples
# # X = np.random.randn(12, 2)
# # y = np.random.randint(0, 2, 12)
#%%


def mean_absolute_percentage_error(actual, forecast): 
    ape = np.abs((actual - forecast) / actual) * 100
    mape = np.mean(ape)
    return mape

#%%
      

        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        #for i in len(predict_array):
        # for i in range(steps):
        #     pred_1, pred_2, pred_3, pred_4 = results.forecast(steps=steps)
        #     pred = [i for i in range(4)]
        #     pred[0],pred[1],pred[2],pred[3] = pred_1, pred_2, pred_3, pred_4
#This how I use enumerate to test the data set
# results.forecast(steps=4)
# Create a null DataFrame with only column names
# metrics = pd.DataFrame(columns=column_names#%%
#%%

#%%

# #%%
# indexedDataset.iloc[train_index,].shape
# #%%
# # Add in a 2 period gap
# tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
# for i, (train_index, test_index) in enumerate(tscv.split(X)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}")
#     print(f"  Test:  index={test_index}")
# #%%
# import pandas as pd

# # Assuming 'data' is a list of numerical values with corresponding dates as the index
# dates = pd.date_range('2023-01-01', periods=len(time_series_data))  # Replace with actual date range
# time_series = pd.Series(time_series_data, index=dates)
# time_series
# # %%
# import pandas as pd

# # Create a pandas DataFrame
# data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
# df = pd.DataFrame(data)

# # Convert the DataFrame to a NumPy array
# array = df.to_numpy()

# print(array)

# # %%
# import pandas as pd
# import numpy as np
# import datetime

# # Generate Date/Time Index
# start_date = datetime.date(2023, 1, 1)
# end_date = datetime.date(2023, 12, 31)
# date_index = pd.date_range(start=start_date, end=end_date, freq='D')

# # Generate Time Series Data
# data = np.sin(np.arange(len(date_index)) * 2 * np.pi / 365)

# # Create a DataFrame
# time_series_data = pd.DataFrame({'Date': date_index, 'Value': data})

# # Set the Date as the Index
# time_series_data.set_index('Date', inplace=True)

# # Print the first few rows of the time series data
# print(time_series_data.head())
# #%%
# import pandas as pd
# import numpy as np
# import datetime

# # Generate Date/Time Index
# start_date = datetime.date(2023, 1, 1)
# end_date = datetime.date(2023, 12, 31)
# date_index = pd.date_range(start=start_date, end=end_date, freq='M')

# # Generate Time Series Data
# data = np.sin(np.arange(len(date_index)) * 2 * np.pi / 365)

# # Create a DataFrame
# time_series_data = pd.DataFrame({'Date': date_index, 'Value': data})

# # Set the Date as the Index
# time_series_data.set_index('Date', inplace=True)

# # Print the first few rows of the time series data
# print(time_series_data.head)

# # %%
# #check the AIC and AICc
# import statsmodels.api as sm
# model = sm.tsa.ARIMA(time_series_data["Value"], order= (1,0,0))
# results = model.fit()
# aic = results.aic
# aic
# # %%
# results.summary()
# # %%
