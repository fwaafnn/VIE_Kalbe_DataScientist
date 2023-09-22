# %% [markdown]
# ##### Afwa Afini - VIE Data Scientist - Kalbe 
# ```
# Tujuan dari pembuatan model machine learning ini adalah untuk dapat memprediksi total quantity harian dari product yang terjual.
# 

# %% [markdown]
# ##### Import library

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.core.reshape.merge import merge
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# %% [markdown]
# ##### Exploratory Data Analysis

# %%
# membaca data transaction
df_trs = pd.read_csv('../data-source/DS-Challenge-Transaction.csv')
df_trs

# %%
# membaca data product
df_prd = pd.read_csv('../data-source/DS-Challenge-Product.csv')
df_prd

# %% [markdown]
# ##### Data Preprocessing
# ##### Merge data

# %%
df_merge = df_trs.merge(df_prd, how = 'left', on = ['ProductID', 'Price'])
df_merge

# %%
df_merge.info()

# %% [markdown]
# ##### Transformasi Data

# %%
df_merge['Date'] = pd.to_datetime(df_merge['Date'], format='%d/%m/%Y')

# %%
df_merge.info()

# %%
# set 'Date' menjadi indeks
df_merge.set_index('Date', inplace=True)

# %%
df_merge.info()

# %% [markdown]
# ##### Membuat data baru untuk regresi

# %%
daily_data = df_merge.groupby(df_merge.index.date)['Qty'].sum().reset_index()
daily_data.columns = ['Date', 'TotalQty']
daily_data

# %% [markdown]
# ##### Visualisasi data

# %%
plt.figure(figsize=(12,6))
plt.plot(daily_data['Date'], daily_data['TotalQty'], marker='o', linestyle='-')
plt.title('Total Quantity Harian')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.grid(True)
plt.show()

# %% [markdown]
# ##### Membuat Data Latih dan Data Uji

# %%
# Data latih
train_data = daily_data.iloc[:-30]

# Data uji (30 hari terakhir)
test_data = daily_data.iloc[-30:]

# %% [markdown]
# ##### Membangun Model ARIMA

# %%
model = ARIMA(train_data['TotalQty'], order=(1,1,1))
model_fit = model.fit()

# Model melakukan prediksi
forecast_result = model_fit.forecast(steps=len(test_data))

# %% [markdown]
# ##### Evaluasi Model

# %%
# menghitung MSE (Mean Square Error)
mse = mean_squared_error(test_data['TotalQty'], forecast_result)
print('Mean Squared Error (MSE):', mse)


