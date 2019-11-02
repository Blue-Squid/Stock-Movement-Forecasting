# Necessary Libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import ffn
from pandas_datareader import data as pdr
import yfinance as yf
import datetime

# Ml Libraries 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Pickle 
import pickle 
# Combining pandas-datareader with Yahoo Finance. 
yf.pdr_override() 

# function to classify Stock as pos or neg. 
def classify(actual) :
    if (actual > 0) :
        return 1
    else:
        return -1

# reading stock data into a DataFrame
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2019, 10, 4)
data = pdr.get_data_yahoo('AMZN', start = start_date, end = end_date)

# columns = Date, Open, High, Low, Close, Adjusted Close, Volume. 

# calculating Daily returns. 
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# replacig null values with 0
data['Returns'].fillna(0) 
data['Returns_1'] = data['Returns'].fillna(0)
data['Returns_2'] = data['Returns_1'].replace([np.inf, -np.inf], np.nan)
data['Returns_Final'] = data['Returns_2'].fillna(0) 
data.fillna(0, inplace = True)

# applying classification on dataframe. 
data.iloc[:, len(data.columns) - 1] = data.iloc[:, len(data.columns) - 1].apply(classify) 

# Train-Test spliting, consider recent 20% data as Test Data. 
test_data = data[-int((len(data) * 0.2)):] 
train_data = data[:-int((len(data) * 0.8))]

# feature - target split
features_training = train_data.iloc[:,0:7] 
features_training.fillna(0, inplace = True)
target_training = np.array(train_data['Returns_Final'])
test_features = test_data.iloc[:,0:7]
test_target = np.array(test_data['Returns_Final'])

# Modelling
classifier = LogisticRegression(solver = 'liblinear') 
classifier.fit(features_training, target_training) 
prediction = classifier.predict(test_features) 
print('Accuracy of Base Model:', round(accuracy_score(test_target, prediction),5)) 

# save the model to disk
filename = 'base_model_logReg.sav'
pickle.dump(classifier, open(filename, 'wb')) 

# decision to buy or sell based on SMA  
data['Simple_Moving_Average'] = data.Close.rolling(200).mean() 
data['Sell'] = data.Close < data.Simple_Moving_Average 
data['Buy'] = data.Close > data.Simple_Moving_Average

buy_strength = pd.DataFrame(data['Buy'][-275:]) # buying signal strength for 2019 till now. 
sell_strength = pd.DataFrame(data['Sell'][-275:]) # selling signal strength for 2019 till now. 

last_275_days = pd.DataFrame(classifier.predict(data.iloc[:,0:7][-275:]))
last_275_days['Signal'] = 0 
buy_strength.set_index(last_275_days.index, inplace = True) 
sell_strength.set_index(last_275_days.index, inplace = True) 

# setting up filters
filter1 = (last_275_days[0] > 0)  
filter2 = buy_strength > 0 
filter3 = (last_275_days[0]) < 0  
filter4 = sell_strength > 0  

consolidated_df = pd.concat([last_275_days, buy_strength, sell_strength, filter1, filter2, filter3, filter4], axis = 1)
consolidated_df.columns = ['Prediction', 'Signal', 'Buy_Strength', 'Sell_Strength', 'Filter_1', 'Filter_2', 'Filter_3', 'Filter_4']
consolidated_df['Signal'] = np.where(consolidated_df['Filter_1'] & consolidated_df['Filter_2'], 1, 0)
consolidated_df['Signal'] = np.where(consolidated_df['Filter_3'] & consolidated_df['Filter_4'], -1, consolidated_df['Signal'])

buys = consolidated_df.loc[consolidated_df.Signal == 1]
sells = consolidated_df.loc[consolidated_df.Signal == -1]

if not buys.empty: 
    buys_new_index = buys.index[-1] - buys.index
    buys_new_index_2 = len(data.index) - buys_new_index
    buys.set_index(buys_new_index_2, inplace = True) 
if not sells.empty: 
    sells_new_index = sells.index[-1] - sells.index
    sells_new_index_2 = len(data.index) - sells_new_index
    sells.set_index(sells_new_index_2, inplace = True) 

stock_price = pd.DataFrame(data.iloc[-275:,3].values)

