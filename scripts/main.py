import pandas as pd
import numpy as np
from urllib.parse import urlparse
import mlflow as mlf
import mlflow.sklearn
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


import dvc.api

path='data/AdSmartABdatav4.csv'
repo='C:\\Users\\cthru\\Documents\\psnl_projects\\10academy\\campaign_perfomance'
version='v4'


data_url = dvc.api.get_url(
    path=path,
    repo=repo,
    rev=version
)

mlf.set_experiment('demo')

def clean_df(df):
    df.set_index('auction_id',inplace=True)
    df['experiment'] = pd.Categorical(df.experiment)
    df['device_make'] = pd.Categorical(df.device_make)
    df['browser'] = pd.Categorical(df.browser)
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def make_prediction(x_train, x_test, y_train, y_test):
    lrm = LogisticRegression()
    dtm=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
    xgbm = xgb.XGBClassifier(eval_metric='mlogloss')

    #fitting our classifiers
    lrm.fit(x_train, y_train)
    dtm.fit(x_train, y_train)
    xgbm.fit(x_train, y_train)
    #make predictions
    predictedlrm = cross_val_predict(lrm, x_test, y_test, cv=10)
    predictedclf = cross_val_predict(dtm, x_test, y_test, cv=10)
    predictedxgb = cross_val_predict(xgbm, x_test, y_test, cv=10)

    #print the rmse and r2 scores
    print('lrm RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictedlrm)))
    print('dtm RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictedclf)))
    print('xgb RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictedxgb)))
    print('lrm R2:', r2_score(y_test, predictedlrm))
    print('dtm r2:', r2_score(y_test, predictedclf))
    print('xgb r2:', r2_score(y_test, predictedxgb))

if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    data = pd.read_csv(data_url, sep=",")

    df = clean_df(data)
    print(df.head())
    dataX = df.iloc[:,[1,3,4,5,8]].values  
    dataY = df.iloc[:,9].values
  
    train_ratio = 0.75
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)



    # log data params
    mlf.log_param('data_url', data_url)
    mlf.log_param('data_version', version)
    mlf.log_param('input_rows', dataX.shape[0])
    mlf.log_param('input_cols', dataX.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

    make_prediction(x_train, x_test, y_train, y_test)