import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
# from pandas_datareader import data


def create_features(df_stock, nlags=10):
    df_resampled = df_stock.copy()
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]
    print("df")
    print(df)
    df = df.dropna(axis=0)

    return df


def create_X_Y(df_lags):
    X = df_lags.drop('lags_0', axis=1)
    Y = df_lags[['lags_0']]
    return X, Y


class Stock_model(BaseEstimator, TransformerMixin):

    def __init__(self, data_fetcher):
        self.log = logging.getLogger()
        self.lr = LinearRegression()
        self._data_fetcher = data_fetcher
        self.log.warning('here')

    def fit(self, X, Y=None):
        data = self._data_fetcher(X)
        df_features = create_features(data)
        df_features, Y = create_X_Y(df_features)
        self.lr.fit(df_features, Y)
        return self

    def predict(self, X, Y=None):
        print("X")
        print(X)
        data = self._data_fetcher(X, last=True)
        # data2 = data.DataReader(X,data_source='yahoo',start='2000-01-01',end='2020-10-14')
        # print("data2")
        # print(data2)
        # result = []
        # for i in range(0, len(data)):
        #     if data.iloc[i, 3] > data.iloc[i - 1, 3]:
        #         result.append(0)
        #     else:
        #         result.append(1)
        # data['buy'] = result
        print("data")
        print(data)
        print(data.columns)
        df_features = create_features(data)
        print("df_Feature")
        print(df_features)
        #acc_f = clf.score(x_testf, y_testf)
        #prec_f = precision_score(y_testf, yf_pred)
        #bal_f = balanced_accuracy_score(y_testf, yf_pred)
        df_features, Y = create_X_Y(df_features)
        predictions = self.lr.predict(df_features)
        result = []
        # for i in range(0, len(data)):
        #     if data.iloc[i, 0] > predictions[-1]:
        #         result.append(0)
        #     else:
        #         result.append(1)
        for i in range(0, len(data)):
            if data.iloc[i, 3] > data.iloc[i - 1, 3]:
                result.append(0)
            else:
                result.append(1)
        data['buy'] = result
        print("data again")
        print(data)
        print("predictions")
        print(predictions)
        Xforest = data[['high', 'open', 'low', 'close', 'adjclose']]
        Yforest = data['buy']
        x_trainf, x_testf, y_trainf, y_testf = train_test_split(Xforest, Yforest, test_size=.20)
        log = LogisticRegression(C=5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
        log.fit(x_trainf, y_trainf)
        ylog_pred = log.predict(x_testf)
        bal_log = balanced_accuracy_score(y_testf, ylog_pred) * 100
        print("balanced accuracy")
        print(bal_log)
        print("y_testf and ylog_pred")
        print(y_testf)
        print(ylog_pred)
        close = data['close']
        predictions2 = log.predict(Xforest)
        predictions2 = np.where(predictions2 == 1, 'Buy', 'Sell')
        predictions2 = predictions2.flatten()[-1]
        #print("predictions2")
        #print(predictions2)
        recent_close = "Most recent close price: "
        msg = "The predicted price is: "
        therefore = "therefore you should "
        wit = "with an accuracy of : "
        percent = "%"
        ending = recent_close + str(close[-1])+", " + msg + str(predictions.flatten()[-1])+", " + therefore+ predictions2+" " + wit+ str(bal_log) + percent
        # return recent_close, close[-1], msg, predictions.flatten()[-1], therefore, predictions2, wit, bal_log, percent
        return ending
