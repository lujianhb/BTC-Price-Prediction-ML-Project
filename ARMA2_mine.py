from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn import linear_model
import numpy as np


class MyModel:
    def __init__(self, datas):
        self.datas = datas
        self.trainy = datas['trainy']
        self.trainx = datas['trainx']
        self.testx = datas['testx']
        self.testy = datas['testy']
        self.trainsize = len(self.trainy)
        self.testsize = len(self.testy)
        self.size = self.trainsize + self.testsize - 1

    def get_predictions(self):
        pass


class MARMA(MyModel):

    def get_predictions(self):
        datas = self.datas
        model = ARMA(
            datas['trainy'],
            exog=datas['trainx'],
            order=(0, 1, 1)
        )

        results = model.fit()
        predictions = results.predict(start=self.trainsize,
                                      end=self.size - 1,
                                      exog=self.testx)
        return predictions


class MAR(MyModel):

    def get_predictions(self):
        datas = self.datas
        model = AR(datas['trainy'])
        # training the model
        results = model.fit()
        train_size = len(datas['trainy'])
        test_size = len(datas['testy'])
        predictions = results.predict(start=train_size, end=train_size + test_size - 1)
        return predictions


class MARIMA(MyModel):

    def get_predictions(self):
        model = ARIMA(
            self.trainy,
            exog=self.trainx,
            order=(0, 1, 1)
        )

        results = model.fit()
        predictions = results.predict(start=self.trainsize, end=self.size - 1, exog=self.testx)
        return predictions

class MBayesian(MyModel):

    def get_predictions(self):
        reg = linear_model.BayesianRidge()
        x= self.trainx['Close']
        x = np.array(x)
        y = np.array(self.trainy)
        reg.fit(x.reshape((len(self.trainx), 1)), y)
        tx = self.testx['Close']
        tx = np.array(tx)
        ypred = reg.predict(tx.reshape((len(tx), 1)))
        return ypred
