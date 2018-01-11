# -*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod
import pandas as pd

class Classifier(metaclass = ABCMeta):

    """
    Base class of classifier
    python version : 3.6.3

    parameters
    -------------------------
    _data : object
    _eta : float
    n_iter : int

    attributes
    ------------------------
    w_  : array
    errors : list
    
    """

    w_ = []
    errors = []

    def __init__(self, path, eta = 0.01, n_iter = 10):
        
        """
        constructor
        ----------------------------------

        param path : string
        param eta : float
        param n_iter : int
        return : void
        """

        self._data = pd.read_csv(path)
        self._eta = eta
        self._n_iter = n_iter

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def getObejectiveVariable(self):
        
        """
        目的変数を返す
        ------------------------

        return : y array
        """
        data = self._data

        # 1-100行目の目的変数の抽出
        y = data.iloc[0:100,4].values

        # iris-setosaを-1,iris-virginicaを1に変換
        y = np.where(y == 'Iris-setosa', -1, 1)

        return y
    
    @abstractmethod
    def getDependentVariables(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def net_input(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def error(self):
        pass

    @abstractmethod
    def plot_data(self, output_path):
        pass
    
    @abstractmethod
    def plot_decision_regions(self, X, y,):
        pass
    

        
    