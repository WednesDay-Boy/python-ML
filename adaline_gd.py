# -*- coding:utf-8 -*-

from base import Classifier
from base import Plot

import numpy as np

class AdalineGD(Classifier):
    """
    ADAptive LIner Class that implements the NEuronn classifier
    python version : 3.6.3
    """
    
    def __init__(self, path, eta, n_iter):

        """
        constructor
        ----------------------------------

        param path : string
        param eta : float
        param n_iter : int
        return : void
        """
        super().__init__(path, eta, n_iter)

    def get(self):
        
        """
        ---------------------
        return : data frame
        """
        return self._data

    def getObejectiveVariable(self):
        
        """
        Return the objective variable
        ------------------------

        return : y array
        """
        data = self._data

        # Extraction of objective variable on line 1-100
        y = data.iloc[0:100,4].values

        # Convert iris-setosa to -1, iris-virginica to 1
        y = np.where(y == 'Iris-setosa', -1, 1)

        return y
        
    def getDependentVariables(self):

        """
        Return dependent variable
        -----------------------

        return : X  array
        """
        data = self._data

        # Extraction of the 1st and 3rd rows of the 1-100th row
        x = data.iloc[0:100, [0, 2]].values
        return x
    
    def error(self):
        """
        Detect misclassification
        """
        pass

    def fit(self,X,y):

        """
        Adapt to training data
        -----------------------------

        param X : array shape = [n_samples, n_features] n_sampleはデータの個数、n_featuresは特徴量の個数
        param y: array 配列のようなデータ構造、shape = [n_samples] n_sampleはデータの個数
        return  self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self._n_iter):

            output = self.net_input(X)

            errors = (y - output)

            self.w_[1:] += self._eta * X.T.dot(errors)

            self.w_[0] += self._eta * errors.sum()

            cost  = (errors**2).sum() / 2.0

            self.cost_.append(cost)

        return self
    
    def net_input(self, X):
        """
        Calculate total input
        --------------------

        param X : array
        return : array
        """
        return np.dot(X, self.w_[1:] + self.w_[0])
    
    def predict(self, X):
        """
        Return class label after one step
        ----------------------------------

        return : array
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def activation(self, X):

        """
        Calculate the output of linear activation function
        ---------------------------------------------------
        param : X
        return : 
        """
        return self.net_input(X)

    def plot_data(self, output_path):
        """
        Plot data
        ーーーーーーーーーーーーーーーーーー

        param output_path : string
        return boolean
        """
        data = self.get()

        # Extraction of the 1st and 3rd rows of the 1-100th row
        x = data.iloc[0:100, [0, 2]].values

        plot = Plot('sepal length[cm]', 'petal length[cm]', 'upper left', output_path)

        plot.output_plot([
            [x[:50, 0], x[:50, 1], 'red', 'o', 'setosa'],
            [x[50:100, 0], x[50:100, 1], 'blue', 'x', 'virginica']
        ])

    def plot_decision_regions(self, resolution, output_path):
        """
        Plot non-deterministic regions
        ーーーーーーーーーーーーー

        param resolution : float 
        param output_path : string 
        return void
        """
        data = self.get()

        X = self.getDependentVariables()
        y = self.getObejectiveVariable()

        # Plotting non-deterministic regions
        x1_min = X[:, 0].min() - 1
        x1_max = X[:, 0].max() + 1
        x2_min = X[:, 1].min() - 1
        x2_max = X[:, 1].max() + 1

        # Create a grid point
        xx1, xx2 = np.meshgrid(
                np.arange(x1_min, x1_max, resolution),
                np.arange(x2_min, x2_max, resolution),
                )

        p = self.fit(X, y)
        
        Z = p.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

        Z = Z.reshape(xx1.shape)

        plot = Plot('sepal length[cm]', 'petal length[cm]', 'upper left', output_path)

        plot.output_decision_regions(xx1, xx2, X, y, np.unique(y), Z, 0.4)


if __name__ == '__main__':
    
    adaline = AdalineGD('./datas/iris_data.csv',0.01,10)
    adaline.plot_decision_regions(0.02, 'images/a_boundary.png')
    