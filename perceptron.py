# -*- coding:utf-8 -*-

from base import Classifier
from base import Plot

import numpy as np

class Perceptron(Classifier):
    """
    Class implementing perceptron classifier
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

    def fit(self,X,y):
        """
        Adapt to training data
        -----------------------------

        param X : array shape = [n_samples, n_features] n_sample is number of data、n_features is the number of features
        param y: array 、shape = [n_samples] n_sample is number of data
        return  self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        #　Iterate the training data by the number of training times
        for _ in range(self._n_iter):
            
            errors = 0

            # Update weighting for each sample w1,...wm
            for xi, target in zip(X,y):

                update = self._eta * (target - self.predict(xi))
                
                self.w_[1:] += update * xi
                
                self.w_[0] += update

                # When the weight update is not 0, it counts as misclassification
                errors += int(update != 0.0)

            # Store error for each iteration number
            self.errors_.append(errors)

        return self

    def predict(self, X):
        """
        Return class label after one step
        ----------------------------------

        return : array
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        """
        Calculate total input
        --------------------

        param X : array
        return : array
        """
        return np.dot(X, self.w_[1:] + self.w_[0])

    def error(self):
        """
        Detect misclassification
        """
        pass
    
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

        return true

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
    
    p = Perceptron('./datas/iris_data.csv', 0.01, 10)
    p.plot_decision_regions(0.02, 'images/p_boundary.png')