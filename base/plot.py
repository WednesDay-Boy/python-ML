# -*- coding:utf-8 -*-

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

class Plot:
    """
    Class for plotting data
    python version : 3.6.3

    parameters
    -------------------------
    _xlabel : string
    _ylabel : string
    _legend : string
    _output_path : string
    """

    def __init__(self, xlabel, ylabel, legend, output_path):
        
        """
        constructor
        ----------------------------------

        param path : string
        param eta : float
        param n_iter : int
        return : void
        """

        self._xlabel = xlabel
        self._ylabel = ylabel
        self._legend = legend
        self._output_path = output_path
    
    def output_plot(self, settings):
        """
        output image file
        ---------------------------

        param setting : list
        return void
        """
        
        # Setting axis labels
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)

        # Legend setting
        plt.legend(loc = self._legend)

        for s in enumerate(settings):
            plt.scatter(s[1][0], s[1][1], color = s[1][2], marker = s[1][3], label = s[1][4])
        
        plt.savefig(self._output_path)
    
    def output_decision_regions(self, xx1, xx2, X, y, uniqued_y, Z, alpha):
        """
        Plot non-deterministic regions
        -------------------------------
        param xx1 : array
        param xx2 : array
        param y : list
        param uniqued_y : list
        param Z : array
        param alpha : float
        return void
        """
        
        print(uniqued_y)

        markers = ('s', 'x', 'o', '^', 'v')
        
        colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan') # color map
        
        cmap = ListedColormap(colors[:len(uniqued_y)])

        plt.contourf(xx1, xx2, Z, alpha = alpha, cmap = cmap)

        # Setting axis range
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx ,c1 in enumerate(uniqued_y):

            plt.scatter(
                    x = X[c1 == y, 0],
                    y = X[c1 == y, 1],
                    alpha = 0.8,
                    c = cmap(idx),
                    marker = markers[idx],
                    label = c1
                    )
        
        plt.savefig(self._output_path)


