# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:13:20 2014

@author: baffelli
"""

import matplotlib.pyplot as plt
from pylab import *


class polarimetricVisualization:
    def __init__(*args, **kwargs):
        self = args[0]
        self.T = args[1]
        self.x = self.T.shape[0] / 2
        self.y = self.T.shape[1] / 2
        self.x1 = self.T.shape[0] / 2
        self.y1 = self.T.shape[1] / 2
        l1 = plt.Line2D([self.x], [self.y], marker='o')
        l2 = plt.Line2D([self.x1], [self.y1], marker='o', color='red')
        self.target = self.T[self.y, self.x]
        self.target1 = self.T[self.y1, self.x1]
        self.fig = plt.figure()
        self.analysis_function = kwargs.pop('vf', 1)
        self.text = plt.suptitle("Some text")
        self.gs = matplotlib.gridspec.GridSpec(3, 3)
        ax1 = self.main_spl = plt.subplot(self.gs[0:3, 0])
        self.main_ax = ax1
        plt.imshow(self.T.pauli_image(**kwargs), interpolation='none')
        self.l1 = self.main_spl.add_line(l1)
        self.l2 = self.main_spl.add_line(l2)
        print('here i am')
        cid = self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        print(cid)
        self.fig.canvas.draw()

    def button_press_callback(self, event):
        # if event.inaxes is self.main_ax:
        print(event)
        if event.button == 1:
            x, y = np.ceil(event.xdata), np.ceil(event.ydata)
            self.x = x
            self.y = y
            self.target = self.T[self.y, self.x]
            self.l1.set_data(x, y)
        if event.button == 3:
            x1, y1 = np.ceil(event.xdata), np.ceil(event.ydata)
            self.x1 = x1
            self.y1 = y1
            self.target_1 = self.T[self.y1, self.x1]
            self.l2.set_ydata(y1)
            self.l2.set_xdata(x1)
        #            plt.figure(self.fig.number)
        ps, ps_cr, th, ph = self.analysis_function(self.target)
        plt.subplot(self.gs[0, 2])
        plt.imshow(ps, interpolation='none', cmap='RdBu_r')
        plt.subplot(self.gs[1, 2])
        plt.imshow(ps_cr, interpolation='none', cmap='RdBu_r')
        self.fig.canvas.draw()
