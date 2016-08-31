# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:38:00 2014

@author: baffelli
"""

import numpy as np
import pickle
class interactiveROI:
    
    
    
    def __init__(self,ax, path):
        if ax is None:
            raise RuntimeError('Pass an axis to initiate the ROI')
        try:
            self = np.load(path)
        except:
            self.fig = ax.figure
            self.ax = ax
            self.canvas = self.fig.canvas
            #Line to show the current polygon
            self.line = []
            #Each polygon is contained in a list
            self.poly_list = []
            #Connect evend handles
    #        canvas.mpl_connect('draw_event', self.draw_callback)
            self.canvas.mpl_connect('button_press_event', self.button_press_callback)
            self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
    #        canvas.mpl_connect('key_press_event', self.key_press_callback)
            self.canvas.mpl_connect('button_release_event', self.button_release_callback)
            #Flags
            self.draw_new = True
            self.poly_counter = 0
            self.ind = None
            self.edit_poly = False
            self.epsilon = 20
            self.shape = self.get_data_size()
            self.mask = np.zeros(self.shape)
            self.path = path
        
    def update_line(self,new_data):
        x_previous, y_previous = self.line[self.poly_counter].get_data()
        nd = (x_previous + [new_data[0]],y_previous + [new_data[1]])
        self.line[self.poly_counter].set_data(nd)
        
    def get_data_size(self):
         shape = (sum(self.ax.get_ybound()), sum(self.ax.get_xbound()))
         return shape
    
    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        # display coords
        xy = np.asarray(self.current_poly.xy)
        xyt = self.current_poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
    
        if d[ind]>=self.epsilon:
            ind = None
    
        return ind
    
    def fill_mask(self):
        shape = self.shape
        x_grid = np.arange(shape[0])
        y_grid = np.arange(shape[1])
        x,y = np.meshgrid(y_grid,x_grid)
        grid_to_test = np.array([y.flatten(),x.flatten()]).transpose()
        for p in self.poly_list:
            pts = p.xy[:,::-1]
            path = matplotlib.path.Path(pts)
            self.mask = self.mask  + path.contains_points(grid_to_test, radius = 0.5).reshape(shape)

    def button_release_callback(self, event):
        if self.edit_poly:
            self.ind = None
            self.fill_mask()
            
    

        
    def button_press_callback(self, event):
        if event.inaxes:
            #Get position data
            x, y = np.ceil(event.xdata), np.ceil(event.ydata)
            #Check if an existing porygon has been selected
            for temp_poly in self.poly_list:
                c = temp_poly.contains(event)
                if  c[0] and event.button == 1:
                    self.poly_counter = temp_poly.get_gid()
                    self.draw_new = False
                    self.edit_poly = True
                    break
            if self.edit_poly:
                if event.button ==1:
                     self.ind = self.get_ind_under_point(event)
                if event.button == 3:
                    self.draw_new = True
                    self.edit_poly = False
                    self.ind = None
            if self.draw_new:
                try:
                    self.line[self.poly_counter]
                except:
                    self.line = self.line + [Line2D([x],[y],marker = 'o')]
                    self.ax.add_line(self.line[self.poly_counter])
                else:
                    if event.button == 1:
                        self.update_line((x,y))
                        self.canvas.draw()
                    #Close Porygon
                    elif event.button == 3:
                        self.close_porygon()
                        self.fill_mask()

                      
    def close_porygon(self):
          (x,y) = self.line[self.poly_counter].get_data()
          p = Polygon(zip(x,y))
          p.set_gid(self.poly_counter)
          self.poly_list = self.poly_list + [p]
          self.current_poly = p
          self.ax.add_artist(p)
          self.poly_counter = self.poly_counter + 1
          self.canvas.draw()
                        
    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.edit_poly: return
        if event.inaxes is None: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata
        if self.edit_poly:
            if self.ind is not None:
                #Set porygon
                self.poly_list[self.poly_counter].xy[self.ind] = (x,y)
                data =  self.poly_list[self.poly_counter].xy
                self.line[self.poly_counter].set_data(data[:,0],data[:,1])
                for temp_p in self.poly_list:
                    self.ax.draw_artist(temp_p) 
        self.canvas.draw()
        self.canvas.blit(self.ax.bbox)
    
    def save(self):
        (self, self.path)


        