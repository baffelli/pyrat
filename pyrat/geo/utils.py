import numpy as _np
import matplotlib.pyplot as _plt
import pyrat.visualization.visfun as _vf

class GeocodedVisualization:

    def __init__(self, radar_data, geocoding_table, gt, **kwargs):
        self.f, (self.map_ax, self.radar_ax) = _plt.subplots(1,2)
        self.radar_data = radar_data
        self.table = geocoding_table
        #initialize radar position at center of data
        self.radar_pos = tuple((self.radar_data.shape[0]/2,self.radar_data.shape[1]/2))
        #map position is computed from geocoding table
        self.map_pos = self.table.radar_coord_to_dem_coord(self.radar_pos)
        #Create circles
        c_radar = _plt.Circle(self.radar_pos,  radius=2,ec='red')
        c_map = _plt.Circle(self.map_pos)
        #Visualize data
        k = kwargs.get('k', 0.5)
        sf = kwargs.get('sf', 0.5)
        self.radar_ax.imshow(_vf.exp_im(radar_data,k, sf))
        #Geocode data
        radar_geo = self.table.geocode_data(self.radar_data)
        self.map_ax.imshow(_vf.exp_im(radar_geo, k, sf))
        self.c_radar = self.radar_ax.add_artist(c_radar)
        self.c_map = self.radar_ax.add_artist(c_map)
        #Connect events
        self.f.canvas.mpl_connect('button_press_event', self.on_press)
        # self.f.canvas.mpl_connect('button_release_event', self.on_release())
        self.f.show()

    def update_radar_pos(self):
        self.radar_pos = self.table.dem_coord_to_radar_coord(self.map_pos)
        self.c_radar.center = self.radar_pos

    def update_map_pos(self):
        print('ere')
        self.map_pos = self.table.radar_coord_to_dem_coord(self.radar_pos)
        self.c_map.center = self.map_pos

    def on_press(self, event):
        """
        Event for when the mouse button is pressed
        Parameters
        ----------
        event

        Returns
        -------

        """
        #If not on radar or map axis, return
        if event.inaxes != (self.geo_ax or self.radar_ax): return
        if event.inaxes == self.map_ax:
            self.map_pos = (event.xdata, event.ydata)
            self.update_radar_pos()
        if event.inaxes == self.radar_ax:
            self.radar_pos = (event.xdata, event.ydata)
            self.update_map_pos()


    # def on_release(self):
    #     self.press = None
    #     self.f.canvas.draw()