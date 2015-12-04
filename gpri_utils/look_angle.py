from osgeo import gdal
import numpy as np
import fiona
from shapely.geometry import Point, mapping
from fiona.crs import from_epsg
from_epsg(2056)
{'init': 'epsg:2056', 'no_defs': True}
import matplotlib.pyplot as plt
DEM = '/Data/Geodata/HIL/SwissAlti3D_13_Clip.tif'
GPRI_pos = [680555.13, 251129.41]


#Read dataset
SA = gdal.Open(DEM)
x0, dx, dxdy, y0, dydx, dy = SA.GetGeoTransform()

elevation = SA.ReadAsArray()
nrow, ncol = elevation.shape

def pixel_indices(x,y):
    x1 = int((x - x0)/dx)
    y1 = int((y - y0)/dy)
    return (x1, y1)

radar_pos = GPRI_pos + [elevation[pixel_indices(GPRI_pos[0],GPRI_pos[1])],]


#Dihedral 1
reflectors = {}
reflectors['D1'] = {}
reflectors['D1']['coords'] = [2681014.06, 1250751.81, 566.68]
reflectors['D1']['GS_NR'] = 'HG2488'
reflectors['D1']['type'] = 'dihedral'
reflectors['D1']['elevation_angle'] = 0
#Trihedral 1
reflectors['T1'] = {}
reflectors['T1']['coords'] = [2681065.38, 1250923.24, 557.44 ]
reflectors['T1']['GS_NR'] = 'HG6316'
reflectors['T1']['type'] = 'trihedral'
reflectors['T1']['elevation_angle'] = 0
#Trihedral 1
reflectors['T2'] = {}
reflectors['T2']['coords'] = [2680807.75,1250878.22,533.65]
reflectors['T2']['GS_NR'] = 'HG8134'
reflectors['T2']['type'] = 'trihedral'
reflectors['T2']['elevation_angle'] = 0
#Radar
reflectors['radar'] = {}
reflectors['radar']['coords'] = [ 2680535.7, 1251089.1, 520.54]
reflectors['radar']['GS_NR'] = 'HG8368'
reflectors['radar']['type'] = 'radar'
reflectors['radar']['elevation_angle'] = 0


schema = {
    'geometry': 'Point',
    'properties': {
                   'type': 'str',
                   'elevation_angle': 'float',
                   'GS_NR': 'str'},
}

for key, ref in reflectors.iteritems():
    if key is not 'radar':
        vector = np.array(ref['coords']) - np.array(reflectors['radar']['coords'])
        distance = np.sqrt(np.sum(vector**2))
        elevation_angle = np.rad2deg(np.arcsin(vector[2]/distance))
        if ref['type'] is 'dihedral':
            ref['elevation_angle'] = elevation_angle
        elif ref['type'] is 'trihedral':
            ref['elevation_angle'] = 35.3 + elevation_angle

with fiona.open('/Data/Geodata/HIL/reflectors.shp','w', 'ESRI Shapefile', schema, crs=from_epsg(2056)) as of:
    for key,ref in reflectors.iteritems():
        point = Point(ref['coords'][0], ref['coords'][1])
        print(mapping(point))
        of.write({
           'geometry': mapping(point),
            'properties': {
                'type': ref['type'],
                'elevation_angle': ref['elevation_angle'],
                'GS_NR': ref['GS_NR']
            }
       }
       )

