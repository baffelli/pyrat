import matplotlib as mpl
import numpy as _np
import itertools as _iter

def get_data_coordinates(ann):
    patch_to_data = (ann.axes.transAxes+ ann.get_transform().inverted()).inverted()
    return patch_to_data.transform(ann.get_position())

# def set_relative_position(ann):


def distance(ann1, ann2):
    return _np.linalg.norm(_np.subtract(ann1 ,ann2))

def force(ann1, ann2):
    return -1.0/distance(ann1,ann2)**2

class AnnotationOptimizer:

    def __init__(self, ax):
        self.ann = [child for child in ax.get_children() if isinstance(child, mpl.text.Annotation)]
        # #Initialize position
        # for ann in self.ann:
        #     ann.set_position((50,-10))
        self.axis_to_data = ax.transAxes + ax.transData.inverted()
        self.data_to_axis = self.axis_to_data.inverted()

    def optimize(self, dist):
        for ann1, ann2 in _iter.combinations(self.ann, 2):

            f = force(get_data_coordinates(ann1), get_data_coordinates(ann2))
            print(ann1,get_data_coordinates(ann1),ann2, f)

