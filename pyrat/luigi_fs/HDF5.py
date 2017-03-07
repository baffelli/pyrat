import luigi
import h5py
import tarfile

import os

import glob

import re
from pyrat.diff.utils import ListOfDates







class HDF5FileSystem(luigi.target.FileSystem):

    def __init__(self, store, mode='a'):
        self.store = store
        self.mode = mode

    def open(self):
        return h5py.File(self.store, self.mode)

    def exists(self, path):
        with self.open() as f:
            does = path in f
            return does

    def mkdir(self, path):
        directory = os.path.dirname(path)
        with self.open() as f:
            if not self.exists(path):
                f.create_group(directory)

    def isdir(self, path):
        with self.open() as f:
            return path in f.keys()


    def remove(self):
        pass


class HDF5Target(luigi.target.FileSystemTarget):

    def __init__(self,store, path, mode='a'):
        self.path = path
        self._fs = HDF5FileSystem(store, mode)

    @property
    def fs(self):
        return self._fs

    def open(self, mode):
        self._fs.open(mode)

    def mkdir(self):
        self.fs.mkdir(self.path)

    def exists(self):
        return self.fs.exists(self.path)