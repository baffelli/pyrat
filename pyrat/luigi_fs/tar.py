import luigi
import h5py
import tarfile

import os

import glob

import re
from pyrat.diff.utils import ListOfDates







class TarFileSystem(luigi.target.FileSystem):

    def __init__(self, store, mode='r'):
        self.store = store
        self._mode = mode

    def open(self):
        return tarfile.open(self.store, self._mode)

    def exists(self, path):
        with self.open() as f:
            does = path in f.getnames()
            return does

    # def mkdir(self, path):
    #     directory = os.path.dirname(path)
    #     with self.open() as f:
    #         if not self.exists(path):
    #             f.create_group(directory)
    #
    # def isdir(self, path):
    #     with self.open() as f:
    #         return path in f.keys()


    def remove(self):
        pass


class TarTarget(luigi.target.FileSystemTarget):

    def __init__(self,store, path, mode='r'):
        self.path = path
        self._fs = TarFileSystem(store, mode)
        self._mode = mode

    @property
    def fs(self):
        return self._fs

    def open(self):
        return self.fs.open().extractfile(self.path)

    def exists(self):
        return self.fs.exists(self.path)