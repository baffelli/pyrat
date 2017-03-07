from unittest import TestCase

import h5py

from pyrat.luigi import HDF5 as HDF5

class TestHDF5FileSystem(TestCase):

    def setUp(self):
        #Create store
        self.store_path = './store.hdf5'
        #Add some data
        with h5py.File(self.store_path,mode='a') as of:
            of.create_group('slc')
            of['slc'].create_dataset('test')
        #Create filesystem object
        self.fs = HDF5.HDF5FileSystem(self.store_path)

    def test_open(self):
        self.fs.open()


    def test_exists(self):
        self.fail()

    def test_mkdir(self):
        self.fail()

    def test_isdir(self):
        self.fail()

    def test_remove(self):
        self.fail()
