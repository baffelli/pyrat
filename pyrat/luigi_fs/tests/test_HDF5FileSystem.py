from unittest import TestCase

import h5py

from pyrat.luigi_fs import HDF5 as HDF5

class TestHDF5FileSystem(TestCase):

    def setUp(self):
        #Create store
        self.store_path = './store.hdf5'
        #Add some data
        with h5py.File(self.store_path,mode='a') as of:
            try:
                of.create_group('slc')
                of['slc'].create_dataset('test', [1,2,3])
            except ValueError:
                pass
        #Create filesystem object
        self.fs = HDF5.HDF5FileSystem(self.store_path)

    def test_open(self):
        self.fs.open()

    def test_exists(self):
        self.fs.exists('slc/test')
        self.assertFalse(self.fs.exists('carne'))

    def test_mkdir(self):
        self.fs.mkdir('a')
        self.assertTrue(self.fs.exists('a'))

    def test_isdir(self):
        self.fail()

    def test_remove(self):
        self.fail()
