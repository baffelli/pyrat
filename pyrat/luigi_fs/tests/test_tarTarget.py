from unittest import TestCase
from pyrat.luigi_fs import tar

class TestTarTarget(TestCase):

    def setUp(self):
        self.store_path = './test.tar'
        self.fs = tar.TarTarget(self.store_path, 'a')

    def test_fs(self):
        self.fail()

    def test_open(self):
        with self.fs.open() as f:
            print(f.read())

    def test_exists(self):
        self.fs.exists()

