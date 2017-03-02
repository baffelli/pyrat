import setuptools
from setuptools import setup

# Find all scripts
wrappers = setuptools.findall('rules/wrappers/*/*.py')
setup(name='pyrat',
      version='0.5',
      description='Python radar tools',
      author='Simone Baffelli',
      author_email='baffelli@ifu.baug.ethz.ch',
      license='MIT',
      packages=setuptools.find_packages(exclude=['*tests*']),
      zip_safe=False,
      install_requires=['numpy', 'matplotlib', 'pyparsing', 'pyfftw', 'gdal', 'pillow'],
      package_data={'rules': ['rules/*.snake'], 'default_slc_params': 'fileutils/default_slc_par.par',
                    'default_prf': 'fileutils/default_prf.prf', 'list_of_slcs':['diff/tests/list_of_slcs.csv'],
                    },
      scripts=['gpri_utils/scripts/*.py'],
      include_package_data=True)
