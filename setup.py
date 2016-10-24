from setuptools import setup
import setuptools
print(setuptools.find_packages())
setup(name='pyrat',
      version='0.5',
      description='Python radar tools',
      author='Simone Baffelli',
      author_email='baffelli@ifu.baug.ethz.ch',
      license='MIT',
      packages=setuptools.find_packages(exclude='pyrat.core.tests'),
      zip_safe=False,
      install_requires=['numpy', 'matplotlib', 'pyparsing'],
      package_data={'pyrat': ['rules/*'], 'default_slc_params':'fileutils/default_slc_par.par'},
      include_package_data=True)
