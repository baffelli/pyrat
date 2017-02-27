import setuptools
from setuptools import setup

# all_rules = _glob.glob(_os.path.dirname(__file__) + '/rules/*.snake')
# rules = {_os.path.splitext(_os.path.basename(rule_path))[0]: rule_path
#          for rule_path in all_rules}

# print(setuptools.find_packages(exclude=['*tests*']))

# Find all scripts
wrappers = setuptools.findall('rules/wrappers/*/*.py')
print(wrappers)

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
                    'default_prf': 'fileutils/default_prf.prf'},
      include_package_data=True)
