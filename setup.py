from setuptools import setup

setup(name='Pyrat',
      version='0.1',
      description='Python radar tools',
      author='Simone Baffelli',
      author_email='baffelli@ifu.baug.ethz.ch',
      license='MIT',
      packages=['pyrat'],
      zip_safe=False,
      install_requires=['numpy', 'matplotlib'],
      package_data={'pyrat': ['rules/*']},
      include_package_data=True)
