from setuptools import setup

setup(name='pyrat',
      version='0.5',
      description='Python radar tools',
      author='Simone Baffelli',
      author_email='baffelli@ifu.baug.ethz.ch',
      license='MIT',
      packages=['pyrat', 'pyrat.core', 'pyrat.geo', 'pyrat.diff', 'pyrat.fileutils', 'pyrat.fileutils', 'pyrat.visualization', 'pyrat.gpri_utils'],
      zip_safe=False,
      install_requires=['numpy', 'matplotlib', 'pyparsing'],
      package_data={'pyrat': ['rules/*']},
      include_package_data=True)
