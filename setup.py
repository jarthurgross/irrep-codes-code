from setuptools import setup

requires = [
        'numpy',
        'picos',
        'cvxopt',
        'pysme @ git+https://git@github.com/CQuIC/pysme',
         ]

setup(name='irrep_codes',
      version='0.1',
      install_requires=requires,
      packages=['irrep_codes'],
      package_dir={'': 'src'},
     )
