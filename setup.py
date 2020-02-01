from setuptools import setup

requires = [
        'numpy',
        'scipy',
        'sympy',
        'picos',
        'cvxopt',
        'qinfo @ git+https://github.com/jarthurgross/numpy-quantum-info',
        'pysme @ git+https://git@github.com/CQuIC/pysme',
         ]

setup(name='irrep_codes',
      version='0.1',
      install_requires=requires,
      packages=['irrep_codes'],
      package_dir={'': 'src'},
      package_data={'': ['data/*.txt']},
     )
