from setuptools import setup, find_packages

requires = [
    'numpy',
    'scipy',
    'sympy',
    'picos',
    'cvxopt',
    'joblib',
    'numpy-quaternion',
    'qinfo @ git+https://github.com/jarthurgross/numpy-quantum-info',
    'pysme @ git+https://git@github.com/CQuIC/pysme@develop',
]

setup(
    name='irrep_codes',
    version='0.1',
    install_requires=requires,
    packages=find_packages(),
    package_data={'': ['data/*.txt']},
)
