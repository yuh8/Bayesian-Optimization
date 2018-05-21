from setuptools import setup, find_packages

setup(
    name='Engine Calibration',
    version='0.0.1',
    url='https://github.com/yuh8/Bayesian-Optimization',
    packages=find_packages(),
    author='Hongyang Yu',
    author_email="yhytxwd@gmail.com",
    description='Bayesian Optimization package',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "pandas >= 0.20.0",
        "matplotlib > = 2.2.0"
    ],
)
