from setuptools import setup, find_packages

setup(
    name='PyCalib',
    version='0.0.1',
    packages=find_packages(),
    author='Hongyang Yu',
    author_email="yhytxwd@gmail.com",
    description='An automatic engine calibration package',
    install_requires=[
        'numpy >= 1.9.0',
        'scipy >= 0.14.0',
        'pandas >= 0.20.0',
        'matplotlib >= 2.2.0',
    ],
)
