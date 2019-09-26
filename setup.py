'''Setup.py'''

from distutils.core import setup
from setuptools import find_packages

setup(
    name='t1est',
    version='0.0.2',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/mckib2/t1est',
    license='GPLv3',
    description='T1 fitting',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy>=1.17.2",
        "matplotlib>=3.1.1",
        "phantominator>=0.4.3",
        "tqdm>=4.36.1",
    ],
    python_requires='>=3.6',
)
