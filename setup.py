'''Setup.py'''

from distutils.core import setup
from setuptools import find_packages

setup(
    name='t1est',
    version='0.2.0',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/mckib2/t1est',
    license='GPLv3',
    description='Basic T1 fitting',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy>=1.17.4",
        "matplotlib>=3.1.2",
        "phantominator>=0.4.5",
        "tqdm>=4.40.2",
    ],
    python_requires='>=3.6',
)
