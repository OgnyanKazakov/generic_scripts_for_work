from setuptools import setup
from setuptools import find_packages
import os

with open('..\\1.Documentation\\Requirements.txt') as f:
    install_requires = f.read().splitlines()

kw = dict(
    name='MLRR',
    version='1.0.0',
    description='ML Resolution Routing',
    author='Resolve Systems',
	author_email='',
    install_requires=install_requires,
    packages=find_packages(),
    url='')
	
setup(**kw)