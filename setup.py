from setuptools import setup
from setuptools import find_packages
import os

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

kw = dict(
    name='Subsidiaries',
    version='1.0.0',
    description='Parent - Child relationships',
    author='Ognyan Kazakov',
	author_email='ognaynk@nsogroup.com',
    install_requires=install_requires,
    packages=find_packages(),
    url='')
	
setup(**kw)
