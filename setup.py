#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='COSMOS',
    version='1.0',
    description='The software is to implement the COSMOS. Please see the website for details.',
    url='https://github.com/Lin-Xu-lab/COSMOS.git',
    packages=find_packages(where='COSMOS'), 
    package_dir={'': 'COSMOS'} 
)
