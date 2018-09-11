from setuptools import setup, Extension 
import os

if __name__=='__main__':
    # conda create -n em-pipeline python=2.7 numpy scipy h5py
    # python setup.py develop install
    setup(name='em_pipeline',
       version='1.0',
       install_requires=['numpy','scipy','h5py'],
       packages=['em_pipeline'])


