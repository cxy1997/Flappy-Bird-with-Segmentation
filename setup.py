import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = ['opencv-python', 'numpy', 'gym']

setup(name='Flappy_Bird_with_Segmentation',
      version='1.0',
      description='Flappy bird environment with ground truth segmentation',
      author='Xiangyu Chen',
      author_email='cxy_1997@sjtu.edu.cn',
      url='https://github.com/cxy1997/Flappy-Bird-with-Segmentation/',
      keywords='Flappy Bird',
      packages=find_packages(),
      license='LICENSE',
      install_requires=requires)