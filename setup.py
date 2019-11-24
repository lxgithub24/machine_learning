#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='alibaba_suanfa',
    version=1.0,
    description=(
        '应聘常考算法'
    ),
    long_description=open('README.rst').read(),
    author='Hiang',
    author_email='xiang_liu2013@163.com',
    maintainer='hiang',
    maintainer_email='xiang_liu2013@163.com',
    license='BSD License',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/lxgithub24/leetcode.git',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
    install_requires=[
        'nltk==3.4.5',
        'scikit-learn==0.20.2',
    ]
)
