#!/usr/bin/env python
from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='deepfix',
    version='0.0.1',
    description='',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/adgaudio/deepfix",
    author='Alex Gaudio',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    packages=['deepfix', 'deepfix.models'],
    scripts=[],
    install_requires=[],
        #  "torchvision", "torch",
        #  "pretrainedmodels", "pandas", "numpy", "timm", "einops"],
    extras_require={
        #  'dataset_qualdr': ["pyjq", ]
    }
)
