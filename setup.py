"""
Setup script for FreshRetail Forecasting.
"""
import os
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description from README.md
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="freshretail",
    version="0.1.0",
    description="Forecasting solution using FreshRetailNet-50K dataset with Supabase integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FreshRetail Team",
    author_email="example@example.com",
    url="https://github.com/username/freshretail-forecasting",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'freshretail-api=app.main:main',
            'freshretail-load-data=scripts.load_data:main',
        ],
    },
) 