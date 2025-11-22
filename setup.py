"""
MetaGuard Setup
Author: Moslem Mohseni
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='metaguard',
    version='1.0.0',
    description='Simple fraud detection for metaverse transactions',
    long_description="""
    MetaGuard is a Python library that detects suspicious transactions in the metaverse
    with just 3 lines of code. Using machine learning (Random Forest), it achieves 70%+
    accuracy with minimal setup.

    Developed by: Moslem Mohseni
    """,
    author='Moslem Mohseni',
    author_email='',
    url='https://github.com/moslem-mohseni/MetaGuard',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'metaguard': ['models/*.pkl'],
    },
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='fraud-detection metaverse security machine-learning',
    project_urls={
        'Source': 'https://github.com/moslem-mohseni/MetaGuard',
        'Bug Reports': 'https://github.com/moslem-mohseni/MetaGuard/issues',
    },
)
