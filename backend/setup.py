from setuptools import setup, find_packages

setup(
    name="resumatch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'pandas',
        'scikit-learn',
        'tqdm',
        'numpy',
        'requests'
    ],
)
