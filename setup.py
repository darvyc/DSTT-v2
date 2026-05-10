from setuptools import setup, find_packages

setup(
    name="dstt",
    version="3.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
