from setuptools import setup, find_packages

setup(
    name="signal_generation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
    ],
    python_requires=">=3.8",
) 