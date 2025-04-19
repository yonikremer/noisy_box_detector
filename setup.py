from setuptools import setup, find_packages

setup(
    name="rectangle_finder",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pydantic",
        "pyyaml",
        "opencv-python",
        "scipy",
    ],
    python_requires=">=3.8",
)
