"""Setup script for SBI Ensemble Diagnostics."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sbi-ensemble-diagnostics",
    version="1.0.0",
    author="James Alvey, Carlo R. Contaldi, Mauro Pieroni",
    author_email="jbga2@cam.ac.uk",
    description="Ensemble learning diagnostics for Simulation-Based Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sbi-ensemble-diagnostics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "corner",
        "imageio",
        "scipy>=1.7.0",
        "sbi==0.23.3",
        "sbibm==1.0.6",
        "tqdm",
    ],
)
