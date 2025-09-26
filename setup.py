"""
Setup script for SSEL-CPP
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ssel-cpp",
    version="1.0.0",
    author="Chanwoo Kwon",
    author_email="cwkwon99@ajou.ac.kr",
    description="A machine learning tool for predicting Cell Penetrating Peptide activity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChanwooKwon/SSELCPP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "": ["models/*.pkl", "models/*.csv"],
    },
    entry_points={
        "console_scripts": [
            "ssel-cpp=test:main",
        ],
    },
)