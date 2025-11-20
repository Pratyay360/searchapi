"""
Setup script for DataOrchestra package.
This file is provided for compatibility with older pip versions.
"""
from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of requirements from pyproject.toml equivalent
requirements = [
    "beautifulsoup4>=4.12.0",
    "python-docx>=0.8.11",
    "ddgs>=0.2.0",
    "pytesseract>=0.3.10",
    "pdf2image>=1.16.0",
    "Pillow>=8.0.0",
    "requests>=2.25.0",
    "tqdm>=4.62.0",
]

setup(
    name="DataOrchestra",
    version="0.1.0",
    author="Pratyay Mustafi",
    author_email="pratyaymustafi@outlook.com",
    description="A toolkit for cleaning and preparing text datasets for LLM training and finetuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pratyay360/DataOrchestra",
    project_urls={
        "Bug Tracker": "https://github.com/Pratyay360/DataOrchestra/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: General",
    ],
    package_dir={"": "."},
    packages=find_packages(where=".", include=["DataOrchestra*"]),
    python_requires=">=3.9",
    install_requires=requirements,
)