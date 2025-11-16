"""
Setup script for the datasetgen package.
"""
from setuptools import setup, find_packages
setup(
    name="DataOrchestra",
    version="0.0.1",
    author="Pratyay Mustafi",
    author_email="pratyaymustafi@outlook.com",
    description="A toolkit for cleaning and preparing text datasets for LLM training and finetuning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=[
        "argparse",
        "requests",
        "beautifulsoup4",
        "lxml",
        "python-docx",
        "pdf2image",
        "pytesseract",
        "Pillow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License 2.0",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires='>=3.10+',
    
)

