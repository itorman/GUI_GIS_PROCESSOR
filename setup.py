#!/usr/bin/env python3
"""
Setup script for GIS Document Processing Application
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="gis-document-processor",
    version="1.0.0",
    author="GIS Development Team",
    author_email="team@gis-dev.com",
    description="A comprehensive Python application for extracting addresses and geographic coordinates from documents using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/gis-document-processor",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/gis-document-processor/issues",
        "Documentation": "https://github.com/your-org/gis-document-processor/wiki",
        "Source Code": "https://github.com/your-org/gis-document-processor",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="gis, document-processing, llm, address-extraction, geocoding, arcgis, shapefile",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "geopandas>=0.12.0",
            "arcpy",  # Requires ArcGIS Pro
            "pytesseract>=0.3.10",
            "pdf2image>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gis-doc-processor=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    platforms=["any"],
) 