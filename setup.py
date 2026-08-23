#!/usr/bin/env python
"""
PINN Fatigue Crack Growth Prediction - Setup Configuration
A physics-informed neural network for fatigue crack growth prediction.
"""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    """Read README.md file for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "PINN Fatigue Crack Growth Prediction - A physics-informed neural network for fatigue crack growth prediction."

# Read requirements
def read_requirements():
    """Read requirements.txt file."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name="pinn-fatigue-crack-growth",
    version="1.0.0",
    author="中科院计算机研究生",
    author_email="your-email@example.com",
    description="A physics-informed neural network for fatigue crack growth prediction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pinn-fatigue-crack-growth",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pinn-train=src.training.run_pinn_residual_attention:main",
            "pinn-plot=src.visualization.plot_best_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
    },
    keywords="pinne physics-informed neural networks fatigue crack growth prediction machine learning",
    project_urls={
        "Bug Reports": "https://github.com/your-username/pinn-fatigue-crack-growth/issues",
        "Source": "https://github.com/your-username/pinn-fatigue-crack-growth",
        "Documentation": "https://pinn-fatigue-crack-growth.readthedocs.io/",
    },
)