#!/usr/bin/env python3
"""
Setup script for QUEST (Quantitative Understanding of Epitope Specificity in T-cells)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "QUEST: Quantitative Understanding of Epitope Specificity in T-cells"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.7.0",
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "wandb",
        "evaluate",
        "ray[data,train,tune,serve]",
        "peft",
        "tqdm",
        "scikit-learn",
        "s3fs",
        "pandas",
        "numpy",
        "pyyaml",
    ]

setup(
    name="quest",
    version="0.1.0",
    author="QUEST Team",
    description="Quantitative Understanding of Epitope Specificity in T-cells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shashidhar22/quest",
    packages=find_packages(include=["quest", "quest.*", "scripts", "scripts.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "quest-train=scripts.training.ray_train:main",
            "quest-finetune=scripts.training.ray_fine_tune:main",
            "quest-evaluate=scripts.training.ray_evaluator:main",
            "quest-inference=scripts.inference.run_inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],
    },
)
