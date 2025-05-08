from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="velora",
    version="0.1.0",
    author="Tu Nombre",
    author_email="tu.email@ejemplo.com",
    description="VELORA: Aprendizaje Experto Versátil para Razonamiento y Análisis Operacional",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/velora",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "transformers>=4.28.0",
        "tokenizers>=0.13.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
        ],
        "training": [
            "wandb>=0.15.0",
            "accelerate>=0.18.0",
            "deepspeed>=0.8.0",
        ],
    },
)