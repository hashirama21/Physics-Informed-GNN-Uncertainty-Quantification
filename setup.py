from setuptools import setup, find_packages

setup(
    name="pignn-uq",
    version="1.0.0",
    description=(
        "Physics-Informed Graph Attention Network with Uncertainty Quantification "
        "for Power Transformer Fault Diagnosis via Dissolved Gas Analysis (DGA)"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Vincess Dongmo",
    author_email="sodiaque806@gmail.com",
    url="https://github.com/hashirama21/Physics-Informed-GNN-Uncertainty-Quantification",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*", "outputs*", "logs*", "data*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torch_geometric>=2.3.0",
        "openpyxl>=3.1.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "export": [
            "markdown>=3.5",
            "python-docx>=1.1",
            "weasyprint>=60.0",
        ],
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pignn-train=train:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    keywords=[
        "graph neural network", "transformer diagnostics", "dissolved gas analysis",
        "uncertainty quantification", "monte carlo dropout", "physics-informed",
        "GAT", "DGA", "IEC 60599", "IEEE C57.104",
    ],
)
