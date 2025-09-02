from setuptools import setup, find_packages

setup(
    name="moska-mcts",  # Name of your package
    version="1.0.0",     # Version of your package
    description="A card game engine for Moska with AI players (MCTS, heuristic, and neural network-based).",
    author="Arno Törö",
    url="https://github.com/arnotoro/moska",
    packages=find_packages(),  # Automatically find all packages inside your package directory
    include_package_data=True,  # Includes files specified in MANIFEST.in (if any)
    install_requires=[  # List any external packages your package needs
        "torch>=2.8.0",
        "numpy>=2.3.2",
        "scikit-learn>=1.7.1",
        "tqdm>=4.67.1",
        "pandas>=2.3.2",
        "pyarrow>=21.0.0",
        "matplotlib>=3.10.6"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Minimum Python version
)