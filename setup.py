from setuptools import setup, find_packages

setup(
    name="vocalbaby",
    version="0.1.0",
    description="A CLI tool for infant vocalization classification using Wav2Vec2 and prosodic features",
    author="Arun Singh",
    author_email="arunps@uio.no", 
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "librosa",
        "audiomentations",
        "datasets",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "transformers",
        "accelerate"
    ],
    entry_points={
        "console_scripts": [
            "vocalbaby=vocalbaby.vocalbaby:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
