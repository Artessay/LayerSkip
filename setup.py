from setuptools import setup, find_packages

setup(
    name="layerskip-eval",
    version="0.1.0",
    description="LM Evaluation Harness for comparing Layer Skipping Strategies",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=4.8.0,<5.0.0",
        "accelerate>=0.28.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    ],
    entry_points={
        "console_scripts": [
            "layerskip-eval=eval:main",
        ],
    },
)
