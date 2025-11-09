from setuptools import find_packages, setup

setup(
    name="yourtradebot",
    version="0.1.0",
    packages=find_packages(include=['core', 'core.*']), 
    install_requires=[
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "python-telegram-bot",
        # и т.д.
    ],
    python_requires='>=3.8',
)