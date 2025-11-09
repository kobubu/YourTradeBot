from setuptools import setup, find_packages

setup(
    name="yourtradebot",
    version="0.1.0",
    packages=find_packages(include=['core', 'core.*']),  # явно указываем пакеты
    install_requires=[
        # добавьте сюда зависимости из requirements.txt
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "python-telegram-bot",
        # и т.д.
    ],
    python_requires='>=3.8',
)