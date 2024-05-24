from setuptools import setup, find_packages

setup(
    name='bayesadam',
    version='0.0.1',
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "torch",
    ],
    extras_require={ 'demos': [
            "jupyterlab",
            "matplotlib",
            "seaborn",
            "numpy",
            "x-transformers"
        ]
    },
    entry_points={
        'console_scripts': [
            'sampling_demo=demos.sampling_demo:main',
        ],
    },
)

