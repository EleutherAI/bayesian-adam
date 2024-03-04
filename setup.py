from setuptools import setup, find_packages

setup(
    name='bayesadam',
    version='0.0.1',
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "torch"
    ],
    entry_points={
        'console_scripts': [
            'sampling_demo=demos.sampling_demo:main',
        ],
    },
)
