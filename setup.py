from setuptools import setup

install_requires = ["setuptools", "pydot"]

setup(
        name = 'vector_map',
        version = '0.1',
        packages=['vector_map'],
        install_requires = [
            'numpy',
            'opencv-python',
            'opencv-contrib-python',
            ]
)
