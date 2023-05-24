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
            'sympy',
            ],
        author= ['Wataru MITA',
                'Johei MATSUOKA'
                'Kazuya TAGO'],
        author_email= 'c0a20147d6@edu.teu.ac.jp',
        url='https://github.com/RobotSpatialCognition/vector_map',

)
