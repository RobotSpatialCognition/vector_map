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
        keywords=['vectorize',
                    'SLAM',
                    'Action Planning',
                    'ROS' ],
        classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries'
        ],
        description=(
            "A tool for generating vectorized map from SLAM output."
        ),
        long_description=(
            "A tool for generating vectorized map from SLAM output."
            "The vectorized map consists from a set of geometric components "
            "such as straight lines and curves, and is suitable for "
            "action planning of robots."
        ),
        license='MIT',

)
