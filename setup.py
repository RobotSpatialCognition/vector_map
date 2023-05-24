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
            "A development tool to create ROS package based on "
            "behaviour trees framework."
        ),
        long_description=(
            "A development tool to create ROS package based on "
            "behaviour trees framework."
        ),
        license='MIT',

)
