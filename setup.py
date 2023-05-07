from setuptools import setup

install_requires = ["setuptools", "pydot"]

setup(
        name = 'vector_map'
        version = '0.1'
        package = find_package()
        install_requires = [
            'numpy',
            'opencv',
            'opencv-contrib',
            ]
)


            
