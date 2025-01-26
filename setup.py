'''
The setup.py file is essential part of packaging and distributing Python projects.
It is used by setuptools (or distutils in older Python versions) to define the configuration
of the project, such as its metadata, dependencies, and more
'''
#find_package function will scan through all the folder and whenever there is an __init__.py file, it is going to consider that file as a package.
#setup function is responsible to provide all the information about the project

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return list of requirements

    """
    requirement_list: List[str] = [] 
    try:
        with open('requirements.txt', 'r') as file:
            #Read lines from the file
            lines = file.readlines()
            # Process each line
            for line in lines:
                requirement = line.strip()
                # ignore empty lines and -e .
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_list

setup(
    name="FoodDeliveryTime",
    version="0.0.1",
    author="Anuj Pandey",
    author_email="anuzj007@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)