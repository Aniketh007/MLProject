from setuptools import find_packages,setup
from typing import List

def get_requirements(file:str)-> List[str]:
    requirements=[]
    with open(file) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name='MLProject',
    version='0.0.1',
    author='Aniketh Reddy Adireddy',
    author_email='anikethadireddy@gmail.com',
    packages=find_packages(where="src"),
    install_requires=get_requirements('requirements.txt'),
    package_dir={'': 'src'}
)