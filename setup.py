from os import read

from setuptools import setup
from setuptools import find_packages
# from tpu_server import version

PACKAGE_NAME = "pyjutsu"
VERSION_MINOR = 0
VERSION_MAJOR = 0

# with open("requirements.txt") as f:
#     requirements_lines = f.readlines()
#     requirements = [r for r in requirements_lines]
# 
# with open("readme.md") as f:
#     long_descr = f.read()

setup(
    name=PACKAGE_NAME,
    version=f"{VERSION_MAJOR}.{VERSION_MINOR}",
    packages=find_packages(
        where="src",
    ),
    package_dir={"": "src"},
    install_requires=[
        "pyyaml",
    ],
    entry_points={
        # "console_scripts": "tpus=tpu_server.app:app_main"
    },
    author="Stepan Dyatkovskiy",
    author_email="ml@dyatkovskiy.com",
    description="Top-level python utilities aimed to increase development efficiency.",
    license="GPL3",
    keywords="python boost utilities",
    # url="http://packages.python.org/an_example_pypi_project",
)