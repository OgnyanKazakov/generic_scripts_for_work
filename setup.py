from package import Package
from setuptools import find_packages, setup


setup(
    author="Petar Nikolov",
    author_email="petar.nikolov@resolve.io",
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        "package": Package
    }
)
