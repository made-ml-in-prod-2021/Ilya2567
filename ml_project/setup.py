from setuptools import find_packages, setup


def parse_requirements(path):
    with open(path) as fin:
        requirements = []
        for line in fin:
            requirements.append(line.strip())
    return requirements


setup(
    name="homework1",
    packages=find_packages(),
    version="1.0.0",
    description="Homework1 on ML in production",
    author="Avilov Ilya",
    install_requires=parse_requirements("requirements.txt"),
    license="MIT",
)
