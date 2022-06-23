from setuptools import find_packages, setup

test_requirements = ["pytest"]

setup(
    name="pyDVL",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="0.1.0-dev1",
    description="The python Data Valuation Library",
    install_requires=[
        line
        for line in open("requirements.txt").readlines()
        if not line.startswith("--")
    ],
    setup_requires=["wheel"],
    tests_require=test_requirements,
    author="appliedAI",
)
