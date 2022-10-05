from pathlib import Path

from setuptools import find_packages, setup

# read the contents of README file
repository_root = Path(__file__).parent
long_description = (repository_root / "README.md").read_text()


setup(
    name="pyDVL",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="0.1.0-dev23",
    description="The python Data Valuation Library",
    install_requires=[
        line
        for line in open("requirements.txt").readlines()
        if not line.startswith("--")
    ],
    setup_requires=["wheel"],
    tests_require=["pytest"],
    extras_require={
        "influence": ["torch"],
    },
    author="appliedAI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files=("LICENSE.md",),
)
