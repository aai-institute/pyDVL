from pathlib import Path

from setuptools import find_packages, setup

# read the contents of README file
repository_root = Path(__file__).parent
long_description = (repository_root / "README.md").read_text()

setup(
    name="pyDVL",
    package_dir={"": "src"},
    package_data={"pydvl": ["py.typed"]},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="0.7.1",
    description="The Python Data Valuation Library",
    install_requires=[
        line
        for line in open("requirements.txt").readlines()
        if not line.startswith("--")
    ],
    setup_requires=["wheel"],
    tests_require=["pytest"],
    extras_require={
        "cupy": ["cupy-cuda11x>=12.1.0"],
        "influence": ["torch>=2.0.0"],
        "ray": ["ray>=0.8"],
    },
    author="appliedAI Institute gGmbH",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files=("LICENSE", "COPYING.LESSER"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Typing :: Typed",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    project_urls={
        "Source": "https://github.com/aai-institute/pydvl",
        "Documentation": "https://aai-institute.github.io/pyDVL",
        "TransferLab": "https://transferlab.appliedai.de",
    },
    zip_safe=False,  # Needed for mypy to find py.typed
)
