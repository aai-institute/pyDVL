# valuation development guide

This repository contains the valuation python library together with utilities
for building, testing, documentation and configuration management.

## Local Development

This project uses the [black](https://github.com/psf/black) source code formatter
and [pre-commit](https://pre-commit.com/) to invoke it as a Git prme-commit hook.

When first cloning the repository, run the following command (after
setting up your virtualenv with dev dependencies installed, see below) to set up
the local Git hook:

```shell script
pre-commit install
```

Automated builds, tests, generation of docu and publishing are handled by CI/CD
pipelines. You will find an initial version of the pipeline in this repo. Below
are further details on testing and documentation.

Before pushing your changes to the remote it is often useful to execute `tox`
locally in order to detect mistakes early on.

We strongly suggest using some form of virtual environment for working with the
library. E.g. with venv (if you have created the project locally with the
python-library-template, it will already include a venv)

```shell script
python -m venv ./venv
. venv/bin/activate  # `venv\Scripts\activate` in windoze
```

or conda:

```shell script
conda create -n valuation python=3.8
conda activate valuation
```

A very convenient way of working with your library during development is to
install it in editable mode into your environment by running

```shell script
pip install -e .
```


### Additional requirements

The main requirements for developing the library locally are in
`requirements-dev.txt`. For building documentation locally (which is done as
part of the tox suite) you will need pandoc. It can be installed e.g. via

```shell script
sudo apt-get update -yq && apt-get install -yq pandoc
```

### Testing and packaging

The library is built with tox which will build and install the package, run the
test suite and build the documentation. Running tox will also generate coverage
and pylint reports in html and badges. You can configure pytest, coverage and
pylint by adjusting [pytest.ini](pytest.ini), [.coveragerc](.coveragerc) and
[.pylintrc](.pylintrc) respectively.

Concerning notebooks: all notebooks in the [notebooks](notebooks) directory should
be executed during test run, the results will be added to the documentation in the
_Guides and Tutorials_ section. Thus, notebooks can be conveniently used as integration tests and documentation at the same time.

Unfortunately that is not currently the case as they need to be fixed and updated.

#### Testing

You can run the build by installing tox into your virtual environment (e.g. with
`pip install tox`) and executing `tox`.

You can pass optional command line arguments to pytest, for example to run only certain tests using patterns (-k) or marker (-m).

`tox -e py38 -- <optional arguments>`

One important argument that you can use is `--do-not-start-memcache`. This prevents
the test fixture from starting a new memcache server for testing and instead expects
an already running local server listening on port 11211 ( memcache's default port ).

To test notebooks, you should use:

```shell
tox -e notebooks
```

#### Packaging

To create a package locally, run
```shell script
python setup.py sdist bdist_wheel
```

### Documentation

Documentation is built with sphinx every time tox is executed, doctests are run
during that step. There is a helper script for updating documentation files
automatically. It is called by tox on build and can be invoked manually as

```bash
python build_scripts/update_docs.py
```
See the code documentation in the script for more details on that.

Notebooks also form part of the documentation, see the explanation above.


## CI/CD and Release Process

This repository contains ci/cd pipelines for multiple providers. The most
sophisticated one is the [gitlab ci pipeline](.gitlab-ci.yml) (this is what we
use internally at appliedAI), it will run the test suite and publish docu,
badges and reports. Badges can accessed from the pipeline's artifacts, e.g. for
the coverage badge the url will be:

```
<gitlab_project_url>/-/jobs/artifacts/develop/raw/badges/coverage.svg?job=tox_use_cache
```

The github and azure ci pipelines are rather rudimentary. Pull requests are
always welcome!

### Development and Release Process with Github

In order to be able to automatically release new versions of the package from
the develop branch, the CI pipeline should have access to the following
secret variables:

```
TEST_PYPI_USERNAME
TEST_PYPI_PASSWORD
PYPI_USERNAME
PYPI_PASSWORD
```

The first 2 are used after tests run on the develop branch's CI workflow 
to automatically publish packages to [TestPyPI](https://test.pypi.org/).

The last 2 are used in the [publish.yaml](.github/workflows/publish.yaml) CI workflow
to publish packages to [PyPI](https://pypi.org/) from the develop after a Github release creation.

#### Automatic release process

In order to create an automatic release, a few prerequisites need to be
satisfied:

- The project's virtualenv needs to be active
- The repository needs to be on the `develop` branch
- The repository must be clean (including no untracked files)

Then, a new release can be created using the `build_scripts/release-version.sh`
script (leave off the version parameter to have `bumpversion` automatically
derive the next release version):

```shell script
./scripts/release-version.sh 0.1.6
```

To find out how to use the script, pass the `-h` or `--help` flags:

```shell script
./build_scripts/release-version.sh --help
```

If running in interactive mode (without `-y|--yes`), the script will output a
summary of pending changes and ask for confirmation before executing the
actions.

#### Manual release process
If the automatic release process doesn't cover your use case, you can also
create a new release manually by following these steps:

1. (Repeat as needed) implement features on feature branches merged into
  `develop`. Each merge into develop will advance the `.devNNN` version suffix
   and publish the pre-release version into the package registry. These versions
   can be installed using `pip install --pre`.
2. When ready to release: From the develop branch create the release branch and
   perform release activities (update changelog, news, ...). For your own
   convenience, define an env variable for the release version
    ```shell script
    export RELEASE_VERSION="vX.Y.Z"
    git checkout develop
    git branch release/${RELEASE_VERSION} && git checkout release/${RELEASE_VERSION}
    ```
3. Run `bumpversion --commit release` if the release is only a patch release,
   otherwise the full version can be specified using 
   `bumpversion --commit --new-version X.Y.Z release`
   (the `release` part is ignored but required by bumpversion :rolling_eyes:).
4. Merge the release branch into `master`, tag the merge commit, and push back to the repo. 
   The CI pipeline publishes the package based on the tagged commit.
    ```shell script
    git checkout master
    git merge --no-ff release/${RELEASE_VERSION}
    git tag -a ${RELEASE_VERSION} -m"Release ${RELEASE_VERSION}"
    git push --follow-tags origin master
    ```
5. Switch back to the release branch `release/vX.Y.Z` and pre-bump the version:
   `bumpversion --commit patch`. This ensures that `develop` pre-releases are
   always strictly more recent than the last published release version from 
   `master`.
6. Merge the release branch into `develop`:
    ```shell script
    git checkout develop
    git merge --no-ff release/${RELEASE_VERSION}
    git push origin develop
    ```
6. Delete the release branch if necessary: 
   `git branch -d release/${RELEASE_VERSION}`
7. Pour yourself a cup of coffee, you earned it! :coffee: :sparkles:

## Useful information

Mark all autogenerated directories as excluded in your IDE. In particular
docs/_build and .tox should be marked as excluded in order to get a significant
speedup in searches and refactorings.

If using remote execution, don't forget to exclude data paths from deployment
(unless you really want to sync them)
