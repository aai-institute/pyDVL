import shutil
from pathlib import Path
from typing import List

from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

logger = logging.getLogger(__name__)


def copy_notebooks(app: Sphinx, env: BuildEnvironment, docnames: List[str]) -> None:
    logger.info("Copying notebooks to examples directory")
    root_dir = Path(app.confdir).parent
    notebooks_dir = root_dir / "notebooks"
    docs_examples_dir = root_dir / "docs" / "examples"
    notebook_filepaths = list(notebooks_dir.glob("*.ipynb"))
    logger.info(f"Found following notebooks: {notebook_filepaths}")
    for notebook in notebook_filepaths:
        shutil.copy(src=notebook, dst=docs_examples_dir)
    logger.info("Finished copying notebooks to examples directory")


def setup(app):
    app.connect("env-before-read-docs", copy_notebooks)
