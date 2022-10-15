import os
import shutil
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.util import logging

logger = logging.getLogger(__name__)


def copy_notebooks(app: Sphinx, config: Config) -> None:
    logger.info("Copying notebooks to examples directory")
    root_dir = Path(app.confdir).parent
    notebooks_dir = root_dir / "notebooks"
    docs_examples_dir = root_dir / "docs" / "examples"
    notebook_filepaths = list(notebooks_dir.glob("*.ipynb"))
    for notebook in notebook_filepaths:
        target_filepath = docs_examples_dir / notebook.name
        if os.path.getmtime(notebook) <= os.path.getmtime(target_filepath):
            logger.info(
                f"Notebook '{os.fspath(notebook)}' hasn't been updated, skipping."
            )
            continue
        logger.info(
            f"Copying '{os.fspath(notebook)}' to '{os.fspath(target_filepath)}'"
        )
        shutil.copyfile(src=notebook, dst=target_filepath)
    logger.info("Finished copying notebooks to examples directory")


def setup(app):
    app.connect("config-inited", copy_notebooks)
