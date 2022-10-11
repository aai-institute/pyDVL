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
    sep = "\n\t"
    logger.info(
        f"Found the following notebooks: "
        f"{sep.join([str(p) for p in notebook_filepaths])}"
    )
    for notebook in notebook_filepaths:
        target_filepath = docs_examples_dir / notebook.name
        logger.info(
            f"Copying '{os.fspath(notebook)}' to '{os.fspath(target_filepath)}'"
        )
        shutil.copyfile(src=notebook, dst=target_filepath)
    logger.info("Finished copying notebooks to examples directory")


def setup(app):
    app.connect("config-inited", copy_notebooks)
