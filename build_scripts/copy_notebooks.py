import logging
import os
import shutil
from pathlib import Path

import mkdocs.plugins

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent
docs_examples_dir = root_dir / "docs" / "examples"
notebooks_dir = root_dir / "notebooks"


@mkdocs.plugins.event_priority(100)
def on_pre_build(config):
    logger.info("Temporarily copying notebooks to examples directory")
    docs_examples_dir.mkdir(parents=True, exist_ok=True)
    notebook_filepaths = list(notebooks_dir.glob("*.ipynb"))

    for notebook in notebook_filepaths:
        target_filepath = docs_examples_dir / notebook.name

        try:
            if os.path.getmtime(notebook) <= os.path.getmtime(target_filepath):
                logger.info(
                    f"Notebook '{os.fspath(notebook)}' hasn't been updated, skipping."
                )
                continue
        except FileNotFoundError:
            pass
        logger.info(
            f"Copying '{os.fspath(notebook)}' to '{os.fspath(target_filepath)}'"
        )
        shutil.copy2(src=notebook, dst=target_filepath)

    logger.info("Finished copying notebooks to examples directory")


@mkdocs.plugins.event_priority(-100)
def on_shutdown():
    logger.info("Removing temporary examples directory")
    for notebook_file in docs_examples_dir.glob("*.ipynb"):
        notebook_file.unlink()
