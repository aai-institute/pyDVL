import logging
import os
from pathlib import Path

import mkdocs.plugins

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent
docs_dir = root_dir / "docs"
contributing_file = root_dir / "CONTRIBUTING.md"
target_filepath = docs_dir / contributing_file.name


@mkdocs.plugins.event_priority(100)
def on_pre_build(config):
    logger.info("Linking contributing guide to docs directory")
    try:
        target_filepath.symlink_to(contributing_file)
        logger.info(
            f"Created symbolic link for '{os.fspath(contributing_file)}' "
            f"at '{os.fspath(target_filepath)}'"
        )
    except FileExistsError:
        logger.info(
            f"File '{os.fspath(target_filepath)}' already exists, skipping symlink creation."
        )


@mkdocs.plugins.event_priority(-100)
def on_shutdown():
    pass  # Removing the link on shutdown makes mike fail the build
    # logger.info("Removing temporary contributing guide in docs directory")
    # target_filepath.unlink()
