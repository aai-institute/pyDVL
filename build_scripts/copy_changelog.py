import logging
import os
import shutil
from pathlib import Path

import mkdocs.plugins

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent
docs_dir = root_dir / "docs"
changelog_file = root_dir / "CHANGELOG.md"
changelog_docs_file = docs_dir / changelog_file.name


@mkdocs.plugins.event_priority(100)
def on_pre_build(config):
    logger.info("Temporarily copying changelog to docs directory")
    try:
        if os.path.getmtime(changelog_file) <= os.path.getmtime(changelog_docs_file):
            logger.info(
                f"Changelog '{os.fspath(changelog_file)}' hasn't been updated, skipping."
            )
            return
    except FileNotFoundError:
        pass
    logger.info(
        f"Copying '{os.fspath(changelog_file)}' to '{os.fspath(changelog_docs_file)}'"
    )
    shutil.copy2(src=changelog_file, dst=changelog_docs_file)

    logger.info("Finished copying changelog to docs directory")


@mkdocs.plugins.event_priority(-100)
def on_shutdown():
    logger.info("Removing temporary changelog in docs directory")
    changelog_docs_file.unlink()
