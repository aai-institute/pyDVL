"""
This mkdocs hook replaces the binder link in the rendered notebooks
with links to the actual notebooks in the repository.
This is needed because for the docs we create symlinks to the notebooks
inside the docs directory.
This is heavily inspired from:
https://github.com/greenape/mknotebooks/blob/master/mknotebooks/plugin.py#L322
"""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from bs4 import BeautifulSoup
from git import Repo
from mkdocs.plugins import Config, event_priority

if TYPE_CHECKING:
    from mkdocs.plugins import Files, Page

logger = logging.getLogger("mkdocs")

BINDER_BASE_URL = "https://mybinder.org/v2"
BINDER_LOGO_WITH_CAPTION = "[![Binder](https://mybinder.org/badge_logo.svg)]"
BINDER_LOGO_WITHOUT_CAPTION = "[![](https://mybinder.org/badge_logo.svg)]"
BINDER_LINK_PATTERN = re.compile(
    re.escape(BINDER_LOGO_WITH_CAPTION) + r"\(" + re.escape(BINDER_BASE_URL) + r".*\)"
)

branch_name: Optional[str] = None


@event_priority(-50)
def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool) -> None:
    global branch_name
    try:
        branch_name = Repo().active_branch.name
        logger.info(f"Found branch name using git: {branch_name}")
    except TypeError:
        branch_name = os.getenv("GITHUB_REF", "develop").split("/")[-1]
        logger.info(f"Found branch name from environment variable: {branch_name}")


@event_priority(-50)
def on_page_content(
    html: str, page: "Page", config: Config, files: "Files"
) -> Optional[str]:
    if "examples" not in page.url:
        return
    logger.info(
        f"Replacing binder link with link to notebook in repository for notebooks in {page.url}"
    )

    repo_name = config["repo_name"]
    root_dir = Path(config["docs_dir"]).parent
    notebooks_dir = root_dir / "notebooks"
    notebook_filename = Path(page.file.src_path).name
    file_path = (notebooks_dir / notebook_filename).relative_to(root_dir)

    soup = BeautifulSoup(html, features="html.parser")
    binder_anchor = None
    for a in soup.find_all("a", href=True, limit=5):
        if BINDER_BASE_URL in a["href"]:
            binder_anchor = a
            break
    if binder_anchor is None:
        logger.warning(f"Binder link was not found in notebook {file_path}")
        return

    url_path = f"%2Ftree%2F{file_path}"
    binder_url = f"{BINDER_BASE_URL}/gh/{repo_name}/{branch_name}?urlpath={url_path}"
    logger.info(f"New binder url: {binder_url}")

    binder_anchor["href"] = binder_url
    binder_img = binder_anchor.find("img")
    binder_img["style"] = "margin: auto; display: block; width: 7rem"
    binder_img_caption = binder_anchor.find("figcaption")
    binder_img_caption.decompose()

    html = soup.prettify()

    return html
