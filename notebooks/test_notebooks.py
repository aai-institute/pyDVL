import logging
import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

ROOT_DIR = Path(".").parent.parent
NOTEBOOKS_DIR = os.fspath(ROOT_DIR / "notebooks")
DOCS_NOTEBOOKS_DIR = os.fspath(ROOT_DIR / "docs" / "notebooks")
resources = {"metadata": {"path": NOTEBOOKS_DIR}}

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "notebook", [file for file in os.listdir(NOTEBOOKS_DIR) if file.endswith(".ipynb")]
)
def test_notebook(notebook):
    notebook_path = os.path.join(NOTEBOOKS_DIR, notebook)
    log.info(f"Reading jupyter notebook from {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, resource=resources)
    # HACK: this is needed because, for some reason, you cannot pass it to the init of ExecutePreprocessor
    ep.nb = nb
    # TODO: do we really need the log calls in this loop?
    #   If not then we could just simply call the `execute` method and get rid of the loop
    with ep.setup_kernel():
        for i, cell in enumerate(nb["cells"]):
            log.info(f"processing cell {i} from {notebook}")
            ep.preprocess_cell(cell, resources=resources, index=i)

    # saving the executed notebook to docs
    output_path = os.path.join(DOCS_NOTEBOOKS_DIR, notebook)
    log.info(f"Saving executed notebook to {output_path} for documentation purposes")
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
