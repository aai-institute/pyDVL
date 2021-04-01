#!/usr/bin/env python3
import logging
import os

log = logging.getLogger(os.path.basename(__file__))


def module_template(module_path: str):
    title = os.path.basename(module_path).replace("_", r"\_")
    title = title[:-3]  # removing trailing .py
    module_path = module_path[:-3]
    template = f"""{title}
{"="*len(title)}

.. automodule:: {module_path.replace(os.path.sep, ".")}
   :members:
   :undoc-members:
"""
    return template


def package_template(package_path: str):
    package_name = os.path.basename(package_path)
    title = package_name.replace("_", r"\_")
    template = f"""{title}
{"="*len(title)}

.. automodule:: {package_path.replace(os.path.sep, ".")}
   :members:
   :undoc-members:

.. toctree::
   :glob:

   {package_name}/*
"""
    return template


def write_to_file(content: str, path: str):
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o666)


def make_docu(basedir=os.path.join("src", "valuation"), overwrite=False):
    """
    Creates/updates documentation in form of rst files for modules and packages.
    Does not delete any existing rst files. Thus, rst files for packages or modules that have been removed or renamed
    should be deleted by hand.

    This method should be executed from the project's top-level directory

    :param basedir: path to library basedir, typically "src/<library_name>"
    :param overwrite: whether to overwrite existing rst files. This should be used with caution as it will delete
        all manual changes to documentation files
    :return:
    """
    library_basedir = basedir.split(os.path.sep, 1)[1]  # splitting off the "src" part
    for file in os.listdir(basedir):
        if file.startswith("_"):
            continue

        library_file_path = os.path.join(library_basedir, file)
        full_path = os.path.join(basedir, file)
        file_name, ext = os.path.splitext(file)
        docs_file_path = os.path.join("docs", library_basedir, f"{file_name}.rst")
        if os.path.exists(docs_file_path) and not overwrite:
            log.debug(f"{docs_file_path} already exists, skipping it")
            if os.path.isdir(full_path):
                make_docu(basedir=full_path, overwrite=overwrite)
            continue
        os.makedirs(os.path.dirname(docs_file_path), exist_ok=True)

        if ext == ".py":
            log.info(f"writing module docu to {docs_file_path}")
            write_to_file(module_template(library_file_path), docs_file_path)
        elif os.path.isdir(full_path):
            log.info(f"writing package docu to {docs_file_path}")
            write_to_file(package_template(library_file_path), docs_file_path)
            make_docu(basedir=full_path, overwrite=overwrite)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_docu()
