#!/usr/bin/env python3
"""
This script walks through the python source files and creates documentation in .rst format which can
then be compiled with Sphinx. It is suitable for a standard repository layout src/<library_name> as well as for
a repo containing multiple packages src/<package_1>, ...,  src/<package_n>.
"""


import logging
import os
import shutil

log = logging.getLogger(os.path.basename(__file__))


def module_template(module_qualname: str):
    module_name = module_qualname.split(".")[-1]
    title = module_name.replace("_", r"\_")
    template = f"""{title}
{"="*len(title)}
.. automodule:: {module_qualname}
   :members:
   :undoc-members:
"""
    return template


def package_template(package_qualname: str):
    package_name = package_qualname.split(".")[-1]
    title = package_name.replace("_", r"\_")
    template = f"""{title}
{"="*len(title)}
.. automodule:: {package_qualname}
   :members:
   :undoc-members:

.. toctree::
   :glob:

   {package_name}/*
"""
    return template


def index_template(package_name):
    title = package_name.replace("_", r"\_")
    template = f"""{title}
{"="*len(title)}
.. automodule:: {package_name}
   :members:
   :undoc-members:

.. toctree::
   :glob:

   *
"""
    return template


def write_to_file(content: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o666)


def make_rst(src_root="src", docs_root="docs", clean=False, overwrite=False):
    """
    Creates/updates documentation in form of rst files for modules and packages.
    Does not delete any existing rst files if clean and overwrite are False.
    This method should be executed from the project's top-level directory
    :param src_root: path to project's src directory that contains all packages, usually src. Most projects will
        only need one top-level package, then your layout typically should be src/<library_name>.
    :param docs_root: path to the project's docs directory containing the conf.py and the top level index.rst
    :param clean: whether to completely clean the docs target directories beforehand, removing any existing files
    :param overwrite: whether to overwrite existing rst files. This should be used with caution as it will delete
        all manual changes to documentation files
    :return:
    """
    docs_root = os.path.abspath(docs_root)
    src_root = os.path.abspath(src_root)

    for top_level_package_name in os.listdir(src_root):
        top_level_package_dir = os.path.join(src_root, top_level_package_name)
        # skipping things in src that are not packages, like .egg files
        if (
            not os.path.isdir(top_level_package_dir)
            or "." in top_level_package_name
            or top_level_package_name.startswith("_")
        ):
            continue

        log.info(f"Generating docu for top-level package {top_level_package_name}")
        top_level_package_docs_dir = os.path.join(docs_root, top_level_package_name)
        if clean and os.path.isdir(top_level_package_docs_dir):
            log.info(f"Deleting {top_level_package_docs_dir} since clean=True")
            shutil.rmtree(top_level_package_docs_dir)

        index_rst_path = os.path.join(docs_root, top_level_package_name, "index.rst")
        log.info(f"Creating {index_rst_path}")
        write_to_file(index_template(top_level_package_name), index_rst_path)

        for root, dirnames, filenames in os.walk(top_level_package_dir):
            if os.path.basename(root).startswith("_"):
                log.debug(f"Skipping docu generation in {root}")
                continue

            base_package_relpath = os.path.relpath(root, start=top_level_package_dir)
            base_package_qualname = os.path.relpath(root, start=src_root).replace(
                os.path.sep, "."
            )

            for dirname in dirnames:
                if not dirname.startswith("_"):
                    package_qualname = f"{base_package_qualname}.{dirname}"
                    package_rst_path = os.path.abspath(
                        os.path.join(
                            top_level_package_docs_dir,
                            base_package_relpath,
                            f"{dirname}.rst",
                        )
                    )
                    log.info(f"Writing package documentation to {package_rst_path}")
                    write_to_file(package_template(package_qualname), package_rst_path)

            for filename in filenames:
                base_name, ext = os.path.splitext(filename)
                if ext == ".py" and not filename.startswith("_"):
                    module_qualname = f"{base_package_qualname}.{filename[:-3]}"
                    module_rst_path = os.path.abspath(
                        os.path.join(
                            top_level_package_docs_dir,
                            base_package_relpath,
                            f"{base_name}.rst",
                        )
                    )
                    if os.path.exists(module_rst_path) and not overwrite:
                        log.debug(f"{module_rst_path} already exists, skipping it")

                    log.info(f"Writing module documentation to {module_rst_path}")
                    write_to_file(module_template(module_qualname), module_rst_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_rst(clean=True)
