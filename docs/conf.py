# -*- coding: utf-8 -*-
#
# pyDVL documentation build configuration file
#
# This file is execfile()d with the current directory set to its containing dir.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import ast
import logging
import os
import sys
from pathlib import Path

import pkg_resources

logger = logging.getLogger("docs")

ROOT_DIR = Path(__file__).resolve().parent.parent

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.fspath(ROOT_DIR / "src"))

# For custom extensions
sys.path.append(os.path.abspath("_ext"))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx_math_dollar",
    "sphinx.ext.todo",
    "hoverxref.extension",  # This only works on read the docs
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    # see https://github.com/spatialaudio/nbsphinx/issues/24 for an explanation why this extension is necessary
    "IPython.sphinxext.ipython_console_highlighting",
    # Custom extensions
    "copy_notebooks",
]

# sphinx_math_dollar
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    }
}

extlinks_detect_hardcoded_links = True
extlinks = {
    "gh": ("https://github.com/aai-institute/pyDVL/%s", "GitHub %s"),
    "issue": ("https://github.com/aai-institute/pyDVL/issues/%s", "issue %s"),
    "tfl": ("https://transferlab.appliedai.de/%s", "%s"),
}

bibtex_bibfiles = ["pydvl.bib"]
bibtex_bibliography_header = "References\n=========="
bibtex_footbibliography_header = bibtex_bibliography_header

# NBSphinx

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=False).replace('examples', 'notebooks') %}

.. raw:: html

    <div class="admonition note">
        This page was generated from
        <a class="reference external" href="https://github.com/aai-institute/pyDVL/blob/develop/{{ docname|e }}">{{ docname|e }}</a>
        <br>
        Interactive online version:
        <span style="white-space: nowrap;">
            <a href="https://mybinder.org/v2/gh/aai-institute/pyDVL/develop?filepath={{ docname|e }}">
                <img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom">
            </a>
        </span>
    </div>

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
        @media not print {
            [data-theme='dark'] .output_area img {
                filter: invert(0.9);
            }
            @media (prefers-color-scheme: dark) {
                :root:not([data-theme="light"]) .output_area img {
                    filter: invert(0.9);
                }
            }
        }
    </style>
"""

# Display todos by setting to True
todo_include_todos = True


# adding links to source files (this works for gitlab and github like hosts and might need to be adjusted for others)
# see https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html#module-sphinx.ext.linkcode
def linkcode_resolve(domain, info):
    link_prefix = "https://github.com/aai-institute/pyDVL/blob/develop"
    if domain != "py":
        return None
    if not info["module"]:
        return None

    path, link_extension = get_path_and_link_extension(info["module"])
    object_name = info["fullname"]
    if (
        "." in object_name
    ):  # don't add source link to methods within classes (you might want to change that)
        return None
    lineno = lineno_from_object_name(path, object_name)
    return f"{link_prefix}/{link_extension}#L{lineno}"


def get_path_and_link_extension(module: str):
    """
    :return: tuple of the form (path, link_extension) where
        the first entry is the local path to a given module or to __init__.py of the package
        and the second entry is the corresponding path from the top level directory
    """
    filename = module.replace(".", "/")
    docs_dir = os.path.dirname(os.path.realpath(__file__))
    source_path_prefix = os.path.join(docs_dir, f"../src/{filename}")

    if os.path.exists(source_path_prefix + ".py"):
        link_extension = f"src/{filename}.py"
        return source_path_prefix + ".py", link_extension
    elif os.path.exists(os.path.join(source_path_prefix, "__init__.py")):
        link_extension = f"src/{filename}/__init__.py"
        return os.path.join(source_path_prefix, "__init__.py"), link_extension
    else:
        raise Exception(
            f"{source_path_prefix} is neither a module nor a package with init - "
            f"did you forget to add an __init__.py?"
        )


def lineno_from_object_name(source_file, object_name):
    desired_node_name = object_name.split(".")[0]
    with open(source_file, "r") as f:
        source_node = ast.parse(f.read())
    desired_node = next(
        (
            node
            for node in source_node.body
            if getattr(node, "name", "") == desired_node_name
        ),
        None,
    )
    if desired_node is None:
        logger.warning(f"Could not find object {desired_node_name} in {source_file}")
        return 0
    else:
        return desired_node.lineno


# this is useful for keeping the docs build environment small. Add heavy requirements here
# and all other requirements to docs/requirements.txt
autodoc_mock_imports = ["ray"]

autodoc_default_options = {
    "exclude-members": "log",
    "member-order": "bysource",
    "show-inheritance": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "pyDVL"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
version = pkg_resources.get_distribution(project).version
release = version
# The short X.Y version.
major_v, minor_v = version.split(".")[:2]
version = f"{major_v}.{minor_v}"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------

# Add a tooltip to all :ref: roles
# This requires a backend server to retrieve the tooltip content. As of Nov 22,
# sphinx-hoverxref only supports Read the Docs as backend.
# See https://sphinx-hoverxref.readthedocs.io/en/latest/configuration.html
# for further configuration options
# hoverxref_auto_ref = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"

# Furo theme options:
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "announcement": "pyDVL is in an early stage of development. Expect changes to functionality and the API until version 1.0.0.",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/aai-institute/pyDVL",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = os.fspath(ROOT_DIR.joinpath("logo.svg"))

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True
copyright = "AppliedAI Institute gGmbH"

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "pydvl_doc"


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
# latex_documents = []

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        "index",
        "pydvl",
        "",
        ["appliedAI"],
        1,
    )
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
# texinfo_documents = []

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'
