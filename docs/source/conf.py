# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "PeekingDuck"
copyright = "2021, CVHub AI Singapore"
author = "CVHub AI Singapore"

# The full version, including alpha/beta/rc tags
release = "developer"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
myst_heading_anchors = 2

napoleon_custom_sections = [
    ("Configs", "params_style"),
    ("Inputs", "returns_style"),
    ("Outputs", "returns_style"),
]


autosummary_generate = True
master_doc = "master"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
source_suffix = [".rst", ".md"]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ["_static"]
html_style = "css/pkdk.css"
html_logo = "../../images/readme/peekingduck.png"
html_sidebars = {"**": ["globaltoc.html", "searchbox.html"]}
html_theme_options = {"logo_only": True}

autosummary_mock_imports = [
    "peekingduck.pipeline.nodes.dabble.trackingv1",
    "peekingduck.pipeline.nodes.dabble.utils",
    "peekingduck.pipeline.nodes.dabble.zoningv1",
    "peekingduck.pipeline.nodes.draw.utils",
    "peekingduck.pipeline.nodes.input.utils",
    "peekingduck.pipeline.nodes.model.csrnetv1",
    "peekingduck.pipeline.nodes.model.efficientdet_d04",
    "peekingduck.pipeline.nodes.model.hrnetv1",
    "peekingduck.pipeline.nodes.model.jdev1",
    "peekingduck.pipeline.nodes.model.mtcnnv1",
    "peekingduck.pipeline.nodes.model.posenetv1",
    "peekingduck.pipeline.nodes.model.movenetv1",
    "peekingduck.pipeline.nodes.model.yolov4",
    "peekingduck.pipeline.nodes.model.yolov4_face",
    "peekingduck.pipeline.nodes.model.yolov4_license_plate",
    "peekingduck.pipeline.nodes.model.yoloxv1",
    "peekingduck.pipeline.nodes.output.utils",
    "peekingduck.utils",
]
