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
sys.path.insert(0, os.path.abspath("../../peekingduck/nodes"))


# -- Project information -----------------------------------------------------

project = "PeekingDuck"
copyright = "2022, CVHub AI Singapore"
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
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinxcontrib.jquery",
]
myst_heading_anchors = 2

napoleon_custom_sections = [
    ("Configs", "params_style"),
    ("Inputs", "returns_style"),
    ("Outputs", "returns_style"),
]

add_module_names = False
autosummary_generate = True
master_doc = "master"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**/_template_*.rst"]
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
html_logo = "assets/peekingduck.png"
html_sidebars = {"**": ["globaltoc.html", "searchbox.html"]}
html_theme_options = {"logo_only": True}
html_js_files = ["js/pkdk.js"]

autosummary_mock_imports = [
    "dabble.statisticsv1",
    "dabble.trackingv1",
    "dabble.utils",
    "dabble.zoningv1",
    "draw.utils",
    "input.utils",
    "model.csrnetv1",
    "model.efficientdet_d04",
    "model.fairmotv1",
    "model.hrnetv1",
    "model.huggingface_hubv1",
    "model.jdev1",
    "model.mask_rcnnv1",
    "model.mediapipe_hubv1",
    "model.mtcnnv1",
    "model.posenetv1",
    "model.movenetv1",
    "model.yolact_edgev1",
    "model.yolov4",
    "model.yolov4_face",
    "model.yolov4_license_plate",
    "model.yoloxv1",
    "output.utils",
]
