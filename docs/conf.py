# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import shutil
import sys

# We need to "trick" sphinx due to it thinking that decorated classes are just aliases
# We thus need to import and later define some specific names
from baybe.surrogates import get_available_surrogates

# -- Path setup --------------------------------------------------------------

__location__ = os.path.dirname(__file__)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# Seems to be not necessary at the moment
# sys.path.insert(0, os.path.join(__location__, "../examples"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BayBE"
copyright = "2022-2023 Merck KGaA, Darmstadt, Germany and/or its affiliates. All rights reserved."  # noqa
author = "Martin Fitzner, Adrian Šošić, Alexander Hopp, Alex Lee"


# -- Run sphinx-apidoc -------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html . _build/html`. See Issue:
# https://github.com/readthedocs/readthedocs.org/issues/1139
# DON'T FORGET: Check the box "Install your project inside a virtualenv using
# setup.py install" in the RTD Advanced Settings.
# Additionally it helps us to avoid running apidoc manually

try:  # for Sphinx >= 1.7
    from sphinx.ext import apidoc
except ImportError:
    from sphinx import apidoc

output_dir = os.path.join(__location__, "sdk")
baybe_module_dir = os.path.join(__location__, "../baybe")
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass

try:
    args = [
        "--implicit-namespaces",
        "-M",
        "-T",
        "-e",
        "-f",
        "-o",
        output_dir,
    ]

    apidoc.main(
        [
            *args,
            baybe_module_dir,
            baybe_module_dir + "/__init__.py",
        ]
    )
except Exception as e:
    print(f"Running `sphinx-apidoc` failed!\n{e}")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Note that the order here is important! E.g. napoleon needs to be loaded before
# autodoc_typehints
extensions = [
    "sphinx.ext.napoleon",  # Necessary for numpy/google docstrings to work
    "sphinx_autodoc_typehints",  # Proper typehints
    "sphinx.ext.autodoc",  # Crucial for autodocumentation
    "sphinx.ext.autosummary",  # Autosummary
    "sphinx.ext.intersphinx",  # Links to other documentations like numpy, python,...
    "sphinx.ext.viewcode",  # Links to code
]
myst_enable_extensions = ["dollarmath"]  # Enables Latex-like math in markdown files


# Tell sphinx where to find the templates
templates_path = ["templates"]
# Tell sphinx which files should be excluded
exclude_patterns = ["sdk"]

# Enable markdown
# Note that we do not need additional configuration here.
extensions.append("myst_parser")

# The suffix of source filename
source_suffix = [".rst", ".md"]

# Here, we define regex expressions for errors produced by nitpick that we want to
# ignore.
nitpick_ignore_regex = [
    # Ignore everything that does not include baybe
    (r"py:.*", r"^(?!.*baybe).*"),
    # Ignore errors that are from inherited classes we cannot control
    (r"py:.*", r".*DTypeFloatNumpy.*"),
    (r"py:.*", r".*DTypeFloatONNX.*"),
    (r"py:.*", r".*AdapterModel.*"),
    # Ignore the functions that we manually delete from in child classes
    (r"py:.*", r".*from_dict.*"),
    (r"py:.*", r".*from_json.*"),
    (r"py:.*", r".*to_dict.*"),
    (r"py:.*", r".*to_json.*"),
    (r"py:.*", r".*_T.*"),
    # Ignore files for which no __init__ is available at all
    (r"py:.*", "baybe.constraints.conditions.Condition.__init__"),
    (r"py:.*", "baybe.utils.serialization.SerialMixin.__init__"),
    (r"DeprecationWarning:", ""),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# We use the read-the-docs theme
html_theme = "furo"
# We want to have links to the source
html_show_sourcelink = True
# Use the provided html theme
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.ico"
# Change the colors. The following colors are used:
BROWN = "#ffdcb9"
DARK_BLUE = "#0f69af"
LIGHT_BLUE = "#65bbca"
DARK_GRAY = "#212723"  # The dark gray of Githubs Dimmed Theme
LIGHT_GRAY = "#2c333b"  # The dark gray of Githubs Dimmed Theme
html_theme_options = {
    # We do not want the name of the package shown separately in the left sidebar
    "sidebar_hide_name": True,
    # Colors for light mode. For details on the keys, see colors for dark mode.
    "light_css_variables": {
        "color-background-secondary": DARK_BLUE,
        "color-background-border": "transparent",
        # Colors of headings in left sidebar
        # "color-foreground-primary": "white",
        "color-brand-primary": BROWN,
        "color-brand-secondary": BROWN,
        # "Muted" text
        "color-foreground-muted": "white",
        "color-highlight-on-target": BROWN,
        "color-api-background": BROWN,
        "color-api-pre-name": LIGHT_BLUE,
        "color-api-name": DARK_BLUE,
        "color-toc-item-text": "black",
        # Background of code in text
        "color-inline-code-background": "#efeff4",  # Very light gray
        # Color of search bar when clicking search
        "color-sidebar-search-background--focus": DARK_BLUE,
        "color-link": DARK_BLUE,
        # Color when hovering over the item expander in the left sidebar.
        "color-sidebar-item-expander-background--hover": LIGHT_BLUE,
        # Color when hovering over an item in the left sidebar
        "color-sidebar-item-background--hover": LIGHT_BLUE,
        # Color of the links in the left side bar
        "color-sidebar-link-text": "white",
        # Color of text in the right navigation side bar when chosen
        "color-toc-item-text--active": DARK_BLUE,
        # The color around the search bar
        "color-sidebar-search-border": BROWN,
        # Color of the search text and icon
        "color-sidebar-search-text": "white",
        "color-sidebar-search-icon": BROWN,
    },
    # Colors for dark mode.
    "dark_css_variables": {
        # Background of content
        "color-background-primary": LIGHT_GRAY,
        "color-background-secondary": DARK_BLUE,
        # Colors of texts and other "foreground" elements
        "color-foreground-primary": "white",
        "color-foreground-secondary": "white",
        # General borders
        "color-background-border": "transparent",
        # Two general colors for several aspects of the color cheme
        "color-brand-primary": BROWN,
        "color-brand-secondary": "black",
        # "Muted" text
        "color-foreground-muted": "white",
        # Background of the current target (e.g. after clicking on an internal link)
        "color-highlight-on-target": DARK_BLUE,
        # General backgrounds of headers and similar in the API
        "color-api-background": DARK_BLUE,
        # Names in the API
        "color-api-name": BROWN,
        # Prefix in API names (e.g. baybe.campaign)
        "color-api-pre-name": "white",
        # Background of code in text
        "color-inline-code-background": DARK_GRAY,  # Gray
        # The color of links
        "color-link": BROWN,
        # Color of text in the right navigation side bar
        "color-toc-item-text": LIGHT_BLUE,
        # Color of search bar when clicking search
        "color-sidebar-search-background--focus": DARK_BLUE,
    },
    # Logos. Location is relative to _static folder.
    "light_logo": "logo1.svg",  # Logo for light mode
    "dark_logo": "logo1.svg",  # Logo for dark mode
}

# Ignored links for linkcheck
if "BAYBE_DOCS_LINKCHECK_IGNORE" in os.environ:
    linkcheck_ignore = ["https://emdgroup.github.io/baybe/"]


# Everything in the module has the prefix baybe
modindex_common_prefix = ["baybe."]

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

# Mappings to all external packages that we want to have clickable links to
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "sklearn_extra": ("https://scikit-learn-extra.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "rdkit": ("https://rdkit.org/docs/", None),
}

# --- Options for autodoc typehints and autodoc -------------------------------
# https://pypi.org/project/sphinx-autodoc-typehints/

# For some reason, sphinx does not like it if we use the -D option to just tell it
# that we want to include private members. We thus manually verify whether the option
# was set.
private_members = "True" in sys.argv[sys.argv.index("-D") + 1]

# Represent typehints whenever possible.
autodoc_typehints = "both"
# Separate class names and init functions.
autodoc_class_signature = "separated"
# Do not preserve the defaults as we want them to be evaluated due to attrs
# Seems to be bugged for attrs defaults, see
# https://github.com/sphinx-toolbox/sphinx-toolbox/issues/146
autodoc_preserve_defaults = False
# Set the default options that should be used for autodoc
autodoc_default_options = {
    # Order by type (function, attribute...), required for proper inheritance
    "member-order": "groupwise",
    # Include private members if this was requested
    "private-members": private_members,
}
# Only show parameters that are documented.
autodoc_typehints_description_target = "documented_params"
# Shorten the links in the function signatures
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True
# Typehints should be shown in the signature
typehints_use_signature = True


# This function enables us to hook into the internal sphinx processes
# These allow us to change docstrings (resp. how they are processed) which is currently
# necessary for properly rendering the surrogates
def setup(app):  # noqa: D103
    get_available_surrogates()
    app.connect("autodoc-process-docstring", autodoc_process_docstring)


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Process the docstrings that sphinx creates."""
    if "__init__" in name and len(lines) > 0:
        lines.append(
            "For details on the parameters, see **Public attributes and properties**."
        )
