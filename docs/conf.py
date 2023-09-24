# conf.py

# -- Project information -----------------------------------------------------

project = 'My Documentation'
author = 'Your Name'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = []

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# Use the default theme.
html_theme = 'alabaster'

# HTML theme options.
html_theme_options = {
    'logo': 'logo.png',  # You can specify a logo file.
    'github_user': 'yourusername',
    'github_repo': 'yourrepository',
    'github_banner': True,
    'show_related': False,
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
}

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'mydocumentation', 'My Documentation', [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'MyDocumentation', 'My Documentation', author, 'MyDocumentation', 'One line description of project.', 'Miscellaneous'),
]
