.. :auto introduction:

Installation
============

There are multiple ways to obtain the pybsm package.
The simplest is to install via the :command:`pip` command.
Alternatively, you can install via :command:`conda-forge` command.
For local development, you can use `Poetry`_.

pybsm installation has been tested on Unix and Linux systems.

.. :auto introduction:


.. :auto install-commands:

From :command:`pip`
-------------------

.. prompt:: bash

    pip install pybsm

From :command:`conda-forge`
---------------------------

.. prompt:: bash

    conda install -c conda-forge pybsm

.. :auto install-commands:

.. :auto from-source:

From Source
-----------
The following assumes `Poetry`_ (`installation`_ and `usage`_) is already installed.

`Poetry`_ is used for development of pybsm. Unlike the previous options,
`Poetry`_ will not only allows developers to install any extras they need,
but also install developmental dependencies like ``pytest`` and pybsm's linting tools.

.. :auto from-source:

.. :auto quick-start:

Quick Start
^^^^^^^^^^^

.. prompt:: bash

    cd /where/things/should/go/
    git clone https://github.com/kitware/pybsm.git ./
    poetry install

.. :auto quick-start:

.. :auto dev-deps:

Installing Developer Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following installs both core and development dependencies as
specified in the :file:`pyproject.toml` file, with versions specified
(including for transitive dependencies) in the :file:`poetry.lock` file:

.. prompt:: bash

    poetry install --sync --with linting,tests,docs

.. :auto dev-deps:

.. :auto build-docs:

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^
The documentation for pybsm is maintained as a collection of
`reStructuredText`_ documents in the :file:`docs/` folder of the project.
The :program:`Sphinx` documentation tool can process this documentation
into a variety of formats, the most common of which is HTML.

Within the :file:`docs/` directory is a Unix :file:`Makefile` (for Windows
systems, a :file:`make.bat` file with similar capabilities exists).
This :file:`Makefile` takes care of the work required to run :program:`Sphinx`
to convert the raw documentation to an attractive output format.
For example, calling the command below will generate
HTML format documentation rooted at :file:`docs/_build/html/index.html`.

.. prompt:: bash

    poetry run make html


Calling the command ``make help`` here will show the other documentation
formats that may be available (although be aware that some of them require
additional dependencies such as :program:`TeX` or :program:`LaTeX`).

.. :auto build-docs:

.. :auto live-preview:

Live Preview
""""""""""""

While writing documentation in a markup format such as `reStructuredText`_, it
is very helpful to preview the formatted version of the text.
While it is possible to simply run the ``make html`` command periodically, a
more seamless workflow of this is available.
Within the :file:`docs/` directory is a small Python script called
:file:`sphinx_server.py` that can simply be called with:

.. prompt:: bash

    poetry run python sphinx_server.py

This will run a small process that watches the :file:`docs/` folder contents,
as well as the source files in :file:`src/pybsm/`, for changes.
:command:`make html` is re-run automatically when changes are detected.
This will serve the resulting HTML files at http://localhost:5500.
Having this URL open in a browser will provide you with an up-to-date
preview of the rendered documentation.

.. :auto live-preview:

.. :auto installation-links:

.. _Poetry: https://python-poetry.org
.. _installation: https://python-poetry.org/docs/#installation
.. _usage: https://python-poetry.org/docs/basic-usage/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html

.. :auto installation-links:
