************
sunkit-image
************

.. grid:: 3

    .. grid-item::

        A toolbox of useful image processing routines applicable to solar physics data.

    .. grid-item-card:: API Reference

        .. toctree::
          :maxdepth: 1

          code_ref/index

    .. grid-item-card:: Other info

        .. toctree::
          :maxdepth: 1

          generated/gallery/index
          changelog

Installation
============

For detailed installation instructions, see the `installation guide`_ in the ``sunpy`` docs.
This takes you through the options for getting a virtual environment and installing ``sunpy``.
You will need to replace "sunpy" with "sunkit-image".

Getting Help
============

Stop by our chat room `#sunpy:matrix.org`_ if you have any questions.

Contributing
============

Help is always welcome so let us know what you like to work on, or check out the `issues page`_ for the list of known outstanding items.
If you would like to get involved, please read our `contributing guide`_, this talks about ``sunpy`` but the same is for ``sunkit-image``.

If you want help develop ``sunkit-image`` you will need to install it from GitHub.
The best way to do this is to create a new python virtual environment.
Once you have that virtual environment, you will want to fork the repo and then run::

    $ git clone https://github.com/<your_username>/sunkit-image.git
    $ cd sunkit-image
    $ pip install -e ".[dev]"

.. _installation guide: https://docs.sunpy.org/en/stable/tutorial/installation.html
.. _`#sunpy:matrix.org`: https://app.element.io/#/room/#sunpy:openastronomy.org
.. _issues page: https://github.com/sunpy/sunkit-image/issues
.. _contributing guide: https://docs.sunpy.org/en/latest/dev_guide/contents/newcomers.html
