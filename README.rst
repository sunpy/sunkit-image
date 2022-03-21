sunkit-image
============

|Latest Version| |codecov| |Powered by NumFOCUS| |Powered by Sunpy|

.. |Powered by Sunpy| image:: http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat
   :target: https://www.sunpy.org
.. |Latest Version| image:: https://img.shields.io/pypi/v/sunkit-image.svg
   :target: https://pypi.python.org/pypi/sunkit-image/
.. |codecov| image:: https://codecov.io/gh/sunpy/sunpy/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/sunpy/sunkit-image
.. |Powered by NumFOCUS| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: http://numfocus.org

sunkit-image is a a open-source toolbox for solar physics image processing.
Currently it is an experimental library for various solar physics specific image processing routines.

Installation
------------

For detailed (general) installation instructions, see the `installation guide`_ in the SunPy docs.
This takes you through the options for installing sunpy, but they are the same for sunkit-image.

If you want help develop sunkit-image you will need to install it from GitHub.
The best way to do this is to create a new python virtual environment (either with ``pipenv`` or ``conda``).
Once you have that virtual environment, you will want to fork the repo and then run::

    $ git clone https://github.com/<username>/sunkit-image.git
    $ cd sunkit-image
    $ pip install -e .[dev]

Getting Help
------------

For more information or to ask questions about sunkit-image or sunpy, check out:

-  `#sunpy:matrix.org`_
-  `sunkit-image Documentation`_

Contributing
------------

If you would like to get involved, please read our `contributing guide`_.
Stop by our chat room `#sunpy:matrix.org`_ if you have any questions.
Help is always welcome so let us know what you like to work on, or check out the `issues page`_ for the list of known outstanding items.

Code of Conduct
---------------

When you are interacting with the SunPy community you are asked to follow our `Code of Conduct`_.

License
-------

This project is Copyright (c) SunPy Developers and licensed under the terms of the BSD 3-Clause license.
See the licenses folder for more information.

.. _installation guide: https://docs.sunpy.org/en/stable/guide/installation/index.html
.. _`#sunpy:matrix.org`: https://app.element.io/#/room/#sunpy:openastronomy.org
.. _issues page: https://github.com/sunpy/sunkit-image/issues
.. _contributing guide: https://docs.sunpy.org/en/latest/dev_guide/contents/newcomers.html
.. _Code of Conduct: https://sunpy.org/coc
.. _sunkit-image Documentation: https://docs.sunpy.org/projects/sunkit-image/en/stable/
