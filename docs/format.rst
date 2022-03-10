****************
Formatting Style
****************

This outlines some of the code style that sunkit-image follows::

  $ autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports --exclude=__init__.py ./sunkit_image
  $ isort -rc ./sunkit_image
  $ docformatter -ri --pre-summary-newline --make-summary-multi-line  ./sunkit_image
  $ black ./sunkit_image

These commands run in the root of the repository, should be done before you commit any new code.
You will need to install these packages.
For the first command, if you need to import but not use it directly, you should add ``# noqa`` next to the import to skip this line.
