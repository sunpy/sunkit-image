import numpy as np

try:
    from sunkit_image.flct import _pyflct
except ImportError:
    _pyflct = None

__all__ = [
    "read_2_images",
    "read_3_images",
    "write_2_images",
    "write_3_images",
    "column_row_of_two",
    "column_row_of_three",
]


def read_2_images(filename, order="row"):
    """
    Reads two arrays of the same size from a ``dat`` file.

    .. note ::
        This function can be used to read only special arrays which were written
        using the ``write`` functions in `~sunkit_image.flct` or the IDL IO routines
        as given on the FLCT `website <http://solarmuri.ssl.berkeley.edu/~fisher/public/software/FLCT/C_VERSIONS/>`__.

    Parameters
    ----------
    filename : `str`
        The name of ``dat`` file.
    order : {"row" | "column"}
        The order in which the array elements are stored that is whether they are stored as row
        major or column major.

    Returns
    -------
    `tuple`
        A tuple containing two `~numpy.ndarray`.
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    if order.lower() not in ["row", "column"]:
        raise ValueError(
            "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
        )

    if order is "row":
        transp = 0
    else:
        transp = 1

    ier, a, b = _pyflct.read_two_images(filename, transp)

    if ier is not 1:
        raise ValueError("The file was not read correctly. Please check the file.")

    else:
        return a, b


def read_3_images(filename, order="row"):
    """
    Read three arrays of the same size from a ``dat`` file.

    .. note ::
        This function can be used to read only special arrays which were written
        using the ``write`` functions in `~sunkit_image.flct` or the IDL IO routines
        as given on the FLCT source `website <http://solarmuri.ssl.berkeley.edu/~fisher/public/software/FLCT/C_VERSIONS/>`__.

    Parameters
    ----------
    filename : `str`
        The name of ``dat`` file.
    order : {"row" | "column"}
        The order in which the array elements are stored that is whether they are stored as row
        major or column major.

    Returns
    -------
    `tuple`
        A tuple containing three `~numpy.ndarray`.
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    if order.lower() not in ["row", "column"]:
        raise ValueError(
            "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
        )

    if order is "row":
        transp = 0
    else:
        transp = 1

    ier, a, b, c = _pyflct.read_three_images(filename, transp)

    if ier is not 1:
        raise ValueError("The file was not read correctly. Please check the file.")

    else:
        return a, b, c


def write_2_images(filename, array1, array2, order="row"):
    """
    Write two arrays of the same size to a ``dat`` file.

    Parameters
    ----------
    filename : `str`
        The name of ``dat`` file.
    array1 : `numpy.ndarray`
        The first array to be stored.
    array2 : `numpy.ndarray`
        The second array to be stored.
    order : {"row" | "column"}
        The order in which the array elements are stored that is whether they are stored as row
        major or column major.
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    if order.lower() not in ["row", "column"]:
        raise ValueError(
            "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
        )

    if order is "row":
        transp = 0
    else:
        transp = 1

    ier = _pyflct.write_two_images(filename, array1, array2, transp)

    if ier is not 1:
        raise ValueError("The file was not read correctly. Please check the file")


def write_3_images(filename, array1, array2, array3, order="row"):
    """
    Write three arrays of the same size to a ``dat`` file.

    Parameters
    ----------
    filename : `string`
        The name of ``dat`` file.
    array1 : `numpy.ndarray`
        The first array to be stored.
    array2 : `numpy.ndarray`
        The second array to be stored.
    array3 : `numpy.ndarray`
        The third array to be stored.
    order : {"row" | "column"}
        The order in which the array elements are stored that is whether they are stored as row
        major or column major.
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    if order.lower() not in ["row", "column"]:
        raise ValueError(
            "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
        )

    if order is "row":
        transp = 0
    else:
        transp = 1

    ier = _pyflct.write_three_images(filename, array1, array2, array3, transp)

    if ier is not 1:
        raise ValueError("The file was not read correctly. Please check the file")


def column_row_of_two(array1, array2):
    """
    Takes two arrays and swaps the order in which they were stored i.e.
    changing from column major to row major and **not** the vice-versa. This
    may change the values stored in the array as the arrays are first converted
    to a binary format and then the order change takes place.

    Parameters
    ----------
    array1 : `numpy.ndarray`
        The first array whose order is to be changed.
    array2 : `numpy.ndarray`
        The second array whose order is to be changed.

    Returns
    -------
    `tuple`
        It returns the two input arrays after changing their order from column major to
        row major.
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    one, two = _pyflct.swap_order_two(array1, array2)

    return (one, two)


def column_row_of_three(array1, array2, array3):
    """
    Takes three arrays and swaps the order in which they were stored i.e.
    changing from column major to row major and **not** the vice-versa. This
    may change the values stored in the array as the arrays are first converted
    to a binary format and then the order change takes place.

    Parameters
    ----------
    array1 : `numpy.ndarray`
        The first array whose order is to be changed.
    array2 : `numpy.ndarray`
        The second array whose order is to be changed.
    array3 : `numpy.ndarray`
        The third array whose order is to be changed.

    Returns
    -------
    `tuple`
        It returns the two input arrays after changing their order from column major to
        row major.
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    one, two, three = _pyflct.swap_order_three(array1, array2, array3)

    return (one, two, three)
