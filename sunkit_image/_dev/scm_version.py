# Try to use setuptools_scm to get the current version; this is only used
# in development installations from the git repository.
from pathlib import Path

try:
    from setuptools_scm import get_version

    version = get_version(root=Path("..") / "..", relative_to=__file__)
except ImportError as e:
    msg = "setuptools_scm not installed"
    raise ImportError(msg) from e
except Exception as e:
    msg = f"setuptools_scm broken with {e}"
    raise ValueError(msg) from e
