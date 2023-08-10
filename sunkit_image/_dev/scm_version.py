# Try to use setuptools_scm to get the current version; this is only used
# in development installations from the git repository.
from pathlib import Path

try:
    from setuptools_scm import get_version

    version = get_version(root=Path("..") / "..", relative_to=__file__)
except ImportError as e:
    raise ImportError("setuptools_scm not installed") from e
except Exception as e:  # NOQA: BLE001
    raise ValueError(f"setuptools_scm broken with {e}") from e
