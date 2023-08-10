# NOTE: First try _dev.scm_version if it exists and setuptools_scm is installed
# This file is not included in sunkit-image wheels/tarballs, so otherwise it will
# fall back on the generated _version module.
try:
    try:
        from ._dev.scm_version import version
    except ImportError:
        from ._version import version  # type: ignore[no-redef]
except Exception:  # NOQA: BLE001
    import warnings

    warnings.warn(
        f'could not determine {__name__.split(".")[0]} package version; this indicates a broken installation',
        stacklevel=3,
    )
    del warnings
    version = "0.0.0"

__all__ = ["version"]
