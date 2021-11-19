Fixed a bug where the way we dealt with `~astropy.unit.Quantity` objects was inconsistent with
`~dask.array.Array` objects in newer versions of `~numpy`. The `pre_check_hook` option keyword
argument has also been removed from `~sunkit_image.time_lag.time_lag` and `post_check_hook`
has been renamed to `array_check` and now accepts two arguments.
