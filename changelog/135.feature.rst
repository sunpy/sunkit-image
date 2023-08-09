Several functions have been updated to accept either numpy array or sunpy map inputs.
The following functions now accept either a numpy array or sunpy map, and return the same data type:

- `sunkit_image.enhance.mgn`
- `sunkit_image.trace.bandpass_filter`
- `sunkit_image.trace.smooth`

The following functions now accept either a numpy array or sunpy map, and their return type is unchanged:

- `sunkit_image.trace.occult2`
