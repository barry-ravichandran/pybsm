Updates / New Features
----------------------

* Added a sensor-calibrated inverse on ``ImageSimulator``:
  ``photoelectrons_to_reflectance`` and ``photoelectrons_to_pixels``
  convert the photoelectron output of ``simulate_image`` back into
  reflectance or display pixels, with output that tracks the sensor's
  calibration (so two sensors viewing the same scene produce comparable
  pixels). ``photoelectrons_to_pixels`` also accepts ``mode="minmax"``
  for per-image min-max stretch — useful for side-by-side qualitative
  comparisons on a fixed scene, but not suitable for cross-sensor work.

* Clarified the input range for ``Scenario(interp=True)``: ``altitude``
  and ``ground_range`` may be any value within the MODTRAN tabulation
  bounds, not just exact tabulated entries (with ``interp=False`` the
  inputs still have to be exact entries).

Fixes
-----

* ``ImageSimulator.apply_convolution`` no longer hangs on uniform-gray
  inputs.

* ``ImageSimulator.apply_noise`` now raises ``RuntimeError`` with a
  diagnostic message if the input contains ``NaN`` or ``Inf``, instead
  of hanging silently.
