Updates / New Features
----------------------

* Added ``ImageSimulator.photoelectrons_to_reflectance`` and
  ``ImageSimulator.photoelectrons_to_pixels`` — analytical inverse of the
  forward reflectance to photoelectrons map. Enables radiometrically
  faithful photoelectron-to-pixel mapping in downstream consumers, replacing per-image
  min-max normalization that destroyed cross-sensor consistency.

* For ``use_reflectance=False`` simulators, ``photoelectrons_to_pixels``
  routes through the sensor's ADC model (well capacity by bit-depth
  quantization) — the physically correct mapping for raw-pixel mode.

* Added three observability guards on the new inverse path: hard-fail on
  non-finite inputs, once-per-instance warning when >1% of pixels clip at
  the forward-table boundary, and once-per-instance warning on first
  fallback to ADC quantization. Also added automatic cache invalidation
  that keeps the analytical inverse coefficients consistent when
  ``_reflect_to_photoelectrons`` is replaced or boundary-mutated.

* Documented the ``Scenario(interp=True)`` envelope contract — accepts any
  ``ground_range`` and ``altitude`` inside the MODTRAN grid envelope; the
  exact-grid-entry restriction applies only when ``interp=False``.

Fixes
-----

* Fixed ``ImageSimulator.apply_convolution`` divide-by-zero on uniform-gray
  input images. Previously, ``image.min() == image.max()`` produced NaN
  that propagated through the FFT and hung the simulator on synthetic
  flat inputs. Uniform inputs now map to the midpoint of the configured
  ``reflectance_range``.

* Fixed an indefinite hang in ``ImageSimulator.apply_noise`` when called
  on an array containing ``NaN`` (numba parallel-fastmath does not handle
  ``NaN`` Poisson lambdas, hanging the worker thread). ``apply_noise``
  now raises ``RuntimeError`` with the NaN/Inf counts and shape, failing
  fast instead of hanging silently.
