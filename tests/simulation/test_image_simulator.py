from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from syrupy.assertion import SnapshotAssertion

from pybsm import simulation
from pybsm.simulation import (
    CircularApertureSimulator,
    DefocusSimulator,
    DetectorSimulator,
    JitterSimulator,
    SystemOTFSimulator,
    TurbulenceApertureSimulator,
)

BASE_FILE_PATH = Path(__file__).parent.parent.parent
IMAGE_FILE_PATH = (
    BASE_FILE_PATH / "docs" / "examples" / "data" / "M-41 Walker Bulldog (USA) width 319cm height 272cm.tiff"
)


class TestImageSimulator:
    @pytest.mark.parametrize(
        ("img_file_path", "use_reflectance", "reflectance_range", "mtf_wavelengths", "mtf_weights"),
        [
            (  # use_reflectance is True, but no reflectance_range is provided
                IMAGE_FILE_PATH,
                True,
                None,
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
            ),
            (  # reflectance_range length is not 2
                IMAGE_FILE_PATH,
                True,
                np.array([1]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
            ),
            (  # reflectance_range is not strictly increasing
                IMAGE_FILE_PATH,
                True,
                np.array([0.5, 0.05]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
            ),
            (  # mtf_wavelengths and mtf_weights are not equal length
                IMAGE_FILE_PATH,
                False,
                None,
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5]),
            ),
            (  # mtf_wavelengths is empty
                IMAGE_FILE_PATH,
                False,
                None,
                np.array([]),
                np.array([0.5, 0.5]),
            ),
            (  # mtf_weights is empty
                IMAGE_FILE_PATH,
                False,
                None,
                np.array([0.5e-6, 0.6e-6]),
                np.array([]),
            ),
        ],
    )
    def test_init_value_error(
        self,
        img_file_path: str,
        use_reflectance: bool,
        reflectance_range: np.ndarray,
        mtf_wavelengths: np.ndarray,
        mtf_weights: np.ndarray,
    ) -> None:
        """Cover cases where ValueError occurs."""
        img = np.array(Image.open(img_file_path))
        gsd = 3.19 / 160.0
        altitude = 1000
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)
        with pytest.raises(ValueError):  # noqa: PT011
            SystemOTFSimulator(
                sensor=sensor,
                scenario=scenario,
                use_reflectance=use_reflectance,
                reflectance_range=reflectance_range,
                mtf_wavelengths=mtf_wavelengths,
                mtf_weights=mtf_weights,
            )


class TestSystemOTFSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation with various parameter combinations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))

    def test_psf_caching(self) -> None:
        """Verify PSF caching behavior."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        assert len(simulator._psf_cache) == 0

        psf1 = simulator._get_psf_cached(gsd=gsd)
        assert len(simulator._psf_cache) == 1

        psf2 = simulator._get_psf_cached(gsd=gsd)
        assert len(simulator._psf_cache) == 1
        assert np.array_equal(psf1, psf2)

        psf3 = simulator._get_psf_cached(gsd=gsd * 2)
        assert len(simulator._psf_cache) == 2
        assert not np.array_equal(psf1, psf3)

    def test_slant_range_altitude_overrides(self) -> None:
        """Verify slant_range and altitude override parameters."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        custom_slant_range = 1500.0
        custom_altitude = 1200.0

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
            slant_range=custom_slant_range,
            altitude=custom_altitude,
        )

        assert simulator.slant_range == custom_slant_range
        assert simulator._altitude == custom_altitude

    def test_sensor_scenario_configuration(self) -> None:
        """Verify sensor and scenario attributes match input configurations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        assert simulator.sensor.name == sensor.name
        assert simulator.sensor.D == sensor.D
        assert simulator.sensor.f == sensor.f
        assert simulator.sensor.p_x == sensor.p_x
        assert np.array_equal(simulator.sensor.opt_trans_wavelengths, sensor.opt_trans_wavelengths)
        assert np.array_equal(simulator.sensor.optics_transmission, sensor.optics_transmission)
        assert simulator.sensor.eta == sensor.eta
        assert simulator.sensor.w_x == sensor.w_x
        assert simulator.sensor.w_y == sensor.w_y
        assert simulator.sensor.int_time == sensor.int_time
        assert simulator.sensor.dark_current == sensor.dark_current
        assert simulator.sensor.read_noise == sensor.read_noise
        assert simulator.sensor.max_n == sensor.max_n
        assert simulator.sensor.bit_depth == sensor.bit_depth
        assert simulator.sensor.max_well_fill == sensor.max_well_fill
        assert simulator.sensor.s_x == sensor.s_x
        assert simulator.sensor.s_y == sensor.s_y
        assert simulator.sensor.da_x == sensor.da_x
        assert simulator.sensor.da_y == sensor.da_y
        assert np.array_equal(simulator.sensor.qe_wavelengths, sensor.qe_wavelengths)
        assert np.array_equal(simulator.sensor.qe, sensor.qe)

        assert simulator.scenario.name == scenario.name
        assert simulator.scenario.ihaze == scenario.ihaze
        assert simulator.scenario.altitude == scenario.altitude
        assert simulator.scenario.ground_range == scenario.ground_range
        assert simulator.scenario.aircraft_speed == scenario.aircraft_speed
        assert simulator.scenario.target_reflectance == scenario.target_reflectance
        assert simulator.scenario.target_temperature == scenario.target_temperature
        assert simulator.scenario.background_reflectance == scenario.background_reflectance
        assert simulator.scenario.background_temperature == scenario.background_temperature
        assert simulator.scenario.ha_wind_speed == scenario.ha_wind_speed
        assert simulator.scenario.cn2_at_1m == scenario.cn2_at_1m

    def test_noise_reproducibility_with_seed(self) -> None:
        """Verify seed reproducibility."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        seed = 1
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator1 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        simulator2 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        _, _, noisy_img1 = simulator1.simulate_image(img, gsd=gsd)
        _, _, noisy_img2 = simulator2.simulate_image(img, gsd=gsd)

        assert noisy_img1 is not None
        assert noisy_img2 is not None
        assert np.array_equal(noisy_img1, noisy_img2)

    def test_noise_reproducibility_with_generator(self) -> None:
        """Verify Generator seed reproducibility."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        seed = 1
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator1 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=np.random.default_rng(seed),
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        simulator2 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=np.random.default_rng(seed),
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        _, _, noisy_img1 = simulator1.simulate_image(img, gsd=gsd)
        _, _, noisy_img2 = simulator2.simulate_image(img, gsd=gsd)

        assert noisy_img1 is not None
        assert noisy_img2 is not None
        assert np.array_equal(noisy_img1, noisy_img2)

    def test_noise_different_seeds(self) -> None:
        """Verify different seeds produce different outputs."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        gsd = 3.19 / 160.0
        altitude = 1000
        seed = 1
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)

        simulator1 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        simulator2 = SystemOTFSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=True,
            rng=seed + 1,
            use_reflectance=True,
            reflectance_range=ref_img.refl_values,
        )

        _, _, noisy_img1 = simulator1.simulate_image(img, gsd=gsd)
        _, _, noisy_img2 = simulator2.simulate_image(img, gsd=gsd)

        assert noisy_img1 is not None
        assert noisy_img2 is not None
        assert not np.array_equal(noisy_img1, noisy_img2)


class TestJitterSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation with various parameter combinations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = JitterSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestCircularApertureSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation with various parameter combinations."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = CircularApertureSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestDetectorSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: np.random.Generator | int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = DetectorSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestDefocusSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: np.random.Generator | int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = DefocusSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


class TestTurbulenceApertureSimulator:
    @pytest.mark.parametrize(
        (
            "add_noise",
            "rng",
            "gsd_input",
            "use_reflectance",
            "reflectance_range",
            "mtf_wavelengths",
            "mtf_weights",
            "is_rgb",
        ),
        [
            # Grayscale tests
            # Full featured grayscale
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                False,
            ),
            # Grayscale no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), False),
            # Grayscale default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, False),
            # Grayscale minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, False),
            # RGB tests
            # Full featured RGB
            (
                False,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB with noise
            (
                True,
                1,
                3.19 / 160.0,
                True,
                np.array([0.05, 0.95]),
                np.array([0.5e-6, 0.6e-6]),
                np.array([0.5, 0.5]),
                True,
            ),
            # RGB no resampling (gsd=None)
            (False, 1, None, True, np.array([0.05, 0.95]), None, None, True),
            # RGB no reflectance
            (False, 1, 3.19 / 160.0, False, None, np.array([0.5e-6, 0.6e-6]), np.array([0.5, 0.5]), True),
            # RGB default MTF
            (False, 1, 3.19 / 160.0, True, np.array([0.05, 0.95]), None, None, True),
            # RGB minimal config
            (False, 1, 3.19 / 160.0, False, None, None, None, True),
        ],
    )
    def test_simulate_image(
        self,
        add_noise: bool,
        rng: np.random.Generator | int,
        gsd_input: float | None,
        use_reflectance: bool,
        reflectance_range: np.ndarray | None,
        mtf_wavelengths: np.ndarray | None,
        mtf_weights: np.ndarray | None,
        is_rgb: bool,
        psnr_tiff_snapshot: SnapshotAssertion,
    ) -> None:
        """Verify image simulation."""
        img = np.array(Image.open(IMAGE_FILE_PATH))
        if is_rgb:
            img = np.stack((img,) * 3, axis=-1)

        gsd = 3.19 / 160.0
        ref_img = simulation.RefImage(img=img, gsd=gsd)
        sensor, scenario = ref_img.estimate_capture_parameters(altitude=1000)

        simulator = TurbulenceApertureSimulator(
            sensor=sensor,
            scenario=scenario,
            add_noise=add_noise,
            rng=rng,
            use_reflectance=use_reflectance,
            reflectance_range=reflectance_range
            if reflectance_range is not None
            else (ref_img.refl_values if use_reflectance else None),
            mtf_wavelengths=mtf_wavelengths,
            mtf_weights=mtf_weights,
        )

        _, blur_img, noisy_img = simulator.simulate_image(img, gsd=gsd_input)

        assert blur_img is not None

        if add_noise:
            assert noisy_img is not None
            psnr_tiff_snapshot.assert_match(np.clip(noisy_img, 0, 255).astype(np.uint8))
        else:
            assert noisy_img is None
            psnr_tiff_snapshot.assert_match(np.clip(blur_img, 0, 255).astype(np.uint8))


def _build_simulator(
    *,
    use_reflectance: bool = True,
    reflectance_range: np.ndarray | None = None,
    altitude: int = 1000,
    sensor_overrides: dict | None = None,
) -> tuple[SystemOTFSimulator, np.ndarray]:
    """Construct a simulator + image pair matching the existing test convention."""
    img = np.array(Image.open(IMAGE_FILE_PATH))
    gsd = 3.19 / 160.0
    ref_img = simulation.RefImage(img=img, gsd=gsd)
    sensor, scenario = ref_img.estimate_capture_parameters(altitude=altitude)
    if sensor_overrides:
        for key, value in sensor_overrides.items():
            setattr(sensor, key, value)
    if reflectance_range is None and use_reflectance:
        reflectance_range = ref_img.refl_values
    simulator = SystemOTFSimulator(
        sensor=sensor,
        scenario=scenario,
        use_reflectance=use_reflectance,
        reflectance_range=reflectance_range,
    )
    return simulator, img


def _pe_with_5pct_below_floor(simulator: SystemOTFSimulator, total: int = 1000) -> np.ndarray:
    """Construct a pe array with the first 5% of elements strictly below the forward floor."""
    pe = np.full(total, (simulator._fwd_y_min + simulator._fwd_y_max) / 2.0)
    pe[: int(0.05 * total)] = simulator._fwd_y_min - 1.0
    return pe


class TestPhotoelectronsToPixels:
    """Cover the analytical inverse, ADC path, and observability guards.

    The forward ``pe(ref) = A*ref + B`` is exactly affine-linear; the inverse
    cached on the simulator is ``ref = (pe - B) / A``. These tests exercise
    each branch (analytical / ADC / fallback / cache) and the four
    observability behaviors documented on ``photoelectrons_to_pixels``:
    non-finite-input raise, clip-engagement counter, non-monotonic-grid
    fallback warning, and cache invalidation on forward-map mutation.
    """

    def test_round_trip_machine_epsilon(self) -> None:
        """forward(inverse(pe)) == pe to ~machine epsilon (relative) over the grid range."""
        simulator, _img = _build_simulator()
        fwd = simulator._reflect_to_photoelectrons
        pe_grid = np.linspace(simulator._fwd_y_min, simulator._fwd_y_max, 1000)
        ref_recovered = simulator.photoelectrons_to_reflectance(pe_grid)
        pe_round_trip = fwd(ref_recovered)
        # Tolerance is relative: at pe-scale ~1e8, IEEE 754 double epsilon (2.22e-16)
        # produces absolute errors around 2e-8, which is correct, not pathological.
        np.testing.assert_allclose(pe_round_trip, pe_grid, rtol=1e-13, atol=0.0)

    def test_analytical_matches_interp1d_argument_swap(self) -> None:
        """The cached analytical inverse matches an explicit interp1d swap."""
        from scipy import interpolate

        simulator, _img = _build_simulator()
        fwd = simulator._reflect_to_photoelectrons
        ref_inv = interpolate.interp1d(fwd.y, fwd.x)
        pe_sample = np.linspace(simulator._fwd_y_min, simulator._fwd_y_max, 100_000)
        analytical = simulator.photoelectrons_to_reflectance(pe_sample, clip_to_unit=False)
        reference = ref_inv(pe_sample)
        # Both produce ref in [0, 1]; absolute machine-epsilon comparison is correct here.
        np.testing.assert_allclose(analytical, reference, rtol=0.0, atol=1e-12)

    def test_clip_to_unit_false_allows_out_of_range_reflectance(self) -> None:
        """clip_to_unit=False lets pe outside the grid produce ref outside [0, 1]."""
        simulator, _img = _build_simulator()
        affine_a, affine_b = simulator._affine_pe_coefs
        below = simulator.photoelectrons_to_reflectance(
            np.array([affine_b - abs(affine_a) * 0.5]),
            clip_to_unit=False,
        )
        above = simulator.photoelectrons_to_reflectance(
            np.array([affine_b + affine_a * 1.5]),
            clip_to_unit=False,
        )
        assert below[0] < 0.0
        assert above[0] > 1.0
        # Default clip pins both to [0, 1].
        clipped_below = simulator.photoelectrons_to_reflectance(
            np.array([affine_b - abs(affine_a) * 0.5]),
        )
        clipped_above = simulator.photoelectrons_to_reflectance(
            np.array([affine_b + affine_a * 1.5]),
        )
        assert clipped_below[0] == 0.0
        assert clipped_above[0] == 1.0

    def test_use_reflectance_false_raises_for_pe_to_reflectance(self) -> None:
        """photoelectrons_to_reflectance requires use_reflectance=True."""
        simulator, _img = _build_simulator(use_reflectance=False)
        with pytest.raises(ValueError, match="use_reflectance=True"):
            simulator.photoelectrons_to_reflectance(np.array([1000.0]))

    def test_use_reflectance_false_routes_to_adc_in_pe_to_pixels(self) -> None:
        """In raw-pixel mode, photoelectrons_to_pixels uses the sensor ADC model."""
        simulator, _img = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 12.0, "max_well_fill": 1.0},
        )
        well_cap = 32400 * 1.0
        pe_sweep = np.linspace(0.0, well_cap, 64)
        pixels = simulator.photoelectrons_to_pixels(pe_sweep)
        assert np.all(np.diff(pixels) >= 0.0)
        assert pixels[0] == pytest.approx(0.0, abs=1.0)
        assert pixels[-1] == pytest.approx(255.0, abs=1.0)

    def test_adc_bit_depth_4_posterizes_to_at_most_16_levels(self) -> None:
        """4-bit ADC produces at most 16 unique uint8 levels in the output sweep."""
        simulator, _img = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 4.0, "max_well_fill": 1.0},
        )
        pe_sweep = np.linspace(0.0, 32400.0, 1024)
        pixels = simulator.photoelectrons_to_pixels(pe_sweep)
        unique_uint8 = np.unique(np.clip(pixels, 0, 255).astype(np.uint8))
        assert unique_uint8.size <= 16

    def test_adc_bit_depth_non_integer_truncates_consistently(self) -> None:
        """bit_depth 12.5 truncates to 12 and matches integer 12 output."""
        sim_12, _ = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 12.0, "max_well_fill": 1.0},
        )
        sim_125, _ = _build_simulator(
            use_reflectance=False,
            sensor_overrides={"max_n": 32400, "bit_depth": 12.5, "max_well_fill": 1.0},
        )
        pe_sweep = np.linspace(0.0, 32400.0, 256)
        np.testing.assert_array_equal(
            sim_12.photoelectrons_to_pixels(pe_sweep),
            sim_125.photoelectrons_to_pixels(pe_sweep),
        )

    @pytest.mark.parametrize(
        ("sensor_overrides", "match_pattern"),
        [
            ({"max_n": 0, "max_well_fill": 1.0}, "degenerate sensor well capacity"),
            ({"max_n": 32400, "bit_depth": 0.0, "max_well_fill": 1.0}, "bit_depth must be >= 1"),
        ],
        ids=["zero_well_capacity", "bit_depth_below_one"],
    )
    def test_adc_degenerate_sensor_raises(self, sensor_overrides: dict, match_pattern: str) -> None:
        """Degenerate ADC sensor configurations raise ValueError on the inverse-pixel path."""
        simulator, _img = _build_simulator(use_reflectance=False, sensor_overrides=sensor_overrides)
        with pytest.raises(ValueError, match=match_pattern):
            simulator.photoelectrons_to_pixels(np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize(
        ("p1", "p2"),
        [(0.0, 0.0), (255.0, 0.0), (10.0, 5.0)],
        ids=["equal", "inverted_full_range", "inverted_partial"],
    )
    def test_pe_to_pixels_invalid_output_range_raises(self, p1: float, p2: float) -> None:
        """Caller must pass a strictly-ascending output range; equal or inverted raises."""
        simulator, _img = _build_simulator()
        with pytest.raises(ValueError, match="strictly ascending"):
            simulator.photoelectrons_to_pixels(np.array([1000.0, 2000.0]), p1=p1, p2=p2)

    @pytest.mark.parametrize(
        ("inject_nan", "inject_inf"),
        [(True, False), (False, True), (True, True)],
        ids=["nan_only", "inf_only", "nan_and_inf"],
    )
    def test_raises_on_non_finite_input(self, inject_nan: bool, inject_inf: bool) -> None:
        """Non-finite pe_img must raise RuntimeError, never silently cast."""
        simulator, _img = _build_simulator()
        pe_img = np.full(64, simulator._fwd_y_min, dtype=np.float64)
        if inject_nan:
            pe_img[0] = np.nan
        if inject_inf:
            pe_img[1] = np.inf
        with pytest.raises(RuntimeError, match="non-finite"):
            simulator.photoelectrons_to_pixels(pe_img)

    @pytest.mark.parametrize(
        ("pe_factory", "expected_frac_lo", "expected_frac_hi", "expects_warning"),
        [
            (lambda sim: _pe_with_5pct_below_floor(sim), 0.05, 0.0, True),
            (lambda sim: np.linspace(sim._fwd_y_min, sim._fwd_y_max, 200), 0.0, 0.0, False),
            (lambda sim: np.full(100, sim._fwd_y_min), 0.0, 0.0, False),
            (lambda sim: np.full((100,), sim._fwd_y_min - 1.0, dtype=np.float64), 1.0, 0.0, True),
        ],
        ids=[
            "5pct_below_floor_warns",
            "all_inside_silent",
            "exactly_at_floor_silent",
            "all_below_floor_warns",
        ],
    )
    def test_clip_fraction_observability(
        self,
        pe_factory: Callable[[SystemOTFSimulator], np.ndarray],
        expected_frac_lo: float,
        expected_frac_hi: float,
        expects_warning: bool,
    ) -> None:
        """Clip-fraction counter and once-per-instance warning track ``pe`` boundary engagement.

        The counter uses strict ``<`` / ``>`` against the forward floor/ceiling
        (so ``pe == floor`` is NOT counted as a clip), and the warning fires when
        either fraction exceeds 1%.
        """
        simulator, _img = _build_simulator()
        pe_img = pe_factory(simulator)
        if expects_warning:
            with pytest.warns(UserWarning, match="clipped to forward-table bounds"):
                simulator.photoelectrons_to_pixels(pe_img)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                simulator.photoelectrons_to_pixels(pe_img)
        frac_lo, frac_hi = simulator._last_clip_fraction
        assert frac_lo == pytest.approx(expected_frac_lo, abs=1e-6)
        assert frac_hi == pytest.approx(expected_frac_hi, abs=1e-6)

    def test_clip_warning_is_sticky_after_first_emission(self) -> None:
        """Once the clip warning fires, subsequent calls do not re-emit (per-instance flag)."""
        simulator, _img = _build_simulator()
        pe_img = _pe_with_5pct_below_floor(simulator)
        with pytest.warns(UserWarning, match="clipped to forward-table bounds"):
            simulator.photoelectrons_to_pixels(pe_img)
        # Second call: warning state stays sticky; no new warning is emitted.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            simulator.photoelectrons_to_pixels(pe_img)

    def test_warns_and_falls_back_on_nonmonotonic_forward_grid(self) -> None:
        """A flattened forward tail triggers the fallback to ADC quantization."""
        simulator, _img = _build_simulator(
            sensor_overrides={"max_n": 32400, "bit_depth": 12.0, "max_well_fill": 1.0},
        )
        ref_grid = np.linspace(0.0, 1.0, 100)
        pe_grid = np.linspace(0.0, 30000.0, 100)
        pe_grid[80:] = pe_grid[80]  # flatten the high tail (saturation cliff).
        simulator._reflect_to_photoelectrons = interpolate.interp1d(ref_grid, pe_grid)
        pe_img = np.linspace(0.0, 30000.0, 64)
        with pytest.warns(UserWarning, match="forward_nonmonotonic"):
            pixels = simulator.photoelectrons_to_pixels(pe_img)
        expected = simulator._adc_photoelectrons_to_pixels(pe_img, out_range=(0.0, 255.0))
        np.testing.assert_array_equal(pixels, expected)

    def test_fallback_warning_is_sticky_after_first_emission(self) -> None:
        """Once the fallback warning fires, subsequent calls do not re-emit (per-instance flag)."""
        simulator, _img = _build_simulator(
            sensor_overrides={"max_n": 32400, "bit_depth": 12.0, "max_well_fill": 1.0},
        )
        ref_grid = np.linspace(0.0, 1.0, 100)
        pe_grid = np.linspace(0.0, 30000.0, 100)
        pe_grid[80:] = pe_grid[80]
        simulator._reflect_to_photoelectrons = interpolate.interp1d(ref_grid, pe_grid)
        pe_img = np.linspace(0.0, 30000.0, 64)
        with pytest.warns(UserWarning, match="forward_nonmonotonic"):
            simulator.photoelectrons_to_pixels(pe_img)
        # Second call: warning state stays sticky; no new warning is emitted.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            simulator.photoelectrons_to_pixels(pe_img)

    def test_cache_invalidates_on_forward_replacement(self) -> None:
        """Replacing _reflect_to_photoelectrons rebuilds (A, B) on next call."""
        simulator, _img = _build_simulator()
        original_a, original_b = simulator._affine_pe_coefs
        new_ref = simulator._reflect_to_photoelectrons.x
        new_pe = simulator._reflect_to_photoelectrons.y * 10.0
        simulator._reflect_to_photoelectrons = interpolate.interp1d(new_ref, new_pe)
        simulator.photoelectrons_to_pixels(np.array([new_pe[0], new_pe[-1]]))
        new_a, new_b = simulator._affine_pe_coefs
        assert new_a == pytest.approx(original_a * 10.0, rel=1e-12)
        assert new_b == pytest.approx(original_b * 10.0, rel=1e-12)

    def test_cache_invalidates_on_boundary_mutation(self) -> None:
        """Replacing fwd with a copy whose y[-1] differs rebuilds (A, B)."""
        simulator, _img = _build_simulator()
        original_a, _original_b = simulator._affine_pe_coefs
        ref_grid = simulator._reflect_to_photoelectrons.x.copy()
        pe_grid = simulator._reflect_to_photoelectrons.y.copy()
        pe_grid[-1] = pe_grid[-1] * 2.0
        simulator._reflect_to_photoelectrons = interpolate.interp1d(ref_grid, pe_grid)
        simulator.photoelectrons_to_pixels(np.array([pe_grid[0], pe_grid[-1]]))
        new_a, _new_b = simulator._affine_pe_coefs
        assert new_a > original_a

    def test_cache_does_not_detect_interior_mutation(self) -> None:
        """Documented contract: interior `.y` mutation is NOT detected (callers replace fwd)."""
        simulator, _img = _build_simulator()
        snapshot = simulator._affine_pe_coefs
        simulator._reflect_to_photoelectrons.y[50] = simulator._reflect_to_photoelectrons.y[50] * 1.5
        simulator.photoelectrons_to_pixels(np.array([simulator._fwd_y_min, simulator._fwd_y_max]))
        assert simulator._affine_pe_coefs == snapshot

    def test_apply_convolution_uniform_input_does_not_explode(self) -> None:
        """§5 fix: apply_convolution on a uniform-gray image stays finite."""
        simulator, _img = _build_simulator(reflectance_range=np.array([0.05, 0.5]))
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        psf = simulator._get_psf_cached(gsd=3.19 / 160.0)
        true_img, blur_img = simulator.apply_convolution(uniform, psf)
        assert np.all(np.isfinite(true_img))
        assert np.all(np.isfinite(blur_img))

    def test_pe_to_pixels_is_sensor_calibrated_not_per_image(self) -> None:
        """Same pe input -> different pixels under different sensor forward calibrations.

        A naive pe-to-pixel mapping (per-image min-max normalization) produces
        output determined by the image's pe min/max rather than the sensor's
        forward calibration — destroying cross-sensor radiometric consistency.
        This test asserts that ``photoelectrons_to_pixels`` does the opposite:
        two simulators with materially different forward maps map an identical
        pe array to materially different pixel arrays, proving the output
        tracks the sensor configuration, not per-image stats.
        """
        from scipy import interpolate

        simulator, _img = _build_simulator()
        # Original forward map for simulator A.
        fwd_a = simulator._reflect_to_photoelectrons
        # Build a divergent forward map for simulator B by shifting both endpoints.
        # Note: this pure-offset shift is a unit-test contrivance — a real
        # sensor recalibration would change both slope and intercept. We just
        # need ``(A, B)`` to differ from simulator A's so the cache rebuild
        # path produces a different inverse, which is what this test exercises.
        ref_b = fwd_a.x.copy()
        pe_b = fwd_a.y.copy() + (fwd_a.y[-1] - fwd_a.y[0]) * 0.25
        # Identical pe input across both simulators.
        pe_input = np.linspace(fwd_a.y[0], fwd_a.y[-1], 5000)
        pixels_a = simulator.photoelectrons_to_pixels(pe_input)
        # Swap simulator's forward to B; cache invalidates on next call.
        simulator._reflect_to_photoelectrons = interpolate.interp1d(ref_b, pe_b)
        pixels_b = simulator.photoelectrons_to_pixels(pe_input)
        # Materially different mean uint8 levels.
        assert abs(pixels_a.mean() - pixels_b.mean()) > 5.0

    def test_a_zero_degenerate_raises(self) -> None:
        """A constant forward map (A=0) raises ValueError on inversion."""
        from scipy import interpolate

        simulator, _img = _build_simulator()
        ref_grid = simulator._reflect_to_photoelectrons.x.copy()
        pe_const = np.full_like(ref_grid, 1000.0)
        simulator._reflect_to_photoelectrons = interpolate.interp1d(ref_grid, pe_const)
        with pytest.raises(ValueError, match="A=0"):
            simulator.photoelectrons_to_reflectance(np.array([1000.0]))

    @pytest.mark.parametrize(
        "shape",
        [(0,), (0, 4)],
        ids=["empty_1D", "empty_2D"],
    )
    def test_pe_to_pixels_empty_input_returns_empty(self, shape: tuple[int, ...]) -> None:
        """Empty input short-circuits before the clip-fraction divide-by-size.

        All ``size == 0`` shapes hit the same early-return branch; one 1D and
        one 2D case are sufficient to pin both ``size==0`` short-circuit and
        ``np.empty(shape, ...)`` shape preservation.
        """
        simulator, _img = _build_simulator()
        empty = np.empty(shape, dtype=np.float64)
        out = simulator.photoelectrons_to_pixels(empty)
        assert out.shape == shape
        assert out.dtype == np.float64
        assert out.size == 0

    def test_pe_to_pixels_single_element_input(self) -> None:
        """Single-element 1D input maps cleanly without divide-by-zero in the clip-fraction."""
        simulator, _img = _build_simulator()
        # Pick a pe value squarely inside the forward range so no clipping occurs.
        pe_mid = 0.5 * (simulator._fwd_y_min + simulator._fwd_y_max)
        out = simulator.photoelectrons_to_pixels(np.array([pe_mid]))
        assert out.shape == (1,)
        assert 0.0 <= out[0] <= 255.0
        # No clipping was engaged for an interior value.
        assert simulator._last_clip_fraction == (0.0, 0.0)

    @pytest.mark.parametrize(
        ("ndim", "inject"),
        [(2, "nan"), (3, "nan"), (2, "inf"), (3, "inf")],
        ids=["2D_nan", "3D_nan", "2D_inf", "3D_inf"],
    )
    def test_apply_noise_raises_on_non_finite_input(self, ndim: int, inject: str) -> None:
        """apply_noise must raise on NaN/Inf to prevent the numba parallel-fastmath hang.

        Empirically verified: with add_noise=True and a NaN pixel, _apply_noise{2,3}d
        hang indefinitely on a 32x32 array (subprocess-bounded repro). Inf is
        added defensively — even though the current numba runtime tolerates it,
        the guard's contract is "no non-finite values reach the numba kernels".
        """
        simulator, _img = _build_simulator()
        # Force the noise path active without rebuilding the whole simulator.
        simulator._add_noise = True
        simulator._g_noise = 1.0
        shape = (32, 32) if ndim == 2 else (32, 32, 3)
        arr = np.full(shape, 100.0, dtype=np.float64)
        arr.flat[0] = np.nan if inject == "nan" else np.inf
        with pytest.raises(RuntimeError, match="non-finite image"):
            simulator.apply_noise(arr)

    def test_apply_noise_finite_input_does_not_raise(self) -> None:
        """The new finite-value guard does not regress the clean noise path."""
        simulator, _img = _build_simulator()
        simulator._add_noise = True
        simulator._g_noise = 1.0
        # 3D and 2D both pass through.
        out_3d = simulator.apply_noise(np.full((32, 32, 3), 100.0, dtype=np.float64))
        out_2d = simulator.apply_noise(np.full((32, 32), 100.0, dtype=np.float64))
        assert out_3d.shape == (32, 32, 3)
        assert out_2d.shape == (32, 32)
        assert np.all(np.isfinite(out_3d))
        assert np.all(np.isfinite(out_2d))

    def test_cross_shape_simulate_image_is_stable(self) -> None:
        """Sentinel: pyBSM's PSF cache and noise functions are shape-independent.

        The PSF cache is keyed on ``(config_hash, gsd_rounded)``, which does
        not include image shape; the parallel-numba noise functions allocate
        fresh scratch buffers per call. So reusing one simulator across input
        frames of different shapes is supported. Combined with the
        ``apply_noise`` non-finite guard, any downstream wrapper that
        introduces ``NaN`` in ``blur_img`` (e.g. via aggressive Wiener-
        regularized deconvolution at small frame sizes) will fail loud rather
        than corrupt output. This test pins the shape-independence invariant
        so a future change that introduces shape-dependent state breaks here,
        not in user code.
        """
        simulator, img = _build_simulator()
        gsd = 3.19 / 160.0
        _, blur_a, _ = simulator.simulate_image(img, gsd=gsd)
        assert np.all(np.isfinite(blur_a))
        small = img[:64, :64].copy()
        _, blur_b, _ = simulator.simulate_image(small, gsd=gsd)
        assert np.all(np.isfinite(blur_b))
        assert blur_b.shape == small.shape
