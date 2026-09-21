import numbers

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData, NDData
from astropy.table import Table
from astropy.visualization import (
    AsymmetricPercentileInterval,
    BaseInterval,
    BaseStretch,
    LogStretch,
    ManualInterval,
)
from astropy.wcs import WCS

__all__ = ["ImageAPITest"]

DEFAULT_IMAGE_SHAPE = (100, 150)


class ImageAPITest:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(1234)
        return rng.random(DEFAULT_IMAGE_SHAPE)

    @pytest.fixture
    def wcs(self):
        # This is a copy/paste from the astropy 4.3.1 documentation...

        # Create a new WCS object.  The number of axes must be set
        # from the start
        w = WCS(naxis=2)

        # Set up an "Airy's zenithal" projection
        # Note: WCS is 1-based, not 0-based
        w.wcs.crpix = [-234.75, 8.3393]
        w.wcs.cdelt = np.array([-0.066667, 0.066667])
        w.wcs.crval = [0, -90]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w.wcs.set_pv([(2, 1, 45.0)])
        return w

    @pytest.fixture
    def catalog(self, wcs: WCS) -> Table:
        """
        A catalog fixture that returns an empty table with the
        expected columns.
        """
        rng = np.random.default_rng(45328975)
        # The image shape is (height, width), i.e. (y, x).
        x = rng.uniform(0, DEFAULT_IMAGE_SHAPE[1], size=10)
        y = rng.uniform(0, DEFAULT_IMAGE_SHAPE[0], size=10)
        coord = wcs.pixel_to_world(x, y)

        cat = Table(
            dict(
                x=x,
                y=y,
                coord=coord,
            )
        )
        return cat

    # This setup is run before each test, ensuring that there are no
    # side effects of one test on another
    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Subclasses MUST define ``image_widget_class`` -- doing so as a
        class variable does the trick.
        """
        self.image = self.image_widget_class()

    def _assert_empty_catalog_table(self, table):
        assert isinstance(table, Table)
        assert len(table) == 0
        assert sorted(table.colnames) == sorted(["x", "y", "coord"])

    def _get_catalog_labels_as_set(self):
        marks = self.image.catalog_labels
        return set(marks)

    @staticmethod
    def _image_data_as_array(image_data):
        """
        Extract a plain array from whatever ``get_image`` returned.

        The API requires ``get_image`` to return either something
        array-like or an object exposing the array through a ``data``
        attribute (`~astropy.nddata.NDData` and friends), so handle
        both forms here.
        """
        return np.asarray(getattr(image_data, "data", image_data))

    @pytest.mark.parametrize("load_type", ["fits", "nddata", "array"])
    def test_load(self, data, tmp_path, load_type):
        match load_type:
            case "fits":
                hdu = fits.PrimaryHDU(data=data)
                image_path = tmp_path / "test.fits"
                hdu.header["BUNIT"] = "adu"
                hdu.writeto(image_path)
                load_arg = image_path
            case "nddata":
                load_arg = NDData(data=data)
            case "array":
                load_arg = data

        self.image.load_image(load_arg)

        # The image must be registered under a (generated) label and its
        # data must be retrievable.
        assert len(self.image.image_labels) == 1
        retrieved = self.image.get_image()
        np.testing.assert_allclose(self._image_data_as_array(retrieved), data)

    def test_set_get_center_xy(self, data):
        self.image.load_image(data, image_label="test")
        self.image.set_viewport(center=(10, 10), image_label="test")  # X, Y
        vport = self.image.get_viewport(image_label="test")
        assert vport["center"] == (10, 10)
        assert vport["image_label"] == "test"

    def test_set_get_center_world(self, data, wcs):
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")
        self.image.set_viewport(
            center=SkyCoord(*wcs.wcs.crval, unit="deg"), image_label="test"
        )

        vport = self.image.get_viewport(image_label="test")
        assert isinstance(vport["center"], SkyCoord)
        assert vport["center"].ra.deg == pytest.approx(wcs.wcs.crval[0])
        assert vport["center"].dec.deg == pytest.approx(wcs.wcs.crval[1])

    @pytest.mark.parametrize("world", [True, False])
    def test_fov_initialization_from_image(self, data, wcs, world):
        # Check that the FOV is initialized to a reasonable value when an image
        # is loaded.
        self.image.load_image(
            NDData(
                data=data,
                wcs=wcs if world else None,
            ),
            image_label="test",
        )
        vport = self.image.get_viewport(image_label="test")
        if world:
            assert isinstance(vport["fov"], u.Quantity)
            assert vport["fov"].unit.physical_type == "angle"
            pixel_scale = np.mean(np.abs(wcs.wcs.cdelt))
            assert vport["fov"].value == pytest.approx(max(data.shape) * pixel_scale)
        else:
            assert isinstance(vport["fov"], numbers.Real)
            assert vport["fov"] == pytest.approx(max(data.shape))

    @pytest.mark.parametrize("shape", [(100, 150), (150, 100)])
    def test_fov_anisotropic_pixel_scale(self, shape):
        # Regression test for #89: the index into proj_plane_pixel_scales
        # is ordered by WCS pixel axis (0 = x, 1 = y), not by numpy shape
        # index (0 = y, 1 = x). Use an anisotropic pixel scale and a
        # rectangular image so that using the wrong axis gives the wrong
        # answer. Both a wide (nx > ny) and a tall (ny > nx) image are
        # checked so that hardcoding either axis index fails the test.
        rng = np.random.default_rng(1234)
        data = rng.random(shape)

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [50, 50]
        # RA scale 0.0002 deg/pix, Dec scale 0.0001 deg/pix
        wcs.wcs.cdelt = np.array([-0.0002, 0.0001])
        wcs.wcs.crval = [150.0, 30.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # The largest dimension sets the field of view; its pixel scale
        # is the RA scale for a wide image and the Dec scale for a tall
        # image.
        largest = max(shape)
        pixel_scale = 0.0002 if shape[1] >= shape[0] else 0.0001
        expected_fov_deg = largest * pixel_scale

        self.image.load_image(NDData(data=data, wcs=wcs), image_label="aniso")

        # Initial FOV should be the size of the largest dimension in degrees.
        vport = self.image.get_viewport(image_label="aniso")
        assert vport["fov"].to(u.degree).value == pytest.approx(expected_fov_deg)

        # sky -> pixel conversion in get_viewport should use the same axis.
        vport_pixel = self.image.get_viewport(image_label="aniso", sky_or_pixel="pixel")
        assert vport_pixel["fov"] == pytest.approx(largest)

        # pixel -> sky conversion in get_viewport should also use that axis.
        self.image.set_viewport(fov=largest, image_label="aniso")
        vport_sky = self.image.get_viewport(image_label="aniso", sky_or_pixel="sky")
        assert vport_sky["fov"].to(u.degree).value == pytest.approx(expected_fov_deg)

    def test_fov_sky_to_pixel_honors_quantity_unit(self, data):
        # Regression test: converting an angular fov to pixels must convert
        # the Quantity to degrees first, not assume it is already in degrees.
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [50, 50]
        wcs.wcs.cdelt = np.array([-0.0002, 0.0002])
        wcs.wcs.crval = [150.0, 30.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        # Whole image width (150 px * 0.0002 deg/px = 0.03 deg), in arcmin.
        fov = (max(data.shape) * 0.0002 * u.degree).to(u.arcmin)
        self.image.set_viewport(fov=fov, image_label="test")

        vport = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")
        assert vport["fov"] == pytest.approx(max(data.shape))

    def test_fov_wcs_with_arcsec_cunit(self, data):
        # A WCS whose CDELT/CUNIT are given in arcsec must give the same
        # results as the equivalent WCS in degrees. wcslib normalizes
        # CUNIT to degrees, so pixel scales derived from the WCS are
        # always in degrees.
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [50, 50]
        # 0.72 arcsec/pix = 0.0002 deg/pix
        wcs.wcs.cdelt = np.array([-0.72, 0.72])
        wcs.wcs.cunit = ["arcsec", "arcsec"]
        wcs.wcs.crval = [150.0, 30.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        expected_fov_deg = max(data.shape) * 0.0002
        vport = self.image.get_viewport(image_label="test", sky_or_pixel="sky")
        assert vport["fov"].to(u.degree).value == pytest.approx(expected_fov_deg)

        vport_pixel = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")
        assert vport_pixel["fov"] == pytest.approx(max(data.shape))

    @pytest.mark.parametrize("world", [True, False])
    def test_initial_center_is_image_center(self, data, wcs, world):
        # Regression test for #97: with 0-indexed pixel-center coordinates
        # the center of the image is ((width - 1) / 2, (height - 1) / 2),
        # not (width / 2, height / 2).
        self.image.load_image(
            NDData(data=data, wcs=wcs if world else None), image_label="test"
        )
        height, width = data.shape
        expected_center = ((width - 1) / 2, (height - 1) / 2)
        vport = self.image.get_viewport(
            image_label="test", sky_or_pixel="pixel" if world else None
        )
        assert vport["center"][0] == pytest.approx(expected_center[0])
        assert vport["center"][1] == pytest.approx(expected_center[1])

    def test_set_get_fov_pixel(self, data):
        # Set data first, since that is needed to determine zoom level
        self.image.load_image(data, image_label="test")

        self.image.set_viewport(fov=100, image_label="test")
        vport = self.image.get_viewport(image_label="test")
        assert vport["fov"] == 100
        assert vport["image_label"] == "test"

    def test_set_get_fov_world(self, data, wcs):
        # Set data first, since that is needed to determine zoom level
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        # Set the FOV in world coordinates
        self.image.set_viewport(fov=0.1 * u.deg, image_label="test")
        vport = self.image.get_viewport(image_label="test")
        assert isinstance(vport["fov"], u.Quantity)
        assert len(np.atleast_1d(vport["fov"])) == 1
        assert vport["fov"].unit.physical_type == "angle"
        fov_degree = vport["fov"].to(u.degree).value
        assert fov_degree == pytest.approx(0.1)

    def test_set_get_viewport_errors(self, data, wcs):
        # Test several of the expected errors that can be raised
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        # fov can be float or an angular Qunatity
        with pytest.raises(u.UnitTypeError, match="[Ii]ncorrect unit for fov"):
            self.image.set_viewport(fov=100 * u.meter, image_label="test")

        # try an fov that is completely the wrong type
        with pytest.raises(TypeError, match="[Ii]nvalid value for fov"):
            self.image.set_viewport(fov="not a valid value", image_label="test")

        # center can be a SkyCoord or a tuple of floats. Try a value that is neither
        with pytest.raises(TypeError, match="[Ii]nvalid value for center"):
            self.image.set_viewport(center="not a valid value", image_label="test")

        # Check that an error is raised if a label is provided that does not
        # match an image that is loaded.
        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.set_viewport(
                center=(10, 10), fov=100, image_label="not a valid label"
            )

        # Getting a viewport for an image_label that does not exist should
        # raise an error
        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.get_viewport(image_label="not a valid label")

        # If there are multiple images loaded, the image_label must be provided
        self.image.load_image(data, image_label="another test")

        with pytest.raises(ValueError, match="Multiple image labels defined"):
            self.image.get_viewport()

        # setting sky_or_pixel to something other than 'sky' or 'pixel' or None
        # should raise an error
        with pytest.raises(ValueError, match="[Ss]ky_or_pixel must be"):
            self.image.get_viewport(sky_or_pixel="not a valid value")

    def test_set_get_viewport_errors_because_no_wcs(self, data):
        # Check that errors are raised when they should be when calling
        # get_viewport when no WCS is present.

        # Load the data without a WCS
        self.image.load_image(data, image_label="test")

        # Set the viewport with a SkyCoord center
        with pytest.raises(TypeError, match="Center must be a tuple"):
            self.image.set_viewport(
                center=SkyCoord(ra=10, dec=20, unit="deg"), image_label="test"
            )

        # Set the viewport with a Quantity fov
        with pytest.raises(TypeError, match="FOV must be a float"):
            self.image.set_viewport(fov=100 * u.arcmin, image_label="test")

        # Try getting the viewport as sky
        with pytest.raises(ValueError, match="WCS is not set"):
            self.image.get_viewport(image_label="test", sky_or_pixel="sky")

    @pytest.mark.parametrize("world", [True, False])
    def test_viewport_is_defined_after_loading_image(self, tmp_path, data, wcs, world):
        # Check that the viewport is set to a default value when an image
        # is loaded, even if no viewport is explicitly set.

        # Load the image from FITS to ensure that at least one image with WCS
        # has been loaded from FITS.
        wcs = wcs if world else None
        ccd = CCDData(data=data, unit="adu", wcs=wcs)

        ccd_path = tmp_path / "test.fits"
        ccd.write(ccd_path)
        self.image.load_image(ccd_path)

        # Getting the viewport should not fail...
        vport = self.image.get_viewport()

        assert "center" in vport

        assert "fov" in vport
        assert "image_label" in vport
        # The unlabeled load used the shared default label, and get_viewport
        # reports that label.
        assert vport["image_label"] == self.image.image_labels[0]
        if world:
            assert isinstance(vport["center"], SkyCoord)
            # fov should be a Quantity since WCS is present
            assert isinstance(vport["fov"], u.Quantity)
        else:
            # No world, so center should be a tuple
            assert isinstance(vport["center"], tuple)
            # fov should be a float since no WCS
            assert isinstance(vport["fov"], numbers.Real)

    def test_set_get_viewport_no_image_label(self, data):
        # If there is only one image, the viewport should be able to be set
        # and retrieved without an image label.

        # Add an image without an image label
        self.image.load_image(data)

        # Set the viewport without an image label
        self.image.set_viewport(center=(10, 10), fov=100)

        # Getting the viewport again should return the same values
        vport = self.image.get_viewport()
        assert vport["center"] == (10, 10)
        assert vport["fov"] == 100
        # The unlabeled load used the shared default label.
        assert vport["image_label"] == self.image.image_labels[0]

    def test_set_get_viewport_single_label(self, data):
        # If there is only one image, the viewport should be able to be set
        # and retrieved without an image label as long as the image
        # has an image label.

        # Add an image with an image label
        self.image.load_image(data, image_label="test")

        # Getting the viewport should not fail...
        vport = self.image.get_viewport()
        assert "center" in vport
        assert "fov" in vport
        assert "image_label" in vport
        assert vport["image_label"] == "test"

        # Set the viewport with an image label
        self.image.set_viewport(center=(10, 10), fov=100)

        # Getting the viewport again should return the same values
        vport = self.image.get_viewport()
        assert vport["center"] == (10, 10)
        assert vport["fov"] == 100
        assert vport["image_label"] == "test"

    def test_get_viewport_sky_or_pixel(self, data, wcs):
        # Check that the viewport can be retrieved in both pixel and world
        # coordinates, depending on the WCS of the image.

        # Load the data with a WCS
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        input_center = SkyCoord(*wcs.wcs.crval, unit="deg")
        input_fov = 2 * u.arcmin
        self.image.set_viewport(center=input_center, fov=input_fov, image_label="test")

        # Get the viewport in pixel coordinates
        vport_pixel = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")
        # The WCS set up for the tests is 1-based, rather than the usual 0-based,
        # so we need to subtract 1 from the pixel coordinates.
        assert all(vport_pixel["center"] == (wcs.wcs.crpix - 1))
        # tbh, not at all sure what the fov should be in pixel coordinates,
        # so just check that it is a float.
        assert isinstance(vport_pixel["fov"], numbers.Real)

        # Get the viewport in world coordinates
        vport_world = self.image.get_viewport(image_label="test", sky_or_pixel="sky")
        assert vport_world["center"].separation(input_center) < 1e-5 * u.arcsec
        assert vport_world["fov"] == input_fov

    @pytest.mark.parametrize("sky_or_pixel", ["sky", "pixel"])
    def test_get_viewport_no_sky_or_pixel(self, data, wcs, sky_or_pixel):
        # Check that get_viewport returns the correct "default" sky_or_pixel
        # value when the result ought to be unambiguous.
        if sky_or_pixel == "sky":
            use_wcs = wcs
        else:
            use_wcs = None

        self.image.load_image(NDData(data=data, wcs=use_wcs), image_label="test")

        vport = self.image.get_viewport(image_label="test")
        match sky_or_pixel:
            case "sky":
                assert isinstance(vport["center"], SkyCoord)
                assert vport["fov"].unit.physical_type == "angle"
            case "pixel":
                assert isinstance(vport["center"], tuple)
                assert isinstance(vport["fov"], numbers.Real)

    def test_get_viewport_with_wcs_set_pixel_or_world(self, data, wcs):
        # Check that the viewport can be retrieved in both pixel and world
        # after setting with the opposite if the WCS is set.
        # Load the data with a WCS
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        # Set the viewport in world coordinates
        input_center = SkyCoord(*wcs.wcs.crval, unit="deg")
        input_fov = 2 * u.arcmin
        self.image.set_viewport(center=input_center, fov=input_fov, image_label="test")

        # Get the viewport in pixel coordinates
        vport_pixel = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")
        assert all(vport_pixel["center"] == (wcs.wcs.crpix - 1))
        assert isinstance(vport_pixel["fov"], numbers.Real)

        # Set the viewport in pixel coordinates
        input_center_pixel = (wcs.wcs.crpix[0], wcs.wcs.crpix[1])
        input_fov_pixel = 100  # in pixels
        self.image.set_viewport(
            center=input_center_pixel, fov=input_fov_pixel, image_label="test"
        )

        # Get the viewport in world coordinates
        vport_world = self.image.get_viewport(image_label="test", sky_or_pixel="sky")
        assert vport_world["center"] == wcs.pixel_to_world(*input_center_pixel)
        assert isinstance(vport_world["fov"], u.Quantity)

    def test_get_viewport_with_wcs_set_return_world(self, data, wcs):
        # Check that the viewport is retrieved with world coordinates
        # if sky_or_pixel is None (the default) and a WCS is present.

        # Load the data with a WCS
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        # Set the viewport in world coordinates
        input_center = SkyCoord(*wcs.wcs.crval, unit="deg")
        input_fov = 2 * u.degree
        self.image.set_viewport(center=input_center, fov=input_fov, image_label="test")
        # Get the viewport without specifying sky_or_pixel
        vport = self.image.get_viewport(image_label="test")
        assert isinstance(vport["center"], SkyCoord)
        assert vport["center"].separation(input_center) < 1e-5 * u.arcsec
        assert vport["fov"] == input_fov

        # Get the viewport in pixels so that we can set_viewport using it later.
        vport_pixel = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")

        # Set the viewport in pixel coordinates to the same size/center
        # as the world coordinates.
        self.image.set_viewport(**vport_pixel)
        # Get the viewport without specifying sky_or_pixel
        vport = self.image.get_viewport(image_label="test")
        assert isinstance(vport["center"], SkyCoord)
        assert vport["center"].separation(input_center) < 1e-5 * u.arcsec
        assert vport["fov"] == input_fov

    def test_viewport_round_trips(self, data, wcs):
        # Check that the viewport retrieved with get can be used to set
        # the viewport again, and that the values are the same.
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")

        self.image.set_viewport(center=(10, 10), fov=100, image_label="test")

        vport = self.image.get_viewport(image_label="test")
        assert isinstance(vport["center"], SkyCoord)

        # Set the viewport again using the values from the get_viewport
        self.image.set_viewport(**vport)
        # Get the viewport again and check that the values are the same
        vport2 = self.image.get_viewport(image_label="test")
        # Check the values of each item in the viewport dict, allowing
        # for small numerical differences in the center and fov.
        assert vport2["center"].separation(vport["center"]) < 1e-5 * u.arcsec
        assert vport2["fov"].value == pytest.approx(vport["fov"].value)
        assert vport2["image_label"] == vport["image_label"]

        # Complete the pixel -> sky -> pixel round trip: converting back to
        # pixel coordinates must recover the original pixel values.
        vport_pixel = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")
        assert vport_pixel["center"][0] == pytest.approx(10)
        assert vport_pixel["center"][1] == pytest.approx(10)
        assert vport_pixel["fov"] == pytest.approx(100)

    def test_set_catalog_style_before_catalog_data_raises_error(self):
        # Make sure that adding a catalog style before adding any catalog
        # data raises an error.
        with pytest.raises(
            ValueError, match="Must load a catalog before setting a catalog style"
        ):
            self.image.set_catalog_style(color="red", shape="circle", size=10)

    def test_set_get_catalog_style_no_labels(self, catalog):
        # Check that getting without setting returns a dict that contains
        # the minimum required keys

        required_style_keys = ["color", "shape", "size"]
        marker_style = self.image.get_catalog_style()
        for key in required_style_keys:
            assert key in marker_style

        # Add some data before setting a style
        self.image.load_catalog(catalog)
        # Check that setting a marker style works
        marker_settings = dict(color="red", shape="crosshair", size=10)
        self.image.set_catalog_style(**marker_settings.copy())

        retrieved_style = self.image.get_catalog_style()
        # Check that the marker style is set correctly
        for key, value in marker_settings.items():
            assert retrieved_style[key] == value

        # Check that set accepts the output of get
        self.image.set_catalog_style(**retrieved_style)

    def test_set_get_catalog_style_with_single_label(self, catalog):
        # Check that when there is only a single catalog label it is
        # not necessary to provide the label on get.
        self.image.load_catalog(catalog, catalog_label="test1")
        set_style_input = dict(
            catalog_label="test1", color="blue", shape="square", size=5
        )
        self.image.set_catalog_style(**set_style_input.copy())
        retrieved_style = self.image.get_catalog_style()

        assert set_style_input == retrieved_style

    def test_get_catalog_style_with_multiple_labels_raises_error(self, catalog):
        # Check that when there are multiple catalog labels, the
        # get_catalog_style method raises an error if no label is given.
        self.image.load_catalog(catalog, catalog_label="test1")
        self.image.load_catalog(catalog, catalog_label="test2")
        self.image.set_catalog_style(
            catalog_label="test1", color="blue", shape="square", size=5
        )
        self.image.set_catalog_style(
            catalog_label="test2", color="red", shape="circle", size=10
        )

        with pytest.raises(ValueError, match="Multiple catalog labels"):
            self.image.get_catalog_style()

    def test_set_get_catalog_style_preserves_extra_keywords(self, catalog):
        # Check that setting a catalog style with extra keywords preserves
        # those keywords.
        self.image.load_catalog(catalog)
        # The only required keywords are color, shape, and size.
        # Add some extra keyword to the style.
        style = dict(
            color="blue", shape="circle", size=10, extra_kw="extra_value", alpha=0.5
        )
        self.image.set_catalog_style(**style.copy())

        retrieved_style = self.image.get_catalog_style()
        del retrieved_style["catalog_label"]  # Remove the label
        assert retrieved_style == style

    def test_catalog_has_style_after_loading(self, catalog):
        # Check that loading a catalog sets a default style for that catalog.
        self.image.load_catalog(catalog, catalog_label="test1")

        retrieved_style = self.image.get_catalog_style(catalog_label="test1")
        assert isinstance(retrieved_style, dict)
        assert "color" in retrieved_style
        assert "shape" in retrieved_style
        assert "size" in retrieved_style

        # Loading again should have the same style
        self.image.load_catalog(catalog, catalog_label="test1")
        retrieved_style2 = self.image.get_catalog_style(catalog_label="test1")
        assert retrieved_style2 == retrieved_style

    @pytest.mark.parametrize("catalog_label", ["test1", None])
    def test_load_get_single_catalog_with_without_label(self, catalog, catalog_label):
        # Make sure we can get a single catalog with or without a label.
        self.image.load_catalog(
            catalog,
            x_colname="x",
            y_colname="y",
            skycoord_colname="coord",
            catalog_label=catalog_label,
            use_skycoord=False,
        )

        # Get the catalog without a label
        retrieved_catalog = self.image.get_catalog()
        assert (retrieved_catalog == catalog).all()

        # Get the catalog with a label if there is one
        if catalog_label is not None:
            retrieved_catalog = self.image.get_catalog(catalog_label=catalog_label)
            assert (retrieved_catalog == catalog).all()

    def test_load_catalog_does_not_modify_input_catalog(self, catalog):
        # Adding a catalog should not modify the input data table.
        orig_tab = catalog.copy()
        self.image.load_catalog(catalog)
        _ = self.image.get_catalog()
        assert (catalog == orig_tab).all()

    def test_load_multiple_catalogs(self, catalog):
        # Load and get multiple catalogs
        # Add a catalog
        self.image.load_catalog(
            catalog,
            x_colname="x",
            y_colname="y",
            catalog_label="test1",
        )
        # Add the catalog again under different name.
        self.image.load_catalog(
            catalog,
            x_colname="x",
            y_colname="y",
            catalog_label="test2",
        )

        assert sorted(self.image.catalog_labels) == ["test1", "test2"]

        # No guarantee markers will come back in the same order, so sort them.
        t1 = self.image.get_catalog(catalog_label="test1")
        # Sort before comparing
        t1.sort(["x", "y"])
        catalog.sort(["x", "y"])
        assert (t1["x"] == catalog["x"]).all()
        assert (t1["y"] == catalog["y"]).all()

        t2 = self.image.get_catalog(catalog_label="test2")
        # Sort before comparing
        t2.sort(["x", "y"])
        assert (t2["x"] == catalog["x"]).all()
        assert (t2["y"] == catalog["y"]).all()

        # get_catalog without a label should fail with multiple catalogs
        with pytest.raises(ValueError, match="Multiple catalog labels defined"):
            self.image.get_catalog()

        # if we remove one of the catalogs we should be able to get the
        # other one without a label.
        self.image.remove_catalog(catalog_label="test1")
        # Make sure test1 is really gone.
        assert self.image.catalog_labels == ("test2",)

        # Get without a catalog
        t2 = self.image.get_catalog()
        # Sort before comparing
        t2.sort(["x", "y"])
        assert (t2["x"] == catalog["x"]).all()
        assert (t2["y"] == catalog["y"]).all()

        # Check that retrieving a catalog that doesn't exist raises an
        # error rather than silently returning an empty result.
        with pytest.raises(ValueError, match="(?i)catalog label.*not found"):
            self.image.get_catalog(catalog_label="test1")

    def test_unlabeled_catalog_loads_share_one_default_label(self, catalog):
        # There is only ever a single unlabeled ("default") catalog: loading a
        # catalog without a label repeatedly replaces the previous unlabeled
        # catalog rather than accumulating new ones.
        self.image.load_catalog(catalog)
        self.image.load_catalog(catalog)

        labels = self.image.catalog_labels
        assert len(labels) == 1

        # The single default catalog is still reachable with no label.
        retrieved = self.image.get_catalog()
        assert len(retrieved) == len(catalog)

        # A labeled load coexists with the default one instead of replacing it.
        self.image.load_catalog(catalog, catalog_label="labeled")
        assert set(self.image.catalog_labels) == set(labels) | {"labeled"}
        assert len(self.image.catalog_labels) == 2

        # Another unlabeled load still replaces only the default catalog.
        self.image.load_catalog(catalog)
        assert len(self.image.catalog_labels) == 2

    def test_unknown_catalog_label_raises_and_does_not_pollute(self, catalog):
        # Regression tests for #113 (unknown explicit catalog labels must
        # raise instead of returning empty/default results) and #95 (failed
        # lookups must not create phantom catalog entries).
        self.image.load_catalog(catalog, catalog_label="real")

        with pytest.raises(ValueError, match="(?i)catalog label.*not found"):
            self.image.get_catalog(catalog_label="phantom")

        with pytest.raises(ValueError, match="(?i)catalog label.*not found"):
            self.image.get_catalog_style(catalog_label="phantom")

        with pytest.raises(ValueError, match="(?i)catalog label.*not found"):
            self.image.set_catalog_style(catalog_label="phantom", color="blue")

        # None of the failed calls above may have created a catalog entry...
        assert self.image.catalog_labels == ("real",)

        # ...so label-free calls still resolve to the single real catalog.
        assert self.image.get_catalog_style()["catalog_label"] == "real"

    def test_load_catalog_multiple_same_label(self, catalog):
        # Check that loading a catalog with the same label multiple times
        # does not raise an error and does not change the catalog.
        self.image.load_catalog(catalog, catalog_label="test1")
        self.image.load_catalog(catalog, catalog_label="test1")

        retrieved_catalog = self.image.get_catalog(catalog_label="test1")
        assert len(retrieved_catalog) == len(catalog)

    def test_load_catalog_with_skycoord_no_wcs(self, catalog, data):
        # Check that loading a catalog with skycoord but no x/y and
        # no WCS returns a catalog with None for x and y.
        self.image.load_image(data)

        # Remove x/y columns from the catalog
        del catalog["x", "y"]
        with pytest.raises(
            ValueError, match="Cannot use pixel coordinates without pixel columns"
        ):
            self.image.load_catalog(catalog)

    def test_load_catalog_with_use_skycoord_no_skycoord_no_wcs(self, catalog, data):
        # Check that loading a catalog with use_skycoord=True but no
        # skycoord column and no WCS raises an error.
        self.image.load_image(data)
        del catalog["coord"]  # Remove the skycoord column
        with pytest.raises(ValueError, match="Cannot use sky coordinates without"):
            self.image.load_catalog(catalog, use_skycoord=True)

    def test_load_catalog_with_xy_and_wcs(self, catalog, data, wcs):
        # Check that loading a catalog that wants to use sky coordinates,
        # has no coordinate column but has x/y and a WCS works.
        self.image.load_image(NDData(data=data, wcs=wcs))

        # Remove the skycoord column from the catalog
        del catalog["coord"]

        # Add the catalog with x/y and WCS
        self.image.load_catalog(catalog, use_skycoord=True)

        # Retrieve the catalog and check that the x and y columns are there
        retrieved_catalog = self.image.get_catalog()
        assert "x" in retrieved_catalog.colnames
        assert "y" in retrieved_catalog.colnames
        assert "coord" in retrieved_catalog.colnames

        # Check that the coordinates are correct
        coords = wcs.pixel_to_world(catalog["x"], catalog["y"])
        assert all(coords.separation(retrieved_catalog["coord"]) < 1e-9 * u.deg)

    def test_get_catalog_empty_on_fresh_viewer(self):
        # Regression test for #90: get_catalog on a viewer with no catalog
        # loaded must return an empty table with the requested column
        # names rather than raising.
        self._assert_empty_catalog_table(self.image.get_catalog())

        tab = self.image.get_catalog(
            x_colname="px", y_colname="py", skycoord_colname="sky"
        )
        assert len(tab) == 0
        assert sorted(tab.colnames) == sorted(["px", "py", "sky"])

    def test_load_get_catalog_custom_column_names(self, catalog):
        # Regression test for #90: a catalog loaded with non-default column
        # names must be retrievable with those names...
        renamed = catalog.copy()
        renamed.rename_columns(["x", "y", "coord"], ["px", "py", "sky"])
        self.image.load_catalog(
            renamed,
            x_colname="px",
            y_colname="py",
            skycoord_colname="sky",
            catalog_label="cat",
        )

        tab = self.image.get_catalog(
            x_colname="px", y_colname="py", skycoord_colname="sky", catalog_label="cat"
        )
        assert sorted(tab.colnames) == sorted(["px", "py", "sky"])
        np.testing.assert_array_equal(tab["px"], catalog["x"])
        np.testing.assert_array_equal(tab["py"], catalog["y"])

        # ...and get_catalog must return whatever column names are asked for.
        tab = self.image.get_catalog(catalog_label="cat")
        assert sorted(tab.colnames) == sorted(["x", "y", "coord"])
        np.testing.assert_array_equal(tab["x"], catalog["x"])

    def test_get_catalog_renames_columns(self, catalog):
        # Regression test for #90: renaming the position columns to the
        # requested names must actually happen instead of being a silent
        # no-op that returns the default names.
        self.image.load_catalog(catalog, catalog_label="cat")

        tab = self.image.get_catalog(
            x_colname="xcen",
            y_colname="ycen",
            skycoord_colname="sky",
            catalog_label="cat",
        )
        assert "xcen" in tab.colnames
        assert "ycen" in tab.colnames
        assert "sky" in tab.colnames
        assert "x" not in tab.colnames
        assert "y" not in tab.colnames
        assert "coord" not in tab.colnames
        np.testing.assert_array_equal(tab["xcen"], catalog["x"])
        np.testing.assert_array_equal(tab["ycen"], catalog["y"])

    def test_get_catalog_returns_copy(self, catalog):
        # Regression test for #90: the returned table must be a copy, so
        # that modifying it cannot corrupt the stored catalog.
        self.image.load_catalog(catalog, catalog_label="cat")

        tab = self.image.get_catalog(catalog_label="cat")
        original_value = tab["x"][0]
        tab["x"][0] = -999.0
        tab.remove_column("y")

        again = self.image.get_catalog(catalog_label="cat")
        assert again["x"][0] == original_value
        assert "y" in again.colnames

    def test_catalog_info_preserved_after_load(self, catalog):
        # Make sure that any catalog columns in addition to the position data
        # is preserved after loading a catalog.
        # Add a column with some extra information
        catalog["extra_info"] = np.arange(len(catalog))
        self.image.load_catalog(catalog, catalog_label="test1")
        # Retrieve the catalog and check that the extra column is there
        retrieved_catalog = self.image.get_catalog(catalog_label="test1")
        assert "extra_info" in retrieved_catalog.colnames
        assert (retrieved_catalog["extra_info"] == catalog["extra_info"]).all()

    def test_load_catalog_with_no_style_has_a_style(self, catalog):
        # Check that loading a catalog without a style sets a default style
        # for that catalog.
        self.image.load_catalog(catalog, catalog_label="test1")

        retrieved_style = self.image.get_catalog_style(catalog_label="test1")
        assert isinstance(retrieved_style, dict)
        assert "color" in retrieved_style
        assert "shape" in retrieved_style
        assert "size" in retrieved_style

    def test_load_catalog_with_style_sets_style(self, catalog):
        # Check that loading a catalog with a style sets the style
        # for that catalog.
        style = dict(color="blue", shape="square", size=10)
        self.image.load_catalog(
            catalog, catalog_label="test1", catalog_style=style.copy()
        )

        retrieved_style = self.image.get_catalog_style(catalog_label="test1")

        # Add catalog_label to the style for comparison
        style["catalog_label"] = "test1"
        assert retrieved_style == style

    def test_remove_catalog(self):
        with pytest.raises(ValueError, match="arf"):
            self.image.remove_catalog(catalog_label="arf")

    def test_remove_catalogs_name_all(self):
        data = np.arange(10).reshape(5, 2)
        tab = Table(data=data, names=["x", "y"])
        self.image.load_catalog(tab, catalog_label="test1", use_skycoord=False)
        self.image.load_catalog(tab, catalog_label="test2", use_skycoord=False)

        self.image.remove_catalog(catalog_label="*")
        self._assert_empty_catalog_table(self.image.get_catalog())

    def test_remove_all_catalogs_restores_fresh_state(self, catalog):
        # Regression test for #96: after remove_catalog("*") the viewer must
        # behave exactly like a freshly created viewer.
        self.image.load_catalog(catalog, catalog_label="test1")
        self.image.set_catalog_style(
            catalog_label="test1", color="blue", shape="square", size=10
        )
        self.image.load_catalog(catalog, catalog_label="test2")

        self.image.remove_catalog(catalog_label="*")

        fresh = self.image_widget_class()
        assert self.image.catalog_labels == fresh.catalog_labels == ()

        # The default style must be restored, including the required keys.
        style = self.image.get_catalog_style()
        assert style == fresh.get_catalog_style()
        for key in ("color", "shape", "size"):
            assert key in style

    def test_remove_catalog_does_not_accept_list(self):
        data = np.arange(10).reshape(5, 2)
        tab = Table(data=data, names=["x", "y"])
        self.image.load_catalog(tab, catalog_label="test1", use_skycoord=False)
        self.image.load_catalog(tab, catalog_label="test2", use_skycoord=False)

        with pytest.raises(
            TypeError, match="Cannot remove multiple catalogs from a list"
        ):
            self.image.remove_catalog(catalog_label=["test1", "test2"])

    def test_adding_catalog_as_world(self, data, wcs):
        ndd = NDData(data=data, wcs=wcs)
        self.image.load_image(ndd)

        # Add markers using world coordinates
        pixels = np.linspace(0, 100, num=10).reshape(5, 2)
        marks_pix = Table(data=pixels, names=["x", "y"], dtype=("float", "float"))
        marks_coords = wcs.pixel_to_world(marks_pix["x"], marks_pix["y"])
        mark_coord_table = Table(data=[marks_coords], names=["coord"])
        self.image.load_catalog(mark_coord_table, use_skycoord=True)
        result = self.image.get_catalog()
        # Check the x, y positions as long as we are testing things...
        # The first test had one entry that was zero, so any check
        # based on rtol will not work. Added a small atol to make sure
        # the test passes.
        np.testing.assert_allclose(result["x"], marks_pix["x"], atol=1e-9)
        np.testing.assert_allclose(result["y"], marks_pix["y"])
        np.testing.assert_allclose(
            result["coord"].ra.deg, mark_coord_table["coord"].ra.deg
        )
        np.testing.assert_allclose(
            result["coord"].dec.deg, mark_coord_table["coord"].dec.deg
        )

    @staticmethod
    def _make_tan_wcs(crval, crpix, scale=0.001):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = list(crpix)
        wcs.wcs.cdelt = [-scale, scale]
        wcs.wcs.crval = list(crval)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs

    def test_catalog_conversion_uses_single_image_wcs(self, data):
        # Regression test for #91: catalog pixel<->sky conversion must use
        # the WCS of the one loaded image, and later image loads must not
        # rewrite a previously loaded catalog's coordinates.
        wcs1 = self._make_tan_wcs(crval=(150.0, 30.0), crpix=(50.0, 50.0))
        wcs2 = self._make_tan_wcs(crval=(10.0, -45.0), crpix=(500.0, 500.0))

        self.image.load_image(NDData(data=data, wcs=wcs1), image_label="a")

        coord = wcs1.pixel_to_world([10.0, 20.0], [30.0, 40.0])
        self.image.load_catalog(Table(dict(coord=coord)), catalog_label="cat")

        # Loading a second image with a different WCS must not change the
        # already-computed pixel positions of the catalog.
        self.image.load_image(NDData(data=data, wcs=wcs2), image_label="b")

        tab = self.image.get_catalog(catalog_label="cat")
        np.testing.assert_allclose(tab["x"], [10.0, 20.0], atol=1e-6)
        np.testing.assert_allclose(tab["y"], [30.0, 40.0], atol=1e-6)

    def test_load_catalog_pixel_only_with_multiple_images(self, data):
        # A pixel-only catalog needs no WCS, so it must load without error
        # even with several images present, and get_catalog must return
        # the pixel positions. Whether (and how) a sky-coordinate column
        # is filled in from an image WCS in this situation is
        # backend-dependent and not part of the contract.
        wcs1 = self._make_tan_wcs(crval=(150.0, 30.0), crpix=(50.0, 50.0))
        wcs2 = self._make_tan_wcs(crval=(10.0, -45.0), crpix=(500.0, 500.0))
        self.image.load_image(NDData(data=data, wcs=wcs1), image_label="a")
        self.image.load_image(NDData(data=data, wcs=wcs2), image_label="b")

        self.image.load_catalog(Table(dict(x=[1.0], y=[2.0])), catalog_label="cat")

        tab = self.image.get_catalog(catalog_label="cat")
        np.testing.assert_allclose(tab["x"], [1.0])
        np.testing.assert_allclose(tab["y"], [2.0])

    @pytest.mark.parametrize("load_order", [("nowcs", "withwcs"), ("withwcs", "nowcs")])
    def test_get_viewport_default_uses_requested_image_wcs(self, data, wcs, load_order):
        # Regression test for #93: the sky-vs-pixel default in get_viewport
        # must be decided from the WCS of the *requested* image, not from
        # the WCS of whichever image was loaded last.
        for label in load_order:
            use_wcs = wcs if label == "withwcs" else None
            self.image.load_image(NDData(data=data, wcs=use_wcs), image_label=label)

        vport = self.image.get_viewport(image_label="nowcs")
        assert isinstance(vport["center"], tuple)
        assert isinstance(vport["fov"], numbers.Real)

        vport = self.image.get_viewport(image_label="withwcs")
        assert isinstance(vport["center"], SkyCoord)
        assert isinstance(vport["fov"], u.Quantity)

    def test_stretch(self, data):
        self.image.load_image(data, image_label="test")
        original_stretch = self.image.get_stretch()

        with pytest.raises(TypeError, match=r"Stretch.*not valid.*"):
            self.image.set_stretch("not a valid value")

        # A bad value should leave the stretch unchanged. Compare by type
        # rather than identity so that backends returning an equivalent
        # object pass.
        assert isinstance(self.image.get_stretch(), type(original_stretch))

        self.image.set_stretch(LogStretch())
        # A valid value should change the stretch
        assert isinstance(self.image.get_stretch(), LogStretch)

    def test_cuts(self, data):
        # Load an image first: the interface only promises behavior of
        # set_cuts for loaded images.
        self.image.load_image(data)

        with pytest.raises(TypeError, match="[mM]ust be"):
            self.image.set_cuts("not a valid value")

        with pytest.raises(TypeError, match="[mM]ust be"):
            self.image.set_cuts((1, 10, 100))

        self.image.set_cuts(AsymmetricPercentileInterval(0.1, 99.9))
        assert isinstance(self.image.get_cuts(), AsymmetricPercentileInterval)

        self.image.set_cuts((10, 100))
        assert isinstance(self.image.get_cuts(), ManualInterval)
        assert self.image.get_cuts().get_limits(data) == (10, 100)

    def test_stretch_cuts_labels(self, data):
        # Check that stretch and cuts can be set with labels
        self.image.load_image(data, image_label="test")

        # Set stretch and cuts with labels
        self.image.set_stretch(LogStretch(), image_label="test")
        self.image.set_cuts((10, 100), image_label="test")

        # Get stretch and cuts with labels
        stretch = self.image.get_stretch(image_label="test")
        cuts = self.image.get_cuts(image_label="test")

        assert isinstance(stretch, LogStretch)
        assert isinstance(cuts, ManualInterval)
        assert cuts.get_limits(data) == (10, 100)

    def test_stretch_cuts_are_set_after_loading_image(self, data):
        # Check that stretch and cuts are set to default values after loading an image
        self.image.load_image(data, image_label="test")

        stretch = self.image.get_stretch(image_label="test")
        cuts = self.image.get_cuts(image_label="test")

        # Backends can set whatever stretch and cuts they want, so
        # we just check that they are instances of the expected classes.
        assert isinstance(stretch, BaseStretch)
        assert isinstance(cuts, BaseInterval)

    def test_stretch_cuts_errors(self, data):
        # Check that errors are raised when trying to get or set stretch or cuts
        # for an image label that does not exist.
        self.image.load_image(data, image_label="test")

        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.get_stretch(image_label="not a valid label")

        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.get_cuts(image_label="not a valid label")

        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.set_stretch(LogStretch(), image_label="not a valid label")

        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.set_cuts((10, 100), image_label="not a valid label")

    @pytest.mark.parametrize(
        "set_method, value",
        [("set_cuts", (10, 100)), ("set_stretch", LogStretch())],
    )
    def test_set_cuts_stretch_preserves_viewport(self, data, set_method, value):
        # Changing the color mapping (cuts or stretch) must never change the
        # viewport (center/fov/rotation).
        self.image.load_image(data, image_label="test")
        self.image.set_viewport(center=(30, 40), fov=20, image_label="test")
        before = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")

        getattr(self.image, set_method)(value, image_label="test")

        after = self.image.get_viewport(image_label="test", sky_or_pixel="pixel")
        assert after == before

    def test_set_get_colormap(self, data):
        # Check setting and getting with a single image label.
        self.image.load_image(data, image_label="test")
        cmap_desired = "gray"
        self.image.set_colormap(cmap_desired)
        assert self.image.get_colormap() == cmap_desired

        # Check that the colormap can be set with an image label
        new_cmap = "viridis"
        self.image.set_colormap(new_cmap, image_label="test")
        assert self.image.get_colormap(image_label="test") == new_cmap

    def test_set_colormap_errors(self, data):
        # Check that setting a colormap raises an error if the colormap
        # is not in the list of allowed colormaps.
        self.image.load_image(data, image_label="test")

        # A colormap name the backend does not support must raise a
        # ValueError.
        with pytest.raises(ValueError, match="not a valid"):
            self.image.set_colormap("this-is-not-a-colormap", image_label="test")

        # Check that getting a colormap for an image label that does not exist
        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.get_colormap(image_label="not a valid label")

        # Check that setting a colormap without an image label fails
        # when there is more than one image label
        self.image.load_image(data, image_label="another test")
        with pytest.raises(ValueError, match="Multiple image labels defined"):
            self.image.set_colormap("gray")

        # Same for getting the colormap without an image label
        with pytest.raises(ValueError, match="Multiple image labels defined"):
            self.image.get_colormap()

    def test_save(self, tmp_path, data):
        # Saving is only meaningful once an image has been loaded.
        self.image.load_image(data)
        filename = tmp_path / "woot.png"
        self.image.save(filename)
        assert filename.is_file()

    def test_save_overwrite(self, tmp_path, data):
        # Saving is only meaningful once an image has been loaded.
        self.image.load_image(data)
        filename = tmp_path / "woot.png"

        # First write should be fine
        self.image.save(filename)
        assert filename.is_file()

        # Second write should raise an error because file exists
        with pytest.raises(FileExistsError):
            self.image.save(filename)

        # Using overwrite should save successfully
        self.image.save(filename, overwrite=True)

    def test_image_labels(self, data):
        # the test viewer begins with a default empty image
        assert len(self.image.image_labels) == 0
        assert isinstance(self.image.image_labels, tuple)

        self.image.load_image(data, image_label="test")
        assert len(self.image.image_labels) == 1
        assert self.image.image_labels[-1] == "test"

    def test_unlabeled_image_loads_share_one_default_label(self, data):
        # There is only ever a single unlabeled ("default") image: loading an
        # image without a label repeatedly replaces the previous unlabeled
        # image rather than accumulating new ones.
        self.image.load_image(data)
        self.image.load_image(data * 2)

        labels = self.image.image_labels
        assert len(labels) == 1

        # The single default image is still reachable with no label, and the
        # most recent unlabeled load is the one that is kept.
        np.testing.assert_allclose(self.image.get_image(), data * 2)

        # A labeled load coexists with the default one instead of replacing it.
        self.image.load_image(data, image_label="labeled")
        assert set(self.image.image_labels) == set(labels) | {"labeled"}
        assert len(self.image.image_labels) == 2
        for label in self.image.image_labels:
            assert self.image.get_image(image_label=label) is not None

        # Another unlabeled load still replaces only the default image.
        self.image.load_image(data * 3)
        assert len(self.image.image_labels) == 2
        default_label = (set(self.image.image_labels) - {"labeled"}).pop()
        np.testing.assert_allclose(
            self.image.get_image(image_label=default_label), data * 3
        )

    def test_empty_viewer_image_operations_raise(self):
        # Regression test for #94: operations that need an image must raise
        # on a viewer with no image loaded instead of silently succeeding.
        image_getters = [
            "get_image",
            "get_viewport",
            "get_stretch",
            "get_cuts",
            "get_colormap",
        ]
        for getter in image_getters:
            with pytest.raises(ValueError, match="[Nn]o image"):
                getattr(self.image, getter)()

        with pytest.raises(ValueError, match="[Nn]o image"):
            self.image.set_colormap("viridis")

        with pytest.raises(ValueError, match="[Nn]o image"):
            self.image.set_stretch(LogStretch())

        with pytest.raises(ValueError, match="[Nn]o image"):
            self.image.set_cuts((10, 100))

        with pytest.raises(ValueError, match="[Nn]o image"):
            self.image.set_viewport(center=(10, 10), fov=100)

        # The same is true for removing a catalog when none is loaded.
        with pytest.raises(ValueError, match="[Nn]o catalog"):
            self.image.remove_catalog()

    def test_get_image(self, data):
        self.image.load_image(data, image_label="test")

        # The type of the return value is not specified in the API, but the
        # data it holds must match what was loaded.
        assert self.image.get_image() is not None
        np.testing.assert_allclose(
            self._image_data_as_array(self.image.get_image(image_label="test")), data
        )

        retrieved_image = self.image.get_image(image_label="test")

        # The retrieved image must be re-loadable, and survive the round trip.
        self.image.load_image(retrieved_image, image_label="another test")
        np.testing.assert_allclose(
            self._image_data_as_array(self.image.get_image(image_label="another test")),
            data,
        )

        with pytest.raises(ValueError, match="[Ii]mage label.*not found"):
            self.image.get_image(image_label="not a valid label")

    def test_all_methods_accept_additional_kwargs(self, data, catalog, tmp_path):
        """
        Make sure all methods accept additional keyword arguments
        that are not defined in the protocol.
        """
        from astro_image_display_api import ImageViewerInterface

        all_methods_and_attributes = ImageViewerInterface.__protocol_attrs__

        all_methods = [
            method
            for method in all_methods_and_attributes
            if callable(getattr(self.image, method))
        ]

        # Make a small dictionary keys that are random characters
        additional_kwargs = {k: f"value{k}" for k in ["fsda", "urioeh", "m898h]"]}

        # Make a dictionary of the required arguments for any methods that have required
        # argument
        required_args = dict(
            load_image=data,
            set_cuts=(10, 100),
            set_stretch=LogStretch(),
            set_colormap="viridis",
            save=tmp_path / "test.png",
            load_catalog=catalog,
        )

        failed_methods = []

        # Take out the loading methods because they must happen first and take out
        # remove_catalog because it must happen last.
        all_methods = list(
            set(all_methods) - set(["load_image", "load_catalog", "remove_catalog"])
        )

        # Load an image and a catalog first since other methods require these
        # have been done
        try:
            self.image.load_image(required_args["load_image"], **additional_kwargs)
        except TypeError as e:
            if "required positional argument" not in str(e):
                # If the error is not about a missing required argument, we
                # consider it a failure.
                failed_methods.append("load_image")
            else:
                raise e

        try:
            self.image.load_catalog(required_args["load_catalog"], **additional_kwargs)
        except TypeError as e:
            if "required positional argument" not in str(e):
                # If the error is not about a missing required argument, we
                # consider it a failure.
                failed_methods.append("load_catalog")
            else:
                raise e

        if not failed_methods:
            # No point in running some of these if setting image or catalog has failed.
            # Run remove_catalog last so that it does not interfere with the
            # other methods that require an image or catalog to be loaded.
            for method in all_methods + ["remove_catalog"]:
                # Call each method with the required arguments and additional kwargs
                # Accumulate the failures and report them at the end
                try:
                    if method in required_args:
                        # If the method has required arguments, call it with those
                        getattr(self.image, method)(
                            required_args[method], **additional_kwargs
                        )
                    else:
                        # If the method does not have required arguments, just call it
                        # with additional kwargs
                        getattr(self.image, method)(**additional_kwargs)
                except TypeError as e:
                    if "required positional argument" not in str(e):
                        # If the error is not about a missing required argument, we
                        # consider it a failure.
                        failed_methods.append(method)
                    else:
                        raise e

        else:
            failed_methods.append(
                "No other methods were tested because the ones above failed."
            )

        assert not failed_methods, (
            "The following methods failed when called with additional kwargs:\n\t"
            f"{'\n\t'.join(failed_methods)}"
        )

    def test_parameter_names_match_interface(self):
        """
        Check that every method has the parameter names, in the order, that
        the interface defines.

        The parameter names are part of the documented API -- users call
        these methods by keyword -- and because every method also accepts
        ``**kwargs``, a wrongly named implementation parameter does not
        raise a ``TypeError`` but silently swallows the keyword argument.
        That makes signature inspection, not calling, the only reliable
        check.
        """
        import inspect

        from astro_image_display_api import ImageViewerInterface

        mismatches = []
        for name in ImageViewerInterface.__protocol_attrs__:
            interface_attr = getattr(ImageViewerInterface, name)
            if not callable(interface_attr):
                # Properties take no arguments, so there is nothing to check.
                continue
            implementation_attr = getattr(type(self.image), name)

            def named_parameters(func):
                return [
                    parameter.name
                    for parameter in inspect.signature(func).parameters.values()
                    if parameter.kind
                    not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
                    and parameter.name != "self"
                ]

            interface_names = named_parameters(interface_attr)
            implementation_names = named_parameters(implementation_attr)

            # The implementation must have the interface's parameters, with
            # the same names, in the same order, before any extra parameters
            # of its own.
            if implementation_names[: len(interface_names)] != interface_names:
                mismatches.append(
                    f"{name}: interface has parameters {interface_names}, "
                    f"implementation has {implementation_names}"
                )

        assert not mismatches, (
            "The following methods have parameter names that do not match "
            "the interface definition:\n\t" + "\n\t".join(mismatches)
        )

    def test_every_method_attribute_has_docstring(self):
        """
        Check that every method and attribute in the protocol has a docstring.
        """
        from astro_image_display_api import ImageViewerInterface

        all_methods_and_attributes = ImageViewerInterface.__protocol_attrs__

        method_attrs_no_docs = []

        for method in all_methods_and_attributes:
            attr = getattr(self.image, method)
            # Make list of methods and attributes that have no docstring
            # and assert that list is empty at the end of the test.
            if not attr.__doc__:
                method_attrs_no_docs.append(method)

        assert not method_attrs_no_docs, (
            "The following methods and attributes have no docstring:\n\t"
            f"{'\n\t'.join(method_attrs_no_docs)}"
        )
