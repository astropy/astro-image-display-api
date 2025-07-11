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
        x = rng.uniform(0, DEFAULT_IMAGE_SHAPE[0], size=10)
        y = rng.uniform(0, DEFAULT_IMAGE_SHAPE[1], size=10)
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
        assert vport["image_label"] is None
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
        assert vport["image_label"] is None

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
        assert vport_world["center"] == input_center
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

    def test_viewport_round_trips(self, data, wcs):
        # Check that the viewport retrieved with get can be used to set
        # the viewport again, and that the values are the same.
        self.image.load_image(NDData(data=data, wcs=wcs), image_label="test")
        self.image.set_viewport(center=(10, 10), fov=100, image_label="test")
        vport = self.image.get_viewport(image_label="test")
        # Set the viewport again using the values from the get_viewport
        self.image.set_viewport(**vport)
        # Get the viewport again and check that the values are the same
        vport2 = self.image.get_viewport(image_label="test")
        assert vport2 == vport

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

        with pytest.raises(ValueError, match="Multiple catalog styles"):
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
        with pytest.raises(ValueError, match="Multiple catalog styles defined."):
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

        # Check that retrieving a marker set that doesn't exist returns
        # an empty table with the right columns
        tab = self.image.get_catalog(catalog_label="test1")
        self._assert_empty_catalog_table(tab)

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

    def test_stretch(self):
        original_stretch = self.image.get_stretch()

        with pytest.raises(TypeError, match=r"Stretch.*not valid.*"):
            self.image.set_stretch("not a valid value")

        # A bad value should leave the stretch unchanged
        assert self.image.get_stretch() is original_stretch

        self.image.set_stretch(LogStretch())
        # A valid value should change the stretch
        assert self.image.get_stretch() is not original_stretch
        assert isinstance(self.image.get_stretch(), LogStretch)

    def test_cuts(self, data):
        with pytest.raises(TypeError, match="[mM]ust be"):
            self.image.set_cuts("not a valid value")

        with pytest.raises(TypeError, match="[mM]ust be"):
            self.image.set_cuts((1, 10, 100))

        # Setting using histogram requires data
        self.image.load_image(data)
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

    def test_save(self, tmp_path):
        filename = tmp_path / "woot.png"
        self.image.save(filename)
        assert filename.is_file()

    def test_save_overwrite(self, tmp_path):
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

    def test_get_image(self, data):
        self.image.load_image(data, image_label="test")

        # currently the type is not specified in the API
        assert self.image.get_image() is not None
        assert self.image.get_image(image_label="test") is not None

        retrieved_image = self.image.get_image(image_label="test")

        self.image.load_image(retrieved_image, image_label="another test")
        assert self.image.get_image(image_label="another test") is not None

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
