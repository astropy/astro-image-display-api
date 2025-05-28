
import pytest

import numpy as np

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table, vstack
from astropy import units as u
from astropy.wcs import WCS
from astropy.visualization import AsymmetricPercentileInterval, LogStretch, ManualInterval

__all__ = ['ImageWidgetAPITest']


class ImageWidgetAPITest:
    cursor_error_classes = (ValueError)

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(1234)
        return rng.random((100, 100))

    @pytest.fixture
    def wcs(self):
        # This is a copy/paste from the astropy 4.3.1 documentation...

        # Create a new WCS object.  The number of axes must be set
        # from the start
        w = WCS(naxis=2)

        # Set up an "Airy's zenithal" projection
        w.wcs.crpix = [-234.75, 8.3393]
        w.wcs.cdelt = np.array([-0.066667, 0.066667])
        w.wcs.crval = [0, -90]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w.wcs.set_pv([(2, 1, 45.0)])
        return w

    @pytest.fixture
    def catalog(self, setup: None, wcs: WCS) -> Table:
        """
        A catalog fixture that returns an empty table with the
        expected columns.
        """
        rng = np.random.default_rng(45328975)
        x = rng.uniform(0, self.image.image_width, size=10)
        y = rng.uniform(0, self.image.image_height, size=10)
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
        self.image = self.image_widget_class(image_width=250, image_height=100)

    def _assert_empty_catalog_table(self, table):
        assert isinstance(table, Table)
        assert len(table) == 0
        assert sorted(table.colnames) == sorted(['x', 'y', 'coord', 'marker name'])

    def _get_catalog_names_as_set(self):
        marks = self.image.get_catalog_names()
        return set(marks)

    def test_width_height(self):
        assert self.image.image_width == 250
        assert self.image.image_height == 100

        width = 200
        height = 300
        self.image.image_width = width
        self.image.image_height = height
        assert self.image.image_width == width
        assert self.image.image_height == height

    @pytest.mark.parametrize("load_type", ["fits", "nddata", "array"])
    def test_load(self, data, tmp_path, load_type):
        match load_type:
            case "fits":
                hdu = fits.PrimaryHDU(data=data)
                image_path = tmp_path / 'test.fits'
                hdu.header["BUNIT"] = "adu"
                hdu.writeto(image_path)
                load_arg = image_path
            case "nddata":
                load_arg = NDData(data=data)
            case "array":
                load_arg = data

        self.image.load_image(load_arg)

    def test_center_on(self):
        self.image.center_on((10, 10))  # X, Y

    def test_offset_by(self, data, wcs):
        self.image.offset_by(10, 10)  # dX, dY

        # Testing offset by WCS requires a WCS. The viewer will (or ought to
        # have) taken care of setting up the WCS internally if initialized with
        # an NDData that has a WCS.
        ndd = NDData(data=data, wcs=wcs)
        self.image.load_image(ndd)

        self.image.offset_by(10 * u.arcmin, 10 * u.arcmin)

        # A mix of pixel and sky should produce an error
        with pytest.raises(u.UnitConversionError, match='are not convertible'):
            self.image.offset_by(10 * u.arcmin, 10)

        # A mix of inconsistent units should produce an error
        with pytest.raises(u.UnitConversionError, match='are not convertible'):
            self.image.offset_by(1 * u.arcsec, 1 * u.AA)

    def test_zoom_level(self, data):
        # Set data first, since that is needed to determine zoom level
        self.image.load_image(data)
        self.image.zoom_level = 5
        assert self.image.zoom_level == 5

    def test_zoom(self):
        self.image.zoom_level = 3
        self.image.zoom(2)
        assert self.image.zoom_level == 6  # 3 x 2

    def test_set_catalog_style_before_catalog_data_raises_error(self):
        # Make sure3 that adding a catalog style before adding any catalog
        # data raises an error.
        with pytest.raises(
            ValueError,
            match='Must load a catalog before setting a catalog style'
        ):
            self.image.set_catalog_style(color='red', marker='x', size=10)

    def test_set_get_catalog_style_no_labels(self, catalog):
        # Check that getting without setting returns a dict that contains
        # the minimum required keys

        required_style_keys = ['color', 'shape', 'size']
        marker_style = self.image.get_catalog_style()
        for key in required_style_keys:
            assert key in marker_style

        # Add some data before setting a style
        self.image.load_catalog(catalog)
        # Check that setting a marker style works
        marker_settings = dict(color='red', marker='x', size=10)
        self.image.set_catalog_style(**marker_settings)

        retrieved_style = self.image.get_catalog_style()
        # Check that the marker style is set correctly
        for key, value in marker_settings.items():
            assert retrieved_style[key] == value

        # Check that set accepts the output of get
        self.image.set_catalog_style(**retrieved_style)

    def test_set_get_catalog_style_with_single_label(self, catalog):
        # Check that when there is only a single catalog label it is
        # not necessary to provide the label on get.
        self.image.load_catalog(catalog, catalog_label='test1')
        set_style_input = dict(catalog_label='test1', color='blue',
                                     shape='square', size=5)
        self.image.set_catalog_style(**set_style_input)
        retrieved_style = self.image.get_catalog_style()

        assert set_style_input == retrieved_style

    def test_get_catalog_style_with_multiple_labels_raises_error(self, catalog):
        # Check that when there are multiple catalog labels, the
        # get_catalog_style method raises an error if no label is given.
        self.image.load_catalog(catalog, catalog_label='test1')
        self.image.load_catalog(catalog, catalog_label='test2')
        self.image.set_catalog_style(catalog_label='test1', color='blue',
                                     shape='square', size=5)
        self.image.set_catalog_style(catalog_label='test2', color='red',
                                     shape='circle', size=10)

        with pytest.raises(ValueError, match='Multiple catalog styles'):
            self.image.get_catalog_style()

    def test_load_catalog(self):
        data = np.arange(10).reshape(5, 2)
        orig_tab = Table(data=data, names=['x', 'y'], dtype=('float', 'float'))
        tab = Table(data=data, names=['x', 'y'], dtype=('float', 'float'))
        self.image.load_catalog(
            tab,
            x_colname='x',
            y_colname='y',
            skycoord_colname='coord',
            catalog_label='test1',
            use_skycoord=False
        )


        # Regression test for GitHub Issue 45:
        # Adding markers should not modify the input data table.
        assert (tab == orig_tab).all()

        # Add more markers under different name.
        self.image.load_catalog(
            tab,
            x_colname='x',
            y_colname='y',
            skycoord_colname='coord',
            catalog_label='test2',
            use_skycoord=False
        )

        marknames = self._get_catalog_names_as_set()
        assert marknames == set(['test1', 'test2'])

        # No guarantee markers will come back in the same order, so sort them.
        t1 = self.image.get_catalog(catalog_label='test1')
        # Sort before comparing
        t1.sort('x')
        tab.sort('x')
        assert np.all(t1['x'] == tab['x'])
        assert (t1['y'] == tab['y']).all()

        t2 = self.image.get_catalog(catalog_label="test2")
        # Sort before comparing
        t2.sort(['x', 'y'])
        tab.sort(['x', 'y'])
        assert (t2['x'] == tab['x']).all()
        assert (t2['y'] == tab['y']).all()

        self.image.remove_catalog(catalog_label='test1')
        marknames = self._get_catalog_names_as_set()
        assert marknames == set(['test2'])

        # Add markers with no marker name and check we can retrieve them
        # using the default marker name
        self.image.load_catalog(
            tab,
            x_colname='x',
            y_colname='y',
            skycoord_colname='coord',
            use_skycoord=False
        )
        # Don't care about the order of the marker names so use set instead of
        # list.
        marknames = self._get_catalog_names_as_set()
        assert (set(marknames) == set(['test2']))

        # Clear markers to not pollute other tests.
        self.image.remove_catalog(catalog_label='*')
        marknames = self._get_catalog_names_as_set()
        assert len(marknames) == 0
        self._assert_empty_catalog_table(self.image.get_catalog())
        # Check that no markers remain after clearing
        tab = self.image.get_catalog()
        self._assert_empty_catalog_table(tab)

        # Check that retrieving a marker set that doesn't exist returns
        # an empty table with the right columns
        tab = self.image.get_catalog(catalog_label='test1')
        self._assert_empty_catalog_table(tab)

    def test_catalog_info_preserved_after_load(self, catalog):
        # Make sure that any catalog columns in addition to the position data
        # is preserved after loading a catalog.
        # Add a column with some extra information
        catalog['extra_info'] = np.arange(len(catalog))
        self.image.load_catalog(catalog, catalog_label='test1')
        # Retrieve the catalog and check that the extra column is there
        retrieved_catalog = self.image.get_catalog(catalog_label='test1')
        assert 'extra_info' in retrieved_catalog.colnames
        assert (retrieved_catalog['extra_info'] == catalog['extra_info']).all()

    def test_remove_catalog(self):
        with pytest.raises(ValueError, match='arf'):
            self.image.remove_catalog(catalog_label='arf')

    def test_remove_catalogs_name_all(self):
        data = np.arange(10).reshape(5, 2)
        tab = Table(data=data, names=['x', 'y'])
        self.image.load_catalog(tab, catalog_label='test1', use_skycoord=False)
        self.image.load_catalog(tab, catalog_label='test2', use_skycoord=False)

        self.image.remove_catalog(catalog_label='*')
        self._assert_empty_catalog_table(self.image.get_catalog())

    def test_remove_catalog_does_not_accept_list(self):
        data = np.arange(10).reshape(5, 2)
        tab = Table(data=data, names=['x', 'y'])
        self.image.load_catalog(tab, catalog_label='test1', use_skycoord=False)
        self.image.load_catalog(tab, catalog_label='test2', use_skycoord=False)

        with pytest.raises(
            ValueError,
            match='Cannot remove multiple catalogs from a list'
        ):
            self.image.remove_catalog(catalog_label=['test1', 'test2'])

    def test_adding_catalog_as_world(self, data, wcs):
        ndd = NDData(data=data, wcs=wcs)
        self.image.load_image(ndd)

        # Add markers using world coordinates
        pixels = np.linspace(0, 100, num=10).reshape(5, 2)
        marks_pix = Table(data=pixels, names=['x', 'y'], dtype=('float', 'float'))
        marks_coords = wcs.pixel_to_world(marks_pix['x'], marks_pix['y'])
        mark_coord_table = Table(data=[marks_coords], names=['coord'])
        self.image.load_catalog(mark_coord_table, use_skycoord=True)
        result = self.image.get_catalog()
        # Check the x, y positions as long as we are testing things...
        # The first test had one entry that was zero, so any check
        # based on rtol will not work. Added a small atol to make sure
        # the test passes.
        np.testing.assert_allclose(result['x'], marks_pix['x'], atol=1e-9)
        np.testing.assert_allclose(result['y'], marks_pix['y'])
        np.testing.assert_allclose(result['coord'].ra.deg,
                                   mark_coord_table['coord'].ra.deg)
        np.testing.assert_allclose(result['coord'].dec.deg,
                                   mark_coord_table['coord'].dec.deg)

    def test_stretch(self):
        original_stretch = self.image.get_stretch()

        with pytest.raises(ValueError, match=r'Stretch.*not valid.*'):
            self.image.set_stretch('not a valid value')

        # A bad value should leave the stretch unchanged
        assert self.image.get_stretch() is original_stretch

        self.image.set_stretch(LogStretch())
        # A valid value should change the stretch
        assert self.image.get_stretch() is not original_stretch
        assert isinstance(self.image.get_stretch(), LogStretch)

    def test_cuts(self, data):
        with pytest.raises(ValueError, match='[mM]ust be'):
            self.image.set_cuts('not a valid value')

        with pytest.raises(ValueError, match='[mM]ust be'):
            self.image.set_cuts((1, 10, 100))

        # Setting using histogram requires data
        self.image.load_image(data)
        self.image.set_cuts(AsymmetricPercentileInterval(0.1, 99.9))
        assert isinstance(self.image.get_cuts(), AsymmetricPercentileInterval)

        self.image.set_cuts((10, 100))
        assert isinstance(self.image.get_cuts(), ManualInterval)
        assert self.image.get_cuts().get_limits(data) == (10, 100)

    @pytest.mark.skip(reason="Not clear whether colormap is part of the API")
    def test_colormap(self):
        cmap_desired = 'gray'
        cmap_list = self.image.colormap_options
        assert len(cmap_list) > 0 and cmap_desired in cmap_list
        self.image.set_colormap(cmap_desired)

    def test_cursor(self):
        assert self.image.cursor in self.image.ALLOWED_CURSOR_LOCATIONS
        with pytest.raises(self.cursor_error_classes):
            self.image.cursor = 'not a valid option'
        self.image.cursor = 'bottom'
        assert self.image.cursor == 'bottom'

    def test_save(self, tmp_path):
        filename = tmp_path / 'woot.png'
        self.image.save(filename)
        assert filename.is_file()

    def test_save_overwrite(self, tmp_path):
        filename = tmp_path / 'woot.png'

        # First write should be fine
        self.image.save(filename)
        assert filename.is_file()

        # Second write should raise an error because file exists
        with pytest.raises(FileExistsError):
            self.image.save(filename)

        # Using overwrite should save successfully
        self.image.save(filename, overwrite=True)
