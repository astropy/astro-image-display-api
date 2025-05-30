
import pytest

import numpy as np  # noqa: E402

from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.nddata import NDData  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
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

    # This setup is run before each test, ensuring that there are no
    # side effects of one test on another
    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Subclasses MUST define ``image_widget_class`` -- doing so as a
        class variable does the trick.
        """
        self.image = self.image_widget_class(image_width=250, image_height=100)

    def _assert_empty_marker_table(self, table):
        assert isinstance(table, Table)
        assert len(table) == 0
        assert sorted(table.colnames) == sorted(['x', 'y', 'coord', 'marker name'])

    def _get_marker_names_as_set(self):
        marks = self.image.get_markers(marker_name="all")["marker name"]
        if hasattr(marks, 'mask') and all(marks.mask):
            marker_names = set()
        else:
            marker_names = set(marks)
        return marker_names

    def test_default_marker_names(self):
        # Check only that default names are set to a non-empty string
        assert self.image.DEFAULT_MARKER_NAME

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

    def test_set_get_center_xy(self, data):
        self.image.load_image(data, image_label='test')
        self.image.set_viewport(center=(10, 10), image_label='test')  # X, Y
        vport = self.image.get_viewport(image_label='test')
        assert vport['center'] == (10, 10)
        assert vport['image_label'] == 'test'

    def test_set_get_center_world(self, data, wcs):
        self.image.load_image(NDData(data=data, wcs=wcs), image_label='test')
        self.image.set_viewport(center=SkyCoord(*wcs.crval, unit='deg'), image_label='test')

        vport = self.image.get_viewport(image_label='test')
        assert isinstance(vport['center'], SkyCoord)
        assert vport['center'].ra.deg == pytest.approx(wcs.crval[0])
        assert vport['center'].dec.deg == pytest.approx(wcs.crval[1])

    def test_set_get_fov_pixel(self, data):
        # Set data first, since that is needed to determine zoom level
        self.image.load_image(data, image_label='test')

        self.image.set_viewport(fov=100, image_label='test')
        vport = self.image.get_viewport(image_label='test')
        assert vport['fov'] == 100
        assert vport['image_label'] == 'test'

    def test_set_get_fov_world(self, data, wcs):
        # Set data first, since that is needed to determine zoom level
        self.image.load_image(NDData(data=data, wcs=wcs), image_label='test')

        # Set the FOV in world coordinates
        self.image.set_viewport(fov=0.1 * u.deg, image_label='test')
        vport = self.image.get_viewport(image_label='test')
        assert isinstance(vport['fov'], SkyCoord)
        assert vport['fov'].deg == pytest.approx(0.1)

    def test_set_get_viewport_errors(self, data, wcs):
        self.image.load_image(NDData(data=data, wcs=wcs), image_label='test')

        # fov can be float or an angular Qunatity
        with pytest.raises(u.UnitTypeError, match='[Ii]ncorrect unit for fov'):
            self.image.set_viewport(fov=100 * u.meter, image_label='test')

        # try an fov that is completely the wrong type
        with pytest.raises(TypeError, match='[Ii]nvalid value for fov'):
            self.image.set_viewport(fov='not a valid value', image_label='test')

        # center can be a SkyCoord or a tuple of floats. Try a value that is neither
        with pytest.raises(TypeError, match='[Ii]nvalid value for center'):
            self.image.set_viewport(center='not a valid value', image_label='test')

        # Check that an error is raised if a label is provided that does not
        # match an image that is loaded.
        with pytest.raises(ValueError, match='[Ii]mage label.*not found'):
            self.image.set_viewport(center=(10, 10), fov=100, image_label='not a valid label')

        # Getting a viewport for an image_label that does not exist should raise an error
        with pytest.raises(ValueError, match='[Ii]mage label.*not found'):
            self.image.get_viewport(image_label='not a valid label')

        # If there are multiple images loaded, the image_label must be provided
        self.image.load_image(data, image_label='another test')

        with pytest.raises(ValueError, match='[Ii]mage label.*not provided'):
            self.image.get_viewport()

    def test_viewport_is_defined_aster_loading_image(self, data):
        # Check that the viewport is set to a default value when an image
        # is loaded, even if no viewport is explicitly set.
        self.image.load_image(data)

        # Getting the viewport should not fail...
        vport = self.image.get_viewport()

        assert 'center' in vport
        # No world, so center should be a tuple
        assert isinstance(vport['center'], tuple)
        assert 'fov' in vport
        # fov should be a float since no WCS
        assert isinstance(vport['fov'], float)
        assert 'image_label' in vport
        assert vport['image_label'] is None

    def test_set_get_view_port_no_image_label(self, data):
        # If there is only one image, the viewport should be able to be set
        # and retrieved without an image label.

        # Add an image without an image label
        self.image.load_image(data)

        # Set the viewport without an image label
        self.image.set_viewport(center=(10, 10), fov=100)

        # Getting the viewport again should return the same values
        vport = self.image.get_viewport()
        assert vport['center'] == (10, 10)
        assert vport['fov'] == 100
        assert vport['image_label'] is None

    def test_set_get_viewport_single_label(self, data):
        # If there is only one image, the viewport should be able to be set
        # and retrieved with an image label.

        # Add an image with an image label
        self.image.load_image(data, image_label='test')

        # Getting the viewport should not fail...
        vport = self.image.get_viewport(image_label='test')
        assert 'center' in vport
        assert 'fov' in vport
        assert 'image_label' in vport
        assert vport['image_label'] == 'test'

        # Set the viewport with an image label
        self.image.set_viewport(center=(10, 10), fov=100, image_label='test')

        # Getting the viewport again should return the same values
        vport = self.image.get_viewport(image_label='test')
        assert vport['center'] == (10, 10)
        assert vport['fov'] == 100
        assert vport['image_label'] == 'test'

    def test_get_viewport_sky_or_pixel(self, data, wcs):
        # Check that the viewport can be retrieved in both pixel and world
        # coordinates, depending on the WCS of the image.

        # Load the data with a WCS
        self.image.load_image(NDData(data=data, wcs=wcs), image_label='test')

        input_center = SkyCoord(*wcs.val, unit='deg')
        input_fov = 2 * u.arcmin
        self.image.set_viewport(center=input_center, fov=input_fov, image_label='test')

        # Get the viewport in pixel coordinates
        vport_pixel = self.image.get_viewport(image_label='test', sky_or_pixel='pixel')
        assert vport_pixel['center'] == wcs.crpix
        # tbh, not at all sure what the fov should be in pixel coordinates,
        # so just check that it is a float.
        assert isinstance(vport_pixel['fov'], float)

        # Get the viewport in world coordinates
        vport_world = self.image.get_viewport(image_label='test', sky_or_pixel='sky')
        assert vport_world['center'] == input_center
        assert vport_world['fov'] == input_fov

    @pytest.mark.parametrize("sky_or_pixel", ['sky', 'pixel'])
    def test_get_viewport_no_sky_or_pixel(self, data, wcs, sky_or_pixel):
        # Check that get_viewport returns the correct "default" sky_or_pixel
        # value when the result ought to be unambiguous.
        if sky_or_pixel == 'sky':
            use_wcs = wcs
        else:
            use_wcs = None

        self.image.load_image(NDData(data=data, wcs=use_wcs), image_label='test')

        vport = self.image.get_viewport(image_label='test')
        match sky_or_pixel:
            case 'sky':
                assert isinstance(vport['center'], SkyCoord)
                assert vport['fov'].unit.physical_type == "angle"
            case 'pixel':
                assert isinstance(vport['center'], tuple)
                assert isinstance(vport['fov'], float)

    def test_viewport_round_trips(self, data, wcs):
        # Check that the viewport retrieved with get can be used to set
        # the viewport again, and that the values are the same.
        self.image.load_image(NDData(data=data, wcs=wcs), image_label='test')
        self.image.set_viewport(center=(10, 10), fov=100, image_label='test')
        vport = self.image.get_viewport(image_label='test')
        # Set the viewport again using the values from the get_viewport
        self.image.set_viewport(**vport)
        # Get the viewport again and check that the values are the same
        vport2 = self.image.get_viewport(image_label='test')
        assert vport2 == vport

    def test_marker_properties(self):
        # Set the marker style
        marker_style = {'color': 'yellow', 'radius': 10, 'type': 'cross'}
        self.image.marker = marker_style
        m_str = str(self.image.marker)
        for key in marker_style.keys():
            assert key in m_str

    def test_add_markers(self):
        original_marker_name = self.image.DEFAULT_MARKER_NAME
        data = np.arange(10).reshape(5, 2)
        orig_tab = Table(data=data, names=['x', 'y'], dtype=('float', 'float'))
        tab = Table(data=data, names=['x', 'y'], dtype=('float', 'float'))
        self.image.add_markers(tab, x_colname='x', y_colname='y',
                               skycoord_colname='coord', marker_name='test1')

        # Make sure setting didn't change the default name
        assert self.image.DEFAULT_MARKER_NAME == original_marker_name

        # Regression test for GitHub Issue 45:
        # Adding markers should not modify the input data table.
        assert (tab == orig_tab).all()

        # Add more markers under different name.
        self.image.add_markers(tab, x_colname='x', y_colname='y',
                               skycoord_colname='coord', marker_name='test2')

        marknames = self._get_marker_names_as_set()
        assert marknames == set(['test1', 'test2'])
        # assert self.image.get_marker_names() == ['test1', 'test2']

        # No guarantee markers will come back in the same order, so sort them.
        t1 = self.image.get_markers(marker_name='test1')
        # Sort before comparing
        t1.sort('x')
        tab.sort('x')
        assert np.all(t1['x'] == tab['x'])
        assert (t1['y'] == tab['y']).all()

        # That should have given us two copies of the input table
        t2 = self.image.get_markers(marker_name="all")
        expected = vstack([tab, tab], join_type='exact')
        # Sort before comparing
        t2.sort(['x', 'y'])
        expected.sort(['x', 'y'])
        assert (t2['x'] == expected['x']).all()
        assert (t2['y'] == expected['y']).all()

        self.image.remove_markers(marker_name='test1')
        marknames = self._get_marker_names_as_set()
        assert marknames == set(['test2'])
        # assert self.image.get_marker_names() == ['test2']

        # Ensure unable to mark with reserved name
        for name in self.image.RESERVED_MARKER_SET_NAMES:
            with pytest.raises(ValueError, match='not allowed'):
                self.image.add_markers(tab, marker_name=name)

        # Add markers with no marker name and check we can retrieve them
        # using the default marker name
        self.image.add_markers(tab, x_colname='x', y_colname='y',
                               skycoord_colname='coord')
        # Don't care about the order of the marker names so use set instead of
        # list.
        marknames = self._get_marker_names_as_set()
        assert (set(marknames) == set(['test2', self.image.DEFAULT_MARKER_NAME]))

        # Clear markers to not pollute other tests.
        self.image.reset_markers()
        marknames = self._get_marker_names_as_set()
        assert len(marknames) == 0
        self._assert_empty_marker_table(self.image.get_markers(marker_name="all"))
        # Check that no markers remain after clearing
        tab = self.image.get_markers(marker_name=self.image.DEFAULT_MARKER_NAME)
        self._assert_empty_marker_table(tab)

        # Check that retrieving a marker set that doesn't exist returns
        # an empty table with the right columns
        tab = self.image.get_markers(marker_name='test1')
        self._assert_empty_marker_table(tab)

    def test_get_markers_accepts_list_of_names(self):
        # Check that the get_markers method accepts a list of marker names
        # and returns a table with all the markers from all the named sets.
        data = np.arange(10).reshape((5, 2))
        tab = Table(data=data, names=['x', 'y'])
        self.image.add_markers(tab, marker_name='test1')
        self.image.add_markers(tab, marker_name='test2')

        # No guarantee markers will come back in the same order, so sort them.
        t1 = self.image.get_markers(marker_name=['test1', 'test2'])
        # Sort before comparing
        t1.sort('x')
        expected = vstack([tab, tab], join_type='exact')
        expected.sort('x')
        np.testing.assert_array_equal(t1['x'], expected['x'])
        np.testing.assert_array_equal(t1['y'], expected['y'])

    def test_remove_markers(self):
        with pytest.raises(ValueError, match='arf'):
            self.image.remove_markers(marker_name='arf')

    def test_remove_markers_name_all(self):
        data = np.arange(10).reshape(5, 2)
        tab = Table(data=data, names=['x', 'y'])
        self.image.add_markers(tab, marker_name='test1')
        self.image.add_markers(tab, marker_name='test2')

        self.image.remove_markers(marker_name='all')
        self._assert_empty_marker_table(self.image.get_markers(marker_name='all'))

    def test_remove_marker_accepts_list(self):
        data = np.arange(10).reshape(5, 2)
        tab = Table(data=data, names=['x', 'y'])
        self.image.add_markers(tab, marker_name='test1')
        self.image.add_markers(tab, marker_name='test2')

        self.image.remove_markers(marker_name=['test1', 'test2'])
        marks = self.image.get_markers(marker_name='all')
        self._assert_empty_marker_table(marks)

    def test_adding_markers_as_world(self, data, wcs):
        ndd = NDData(data=data, wcs=wcs)
        self.image.load_image(ndd)

        # Add markers using world coordinates
        pixels = np.linspace(0, 100, num=10).reshape(5, 2)
        marks_pix = Table(data=pixels, names=['x', 'y'], dtype=('float', 'float'))
        marks_coords = wcs.pixel_to_world(marks_pix['x'], marks_pix['y'])
        mark_coord_table = Table(data=[marks_coords], names=['coord'])
        self.image.add_markers(mark_coord_table, use_skycoord=True)
        result = self.image.get_markers()
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
