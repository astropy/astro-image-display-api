import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData, NDData
from astropy.table import Table, vstack
from astropy.units import Quantity, get_physical_type
from astropy.wcs import WCS
from astropy.visualization import BaseInterval, BaseStretch, ManualInterval
from numpy.typing import ArrayLike

from .interface_definition import ImageViewerInterface


@dataclass
class ImageViewer:
    """
    This viewer does not do anything except making changes to its internal
    state to simulate the behavior of a real viewer.
    """
    # These are attributes, not methods. The type annotations are there
    # to make sure Protocol knows they are attributes. Python does not
    # do any checking at all of these types.
    image_width: int = 0
    image_height: int = 0
    zoom_level: float = 1
    stretch_options: tuple = ("linear", "log", "sqrt")
    autocut_options: tuple = ("minmax", "zscale", "asinh", "percentile", "histogram")
    _cursor: str = ImageViewerInterface.ALLOWED_CURSOR_LOCATIONS[0]
    marker: Any = "marker"
    _cuts: str | tuple[float, float] = (0, 1)
    _stretch: str = "linear"
    # viewer: Any

    # Allowed locations for cursor display
    ALLOWED_CURSOR_LOCATIONS: tuple = ImageViewerInterface.ALLOWED_CURSOR_LOCATIONS

    # List of marker names that are for internal use only
    RESERVED_MARKER_SET_NAMES: tuple = ImageViewerInterface.RESERVED_MARKER_SET_NAMES

    # Default marker name for marking via API
    DEFAULT_MARKER_NAME: str = ImageViewerInterface.DEFAULT_MARKER_NAME

    # some internal variable for keeping track of viewer state
    _interactive_marker_name: str = ""
    _previous_marker: Any = ""
    _markers: dict[str, Table] = field(default_factory=dict)
    _wcs: WCS | None = None
    _center: tuple[float, float] = (0.0, 0.0)

    @property
    def stretch(self) -> BaseStretch:
        return self._stretch

    @stretch.setter
    def stretch(self, value: BaseStretch) -> None:
        if not isinstance(value, BaseStretch):
            raise ValueError(f"Stretch option {value} is not valid. Must be one of {self.stretch_options}.")
        self._stretch = value

    @property
    def cuts(self) -> tuple:
        return self._cuts

    @cuts.setter
    def cuts(self, value: tuple) -> None:
        if isinstance(value, tuple) and len(value) == 2:
            self._cuts = ManualInterval(value[0], value[1])
        elif isinstance(value, BaseInterval):
            self._cuts = value
        else:
            raise ValueError("Cuts must be an Astropy.visualization Interval object or a tuple of two values.")


    @property
    def cursor(self) -> str:
        return self._cursor

    @cursor.setter
    def cursor(self, value: str) -> None:
        if value not in self.ALLOWED_CURSOR_LOCATIONS:
            raise ValueError(f"Cursor location {value} is not valid. Must be one of {self.ALLOWED_CURSOR_LOCATIONS}.")
        self._cursor = value

    # The methods, grouped loosely by purpose

    # Methods for loading data
    def load_image(self, file: str | os.PathLike | ArrayLike | NDData) -> None:
        """
        Load a FITS file into the viewer.

        Parameters
        ----------
        file : str or `astropy.io.fits.HDU`
            The FITS file to load. If a string, it can be a URL or a
            file path.
        """
        if isinstance(file, (str, os.PathLike)):
            if isinstance(file, str):
                is_adsf = file.endswith(".asdf")
            else:
                is_asdf = file.suffix == ".asdf"
            if is_asdf:
                self._load_asdf(file)
            else:
                self._load_fits(file)
        elif isinstance(file, NDData):
            self._load_nddata(file)
        else:
            # Assume it is a 2D array
            self._load_array(file)

    def _load_fits(self, file: str | os.PathLike) -> None:
        ccd = CCDData.read(file)
        self._wcs = ccd.wcs
        self.image_height, self.image_width = ccd.shape
        # Totally made up number...as currently defined, zoom_level means, esentially, ratio
        # of image size to viewer size.
        self.zoom_level = 1.0
        self.center_on((self.image_width / 2, self.image_height / 2))

    def _load_array(self, array: ArrayLike) -> None:
        """
        Load a 2D array into the viewer.

        Parameters
        ----------
        array : array-like
            The array to load.
        """
        self.image_height, self.image_width = array.shape
        # Totally made up number...as currently defined, zoom_level means, esentially, ratio
        # of image size to viewer size.
        self.zoom_level = 1.0
        self.center_on((self.image_width / 2, self.image_height / 2))


    def _load_nddata(self, data: NDData) -> None:
        """
        Load an `astropy.nddata.NDData` object into the viewer.

        Parameters
        ----------
        data : `astropy.nddata.NDData`
            The NDData object to load.
        """
        self._wcs = data.wcs
        # Not all NDDData objects have a shape, apparently
        self.image_height, self.image_width = data.data.shape
        # Totally made up number...as currently defined, zoom_level means, esentially, ratio
        # of image size to viewer size.
        self.zoom_level = 1.0
        self.center_on((self.image_width / 2, self.image_height / 2))

    def _load_asdf(self, asdf_file: str | os.PathLike) -> None:
        """
        Not implementing some load types is fine.
        """
        raise NotImplementedError("ASDF loading is not implemented in this dummy viewer.")

    # Saving contents of the view and accessing the view
    def save(self, filename: str | os.PathLike, overwrite: bool = False) -> None:
        """
        Save the current view to a file.

        Parameters
        ----------
        filename : str or `os.PathLike`
            The file to save to. The format is determined by the
            extension.

        overwrite : bool, optional
            If `True`, overwrite the file if it exists. Default is
            `False`.
        """
        p = Path(filename)
        if p.exists() and not overwrite:
            raise FileExistsError(f"File {filename} already exists. Use overwrite=True to overwrite it.")

        p.write_text("This is a dummy file. The viewer does not save anything.")

    # Marker-related methods
    def add_markers(self, table: Table, x_colname: str = 'x', y_colname: str = 'y',
                    skycoord_colname: str = 'coord', use_skycoord: bool = False,
                    marker_name: str | None = None) -> None:
        """
        Add markers to the image.

        Parameters
        ----------
        table : `astropy.table.Table`
            The table containing the marker positions.
        x_colname : str, optional
            The name of the column containing the x positions. Default
            is ``'x'``.
        y_colname : str, optional
            The name of the column containing the y positions. Default
            is ``'y'``.
        skycoord_colname : str, optional
            The name of the column containing the sky coordinates. If
            given, the ``use_skycoord`` parameter is ignored. Default
            is ``'coord'``.
        use_skycoord : bool, optional
            If `True`, the ``skycoord_colname`` column will be used to
            get the marker positions. Default is `False`.
        marker_name : str, optional
            The name of the marker set to use. If not given, a unique
            name will be generated.
        """
        try:
            coords = table[skycoord_colname]
        except KeyError:
            coords = None

        if use_skycoord:
            if self._wcs is not None:
                x, y = self._wcs.world_to_pixel(coords)
            else:
                raise ValueError("WCS is not set. Cannot convert to pixel coordinates.")
        else:
            x = table[x_colname]
            y = table[y_colname]

            if not coords and self._wcs is not None:
                coords = self._wcs.pixel_to_world(x, y)

        if marker_name in self.RESERVED_MARKER_SET_NAMES:
            raise ValueError(f"Marker name {marker_name} not allowed.")

        marker_name = marker_name if marker_name else self.DEFAULT_MARKER_NAME

        to_add = Table(
            dict(
                x=x,
                y=y,
                coord=coords if coords else [None] * len(x),
            )
        )
        to_add["marker name"] = marker_name

        if marker_name in self._markers:
            marker_table = self._markers[marker_name]
            self._markers[marker_name] = vstack([marker_table, to_add])
        else:
            self._markers[marker_name] = to_add

    def reset_markers(self) -> None:
        """
        Remove all markers from the image.
        """
        self._markers = {}

    def remove_markers(self, marker_name: str | list[str] | None = None) -> None:
        """
        Remove markers from the image.

        Parameters
        ----------
        marker_name : str, optional
            The name of the marker set to remove. If the value is ``"all"``,
            then all markers will be removed.
        """
        if isinstance(marker_name, str):
            if marker_name in self._markers:
                del self._markers[marker_name]
            elif marker_name == "all":
                self._markers = {}
            else:
                raise ValueError(f"Marker name {marker_name} not found.")
        elif isinstance(marker_name, list):
            for name in marker_name:
                if name in self._markers:
                    del self._markers[name]
                else:
                    raise ValueError(f"Marker name {name} not found.")

    def get_markers(self, x_colname: str = 'x', y_colname: str = 'y',
                    skycoord_colname: str = 'coord',
                    marker_name: str | list[str] | None = None) -> Table:
        """
        Get the marker positions.

        Parameters
        ----------
        x_colname : str, optional
            The name of the column containing the x positions. Default
            is ``'x'``.
        y_colname : str, optional
            The name of the column containing the y positions. Default
            is ``'y'``.
        skycoord_colname : str, optional
            The name of the column containing the sky coordinates. Default
            is ``'coord'``.
        marker_name : str or list of str, optional
            The name of the marker set to use. If that value is ``"all"``,
            then all markers will be returned.

        Returns
        -------
        table : `astropy.table.Table`
            The table containing the marker positions. If no markers match the
            ``marker_name`` parameter, an empty table is returned.
        """
        if isinstance(marker_name, str):
            if marker_name == "all":
                marker_name = self._markers.keys()
            else:
                marker_name = [marker_name]
        elif marker_name is None:
            marker_name = [self.DEFAULT_MARKER_NAME]

        to_stack = [self._markers[name] for name in marker_name if name in self._markers]

        result = vstack(to_stack) if to_stack else Table(names=["x", "y", "coord", "marker name"])
        result.rename_columns(["x", "y", "coord"], [x_colname, y_colname, skycoord_colname])

        return result


    # Methods that modify the view
    def center_on(self, point: tuple | SkyCoord):
        """
        Center the view on the point.

        Parameters
        ----------
        tuple or `~astropy.coordinates.SkyCoord`
            If tuple of ``(X, Y)`` is given, it is assumed
            to be in data coordinates.
        """
        # currently there is no way to get the position of the center, but we may as well make
        # note of it
        if isinstance(point, SkyCoord):
            if self._wcs is not None:
                point = self._wcs.world_to_pixel(point)
            else:
                raise ValueError("WCS is not set. Cannot convert to pixel coordinates.")

        self._center = point

    def offset_by(self, dx: float | Quantity, dy: float | Quantity) -> None:
        """
        Move the center to a point that is given offset
        away from the current center.

        Parameters
        ----------
        dx, dy : float or `~astropy.units.Quantity`
            Offset value. Without a unit, assumed to be pixel offsets.
            If a unit is attached, offset by pixel or sky is assumed from
            the unit.
        """
        # Convert to quantity to make the rest of the processing uniform
        dx = Quantity(dx)
        dy = Quantity(dy)

        # This raises a UnitConversionError if the units are not compatible
        dx.to(dy.unit)

        # Do we have an angle or pixel offset?
        if get_physical_type(dx) == "angle":
            # This is a sky offset
            if self._wcs is not None:
                old_center_coord = self._wcs.pixel_to_world(self._center[0], self._center[1])
                new_center = old_center_coord.spherical_offsets_by(dx, dy)
                self.center_on(new_center)
            else:
                raise ValueError("WCS is not set. Cannot convert to pixel coordinates.")
        else:
            # This is a pixel offset
            new_center = (self._center[0] + dx.value, self._center[1] + dy.value)
            self.center_on(new_center)

    def zoom(self, val) -> None:
        """
        Zoom in or out by the given factor.

        Parameters
        ----------
        val : int
            The zoom level to zoom the image.
            See `zoom_level`.
        """
        self.zoom_level *= val
