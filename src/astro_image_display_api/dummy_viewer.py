import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData, NDData
from astropy.table import Table, vstack
from astropy.units import Quantity, get_physical_type
from astropy.wcs import WCS
from astropy.visualization import AsymmetricPercentileInterval, BaseInterval, BaseStretch, LinearStretch, ManualInterval
from numpy.typing import ArrayLike

from .interface_definition import ImageViewerInterface

@dataclass
class CatalogInfo:
    """
    A named tuple to hold information about a catalog.
    """
    style: dict[str, Any] = field(default_factory=dict)
    data: Table | None = None

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
    _cursor: str = ImageViewerInterface.ALLOWED_CURSOR_LOCATIONS[0]
    marker: Any = "marker"
    _cuts: BaseInterval | tuple[float, float] = AsymmetricPercentileInterval(upper_percentile=95)
    _stretch: BaseStretch = LinearStretch
    # viewer: Any

    # Allowed locations for cursor display
    ALLOWED_CURSOR_LOCATIONS: tuple = ImageViewerInterface.ALLOWED_CURSOR_LOCATIONS

    # some internal variable for keeping track of viewer state
    _wcs: WCS | None = None
    _center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        # This is a dictionary of marker sets. The keys are the names of the
        # marker sets, and the values are the tables containing the markers.
        self._default_marker_style = dict(shape="circle", color="yellow", size=10)
        self._catalogs = defaultdict(CatalogInfo)
        self._catalogs[None].data = None
        self._catalogs[None].style = self._default_marker_style.copy()

    def _user_catalog_labels(self) -> list[str]:
        """
        Get the user-defined catalog labels.
        """
        return [label for label in self._catalogs if label is not None]

    def _resolve_catalog_label(self, catalog_label: str | None) -> str:
        """
        Figure out the catalog label if the user did not specify one. This
        is needed so that the user gets what they expect in the simple case
        where there is only one catalog loaded. In that case the user may
        or may not have actually specified a catalog label.
        """
        user_keys = self._user_catalog_labels()
        if catalog_label is None:
            match len(user_keys):
                case 0:
                    # No user-defined catalog labels, so return the default label.
                    catalog_label = None
                case 1:
                    # The user must have loaded a catalog, so return that instead of
                    # the default label, which live in the key None.
                    catalog_label = user_keys[0]
                case _:
                    raise ValueError(
                        "Multiple catalog styles defined. Please specify a catalog_label to get the style."
                    )

        return catalog_label

    def get_stretch(self) -> BaseStretch:
        return self._stretch

    def set_stretch(self, value: BaseStretch) -> None:
        if not isinstance(value, BaseStretch):
            raise ValueError(f"Stretch option {value} is not valid. Must be an Astropy.visualization Stretch object.")
        self._stretch = value

    def get_cuts(self) -> tuple:
        return self._cuts

    def set_cuts(self, value: tuple[float, float] | BaseInterval) -> None:
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

    def get_catalog_style(self, catalog_label=None) -> dict[str, dict[str, Any]]:
        """
        Get the style for the catalog.

        Parameters
        ----------
        catalog_label : str, optional
            The label of the catalog. Default is ``None``.

        Returns
        -------
        dict
            The style for the catalog.
        """
        catalog_label = self._resolve_catalog_label(catalog_label)

        style = self._catalogs[catalog_label].style
        style["catalog_label"] = catalog_label
        return style

    def set_catalog_style(
            self,
            catalog_label: str | None = None,
            shape: str = "",
            color: str = "",
            size: float = 0,
            **kwargs
    ) -> None:
        """
        Set the style for the catalog.

        Parameters
        ----------
        catalog_label : str, optional
            The label of the catalog.
        shape : str, optional
            The shape of the markers.
        color : str, optional
            The color of the markers.
        size : float, optional
            The size of the markers.
        **kwargs
            Additional keyword arguments to pass to the marker style.
        """
        shape = shape if shape else self._default_marker_style["shape"]
        color = color if color else self._default_marker_style["color"]
        size = size if size else self._default_marker_style["size"]

        catalog_label = self._resolve_catalog_label(catalog_label)

        if self._catalogs[catalog_label].data is None:
            raise ValueError("Must load a catalog before setting a catalog style.")

        self._catalogs[catalog_label].style = {
            "shape": shape,
            "color": color,
            "size": size,
            **kwargs,
        }

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
    def load_catalog(self, table: Table, x_colname: str = 'x', y_colname: str = 'y',
                    skycoord_colname: str = 'coord', use_skycoord: bool = True,
                    catalog_label: str | None = None,
                    catalog_style: dict | None = None) -> None:
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
            get the marker positions.
        catalog_label : str, optional
            The name of the marker set to use. If not given, a unique
            name will be generated.
        """
        try:
            coords = table[skycoord_colname]
        except KeyError:
            coords = None

        try:
            xy = (table[x_colname], table[y_colname])
        except KeyError:
            xy = None

        to_add = deepcopy(table)
        if xy is None:
            if self._wcs is not None and coords is not None:
                x, y = self._wcs.world_to_pixel(coords)
                to_add[x_colname] = x
                to_add[y_colname] = y
            else:
                to_add[x_colname] = to_add[y_colname] = None

        if coords is None:
            if use_skycoord and self._wcs is None:
                raise ValueError("Cannot use sky coordinates without a SkyCoord column or WCS.")
            elif xy is not None and self._wcs is not None:
                # If we have xy coordinates, convert them to sky coordinates
                coords = self._wcs.pixel_to_world(xy[0], xy[1])
                to_add[skycoord_colname] = coords
            else:
                to_add[skycoord_colname] = None

        catalog_label = self._resolve_catalog_label(catalog_label)
        if (
            catalog_label in self._catalogs
            and self._catalogs[catalog_label].data is not None
        ):
            old_table = self._catalogs[catalog_label].data
            self._catalogs[catalog_label].data = vstack([old_table, to_add])
        else:
            self._catalogs[catalog_label].data = to_add

    def remove_catalog(self, catalog_label: str | None = None) -> None:
        """
        Remove markers from the image.

        Parameters
        ----------
        marker_name : str, optional
            The name of the marker set to remove. If the value is ``"*"``,
            then all markers will be removed.
        """
        if isinstance(catalog_label, list):
            raise ValueError(
                "Cannot remove multiple catalogs from a list. Please specify "
                "a single catalog label or use '*' to remove all catalogs."
            )
        elif catalog_label == "*":
            # If the user wants to remove all catalogs, we reset the
            # catalogs dictionary to an empty one.
            self._catalogs = defaultdict(CatalogInfo)
            return

        # Special cases are done, so we can resolve the catalog label
        catalog_label = self._resolve_catalog_label(catalog_label)

        try:
            del self._catalogs[catalog_label]
        except KeyError:
            raise ValueError(f"Marker name {catalog_label} not found.")

    def get_catalog(self, x_colname: str = 'x', y_colname: str = 'y',
                    skycoord_colname: str = 'coord',
                    catalog_label: str | None = None) -> Table:
        # Dostring is copied from the interface definition, so it is not
        # duplicated here.
        catalog_label = self._resolve_catalog_label(catalog_label)

        result = self._catalogs[catalog_label].data if catalog_label in self._catalogs else Table(names=["x", "y", "coord"])

        result.rename_columns(["x", "y", "coord"], [x_colname, y_colname, skycoord_colname])

        return result
    get_catalog.__doc__ = ImageViewerInterface.get_catalog.__doc__

    def get_catalog_names(self) -> list[str]:
        return list(self._user_catalog_labels())
    get_catalog_names.__doc__ = ImageViewerInterface.get_catalog_names.__doc__

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
