from typing import Protocol, runtime_checkable, Any
from abc import abstractmethod
import os

from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from astropy.table import Table
from astropy.units import Quantity
from astropy.visualization import BaseInterval, BaseStretch

from numpy.typing import ArrayLike

# Allowed locations for cursor display
ALLOWED_CURSOR_LOCATIONS = ('top', 'bottom', None)


__all__ = [
    'ImageViewerInterface',
]


@runtime_checkable
class ImageViewerInterface(Protocol):
    # These are attributes, not methods. The type annotations are there
    # to make sure Protocol knows they are attributes. Python does not
    # do any checking at all of these types.
    image_width: int
    image_height: int
    zoom_level: float
    cursor: str
    marker: Any

    # Allowed locations for cursor display
    ALLOWED_CURSOR_LOCATIONS: tuple = ALLOWED_CURSOR_LOCATIONS

    # The methods, grouped loosely by purpose

    # Method for loading image data
    @abstractmethod
    def load_image(self, data: Any) -> None:
        """
        Load data into the viewer. At a minimum, this should allow a FITS file
        to be loaded. Viewers may allow additional data types to be loaded, such as
        2D arrays or `astropy.nddata.NDData` objects.

        Parameters
        ----------
        data : Any
            The data to load. This can be a FITS file, a 2D array,
            or an `astropy.nddata.NDData` object.
        """
        raise NotImplementedError

    # Setting and getting image properties
    @abstractmethod
    def set_cuts(self, cuts: tuple | BaseInterval) -> None:
        """
        Set the cuts for the image.

        Parameters
        ----------
        cuts : tuple or any Interval from `astropy.visualization`
            The cuts to set. If a tuple, it should be of the form
            ``(min, max)`` and will be interpreted as a
            `~astropy.visualization.ManualInterval`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cuts(self) -> BaseInterval:
        """
        Get the current cuts for the image.

        Returns
        -------
        cuts : `~astropy.visualization.BaseInterval`
            The Astropy interval object representing the current cuts.
        """
        raise NotImplementedError

    @abstractmethod
    def set_stretch(self, stretch: BaseStretch) -> None:
        """
        Set the stretch for the image.

        Parameters
        ----------
        stretch : Any stretch from `~astropy.visualization`
            The stretch to set. This can be any subclass of
            `~astropy.visualization.BaseStretch`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stretch(self) -> BaseStretch:
        """
        Get the current stretch for the image.

        Returns
        -------
        stretch : `~astropy.visualization.BaseStretch`
            The Astropy stretch object representing the current stretch.
        """
        raise NotImplementedError

    # Saving contents of the view and accessing the view
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def load_catalog(self, table: Table, x_colname: str = 'x', y_colname: str = 'y',
                    skycoord_colname: str = 'coord', use_skycoord: bool = False,
                    catalog_label: str | None = None,
                    catalog_style: dict | None = None) -> None:
        """
        Add markers to the viewer at positions given by a catalog.

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
        catalog_label : str, optional
            The name of the marker set to use. If not given, a unique
            name will be generated.
        catalog_style: dict, optional
            A dictionary that specifies the style of the markers used to
            rerpresent the catalog. See `ImageViewerInterface.set_catalog_style`
            for details.
        """
        raise NotImplementedError

    @abstractmethod
    def set_catalog_style(
        self,
        catalog_label: str | None = None,
        shape: str = 'circle',
        color: str = 'red',
        size: float = 5.0,
        **kwargs
    ):
        """
        Set the style of the catalog markers.

        Parameters
        ----------
        shape : str, optional
            The shape of the markers. Default is ``'circle'``. The set of
            supported shapes is listed below in the `Notes` section.
        color : str, optional
            The color of the markers. Default is ``'red'``. Permitted colors are
            any CSS4 color name. CSS4 also permits hex RGB or RGBA colors.
        size : float, optional
            The size of the markers. Default is ``5.0``.

        **kwargs
            Additional keyword arguments to pass to the marker style.

        Notes
        -----
        The following shapes are supported: "circle", "square", "crosshair", "plus",
        "diamond".
        """
        raise NotImplementedError

    @abstractmethod
    def get_catalog_style(self, catalog_label: str | None = None) -> dict:
        """
        Get the style of the catalog markers.

        Returns
        -------
        dict
            The style of the markers.

        Raises
        ------

        ValueError
            If there are multiple catalog styles set and the user has not
            specified a `catalog_label` for which to get the style.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_catalog(self, catalog_label: str | list[str] | None = None) -> None:
        """
        Remove markers from the image.

        Parameters
        ----------
        catalog_label : str, optional
            The name of the marker set to remove. If the value is ``"all"``,
            then all markers will be removed.
        """
        raise NotImplementedError

    @abstractmethod
    def get_catalog(self, x_colname: str = 'x', y_colname: str = 'y',
                    skycoord_colname: str = 'coord',
                    catalog_label: str | list[str] | None = None) -> Table:
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
        catalog_label : str or list of str, optional
            The name of the marker set to use. If that value is ``"all"``,
            then all markers will be returned.

        Returns
        -------
        table : `astropy.table.Table`
            The table containing the marker positions. If no markers match the
            ``catalog_label`` parameter, an empty table is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def get_catalog_names(self) -> list[str]:
        """
        Get the names of the loaded catalogs.

        Returns
        -------
        list of str
            The names of the loaded catalogs.
        """
        raise NotImplementedError

    # Methods that modify the view
    @abstractmethod
    def center_on(self, point: tuple | SkyCoord):
        """
        Center the view on the point.

        Parameters
        ----------
        tuple or `~astropy.coordinates.SkyCoord`
            If tuple of ``(X, Y)`` is given, it is assumed
            to be in data coordinates.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def zoom(self, val: float) -> None:
        """
        Zoom in or out by the given factor.

        Parameters
        ----------
        val : float
            The zoom level to zoom the image.
            See `zoom_level`.
        """
        raise NotImplementedError
