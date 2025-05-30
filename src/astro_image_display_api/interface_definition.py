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

DEFAULT_MARKER_NAME = 'default-marker-name'

# List of marker names that are for internal use only
RESERVED_MARKER_SET_NAMES = ('all',)

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

    # List of marker names that are for internal use only
    RESERVED_MARKER_SET_NAMES: tuple = RESERVED_MARKER_SET_NAMES

    # Default marker name for marking via API
    DEFAULT_MARKER_NAME: str = DEFAULT_MARKER_NAME

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
        raise NotImplementedError

    # @abstractmethod
    # def remove_all_markers(self):
    #     raise NotImplementedError

    @abstractmethod
    def reset_markers(self) -> None:
        """
        Remove all markers from the image.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_markers(self, marker_name: str | list[str] | None = None) -> None:
        """
        Remove markers from the image.

        Parameters
        ----------
        marker_name : str, optional
            The name of the marker set to remove. If the value is ``"all"``,
            then all markers will be removed.
        """
        raise NotImplementedError

    @abstractmethod
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
