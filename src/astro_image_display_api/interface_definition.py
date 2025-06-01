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
    def load_image(self, data: Any, image_label: str | None = None) -> None:
        """
        Load data into the viewer. At a minimum, this should allow a FITS file
        to be loaded. Viewers may allow additional data types to be loaded, such as
        2D arrays or `astropy.nddata.NDData` objects.

        Parameters
        ----------
        data : Any
            The data to load. This can be a FITS file, a 2D array,
            or an `astropy.nddata.NDData` object.

        image_label : str, optional
            The label for the image.

        Notes
        -----
        Loading an image should also set the viewport for that image.
        """
        raise NotImplementedError

    # Setting and getting image properties
    @abstractmethod
    def set_cuts(self, cuts: tuple | BaseInterval, image_label: str | None = None) -> None:
        """
        Set the cuts for the image.

        Parameters
        ----------
        cuts : tuple or any Interval from `astropy.visualization`
            The cuts to set. If a tuple, it should be of the form
            ``(min, max)`` and will be interpreted as a
            `~astropy.visualization.ManualInterval`.

        image_label : str, optional
            The label of the image to set the cuts for. If not given and there is
            only one image loaded, the cuts for that image are set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cuts(self, image_label: str | None = None) -> BaseInterval:
        """
        Get the current cuts for the image.

        Parameters
        ----------
        image_label : str, optional
            The label of the image to get the cuts for. If not given and there is
            only one image loaded, the cuts for that image are returned. If there are
            multiple images and no label is provided, an error is raised.

        Returns
        -------
        cuts : `~astropy.visualization.BaseInterval`
            The Astropy interval object representing the current cuts.
        """
        raise NotImplementedError

    @abstractmethod
    def set_stretch(self, stretch: BaseStretch, image_label: str | None = None) -> None:
        """
        Set the stretch for the image.

        Parameters
        ----------
        stretch : Any stretch from `~astropy.visualization`
            The stretch to set. This can be any subclass of
            `~astropy.visualization.BaseStretch`.

        image_label : str, optional
            The label of the image to set the cuts for. If not given and there is
            only one image loaded, the cuts for that image are set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stretch(self, image_label: str | None = None) -> BaseStretch:
        """
        Get the current stretch for the image.

        Parameters
        ----------
        image_label : str, optional
            The label of the image to get the cuts for. If not given and there is
            only one image loaded, the cuts for that image are returned. If there are
            multiple images and no label is provided, an error is raised.

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
    def set_viewport(
        self, center: SkyCoord | tuple[float, float] | None = None,
        fov: Quantity | float | None = None,
        image_label: str | None = None
    ) -> None:
        """
        Set the viewport of the image, which defines the center and field of view.

        Parameters
        ----------
        center : `astropy.coordinates.SkyCoord` or tuple of float, optional
            The center of the viewport. If not given, the current center is used.
        fov : `astropy.units.Quantity` or float, optional
            The field of view (FOV) of the viewport. If not given, the current FOV
            is used. If a float is given, it is interpreted a size in pixels. For images that are
            not square, the FOV is interpreted as the size of the longer side of the image.
        image_label : str, optional
            The label of the image to set the viewport for. If not given and there is
            only one image loaded, the viewport for that image is set. If there are
            multiple images and no label is provided, an error is raised.

        Raises
        ------
        TypeError
            If the `center` is not a `SkyCoord` object or a tuple of floats, or if
            the `fov` is not a angular `Quantity` or a float, or if there is no WCS
            and the center or field of view require a WCS to be applied.

        ValueError
            If `image_label` is not provided when there are multiple images loaded.

        `astropy.units.UnitTypeError`
            If the `fov` is a `Quantity` but does not have an angular unit.
        """
        raise NotImplementedError

    @abstractmethod
    def get_viewport(self, sky_or_pixel: str | None = None, image_label: str | None = None) -> dict[str, Any]:
        """
        Get the current viewport of the image.

        Parameters
        ----------
        sky_or_pixel : str, optional
            If 'sky', the center will be returned as a `SkyCoord` object.
            If 'pixel', the center will be returned as a tuple of pixel coordinates.
            If `None`, the default behavior is to return the center as a `SkyCoord` if
            possible, or as a tuple of floats if the image is in pixel coordinates and has
            no WCS information.
        image_label : str, optional
            The label of the image to get the viewport for. If not given and there is only one
            image loaded, the viewport for that image is returned. If there are multiple images
            and no label is provided, an error is raised.

        Returns
        -------
        dict
            A dictionary containing the current viewport settings.
            The keys are 'center', 'fov', and 'image_label'.
            - 'center' is an `astropy.coordinates.SkyCoord` object or a tuple of floats.
            - 'fov' is an `astropy.units.Quantity` object or a float.
            - 'image_label' is a string representing the label of the image.

        Raises
        -------
        ValueError
            If the `sky_or_pixel` parameter is not one of 'sky', 'pixel', or `None`, or if
            the `image_label` is not provided when there are multiple images loaded.
        """
        raise NotImplementedError
