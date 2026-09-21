import numbers
import os
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity
from astropy.visualization import BaseInterval, BaseStretch

__all__ = [
    "ImageViewerInterface",
]


@runtime_checkable
class ImageViewerInterface(Protocol):
    # The methods, grouped loosely by purpose

    # Method for loading image data
    @abstractmethod
    def load_image(self, data: Any, image_label: str | None = None, **kwargs) -> None:
        """
        Load data into the viewer. At a minimum, this should allow a FITS file
        to be loaded. Viewers may allow additional data types to be loaded, such as
        2D arrays or `~astropy.nddata.NDData` objects.

        Parameters
        ----------
        data :
            The data to load. This can be a FITS file, a 2D array,
            or an `~astropy.nddata.NDData` object.

        image_label : optional
            The label for the image. If not given, a single shared default
            label is used, so loading an image without a label repeatedly
            replaces the previously loaded unlabeled image. Loading an
            image using an ``image_label`` that already exists will
            replace the existing image with the new one.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Notes
        -----

        Loading an image should also set an appropriate viewport for that image.
        """
        raise NotImplementedError

    # Setting and getting image properties
    @abstractmethod
    def get_image(
        self,
        image_label: str | None = None,
        **kwargs,
    ) -> Any:
        """
        Get the data of a loaded image.

        Parameters
        ----------
        image_label : optional
            The label of the image to get. If not given and there is
            only one image loaded, that image is returned.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Returns
        -------
        image_data : Any
            The data that was loaded for the image with the given label.
            The returned object must either be array-like or expose the
            array of image values through a ``data`` attribute, as
            `~astropy.nddata.NDData` and its subclasses do. It is expected
            that the returned data can be re-loaded into the viewer using
            `load_image`.

        Raises
        ------
        ValueError
            If the ``image_label`` is not provided when there are multiple images
            loaded, or if the ``image_label`` does not correspond to a loaded image.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def image_labels(
        self,
    ) -> tuple[str, ...]:
        """
        Labels of the loaded images.

        Returns
        -------
        image_labels: tuple of str
            The labels of the loaded images.
        """
        raise NotImplementedError

    @abstractmethod
    def set_cuts(
        self,
        cuts: tuple[numbers.Real, numbers.Real] | BaseInterval,
        image_label: str | None = None,
        **kwargs,
    ) -> None:
        """
        Set the cuts for the image.

        Parameters
        ----------
        cuts: any interval from `astropy.visualization`
            The cuts to set. If a tuple, it should be of the form
            ``(min, max)`` and will be interpreted as a
            `~astropy.visualization.ManualInterval`.

        image_label : optional
            The label of the image to set the cuts for. If not given and there is
            only one image loaded, the cuts for that image are set.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Raises
        ------
        TypeError
            If the ``cuts`` parameter is not a tuple or an
            `astropy.visualization.BaseInterval` object.

        ValueError
            If the ``image_label`` is not provided when there are multiple images
            loaded, or if the ``image_label`` does not correspond to a loaded image.

        Notes
        -----
        Setting cuts should update the display of the image to reflect the new
        cuts.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cuts(self, image_label: str | None = None, **kwargs) -> BaseInterval:
        """
        Get the current cuts for the image.

        Parameters
        ----------
        image_label : optional
            The label of the image to get the cuts for. If not given and there is
            only one image loaded, the cuts for that image are returned. If there are
            multiple images and no label is provided, an error is raised.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Returns
        -------
        cuts : `astropy.visualization.BaseInterval`
            The Astropy interval object representing the current cuts.

        Raises
        ------
        ValueError
            If the ``image_label`` is not provided when there are multiple images
            loaded, or if the ``image_label`` does not correspond to a loaded image.

        Notes
        -----
        This has no effect on the displayed image.
        """
        raise NotImplementedError

    @abstractmethod
    def set_stretch(
        self, stretch: BaseStretch, image_label: str | None = None, **kwargs
    ) -> None:
        """
        Set the stretch for the image.

        Parameters
        ----------
        stretch : Any stretch from `~astropy.visualization`
            The stretch to set. This can be any subclass of
            `~astropy.visualization.BaseStretch`.

        image_label :
            The label of the image to set the stretch for. If not given and there is
            only one image loaded, the stretch for that image are set.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Raises
        ------
        TypeError
            If the ``stretch`` is not a valid `~astropy.visualization.BaseStretch`
            object.

        ValueError
            If the ``image_label`` is not provided when there are multiple images loaded
            or if the ``image_label`` does not correspond to a loaded image.

        Notes
        -----
        Setting the stretch should update the display of the image to reflect the new
        stretch.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stretch(self, image_label: str | None = None, **kwargs) -> BaseStretch:
        """
        Get the current stretch for the image.

        Parameters
        ----------
        image_label : str, optional
            The label of the image to get the stretch for. If not given and there
            is only one image loaded, the stretch for that image is returned. If
            there are multiple images and no label is provided, an error is raised.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Returns
        -------
        stretch : `~astropy.visualization.BaseStretch`
            The Astropy stretch object representing the current stretch.

        Raises
        ------
        ValueError
            If the ``image_label`` is not provided when there are multiple images
            loaded, or if the ``image_label`` does not correspond to a loaded image.

        Notes
        -----
        This has no effect on the displayed image.
        """
        raise NotImplementedError

    @abstractmethod
    def set_colormap(
        self, map_name: str, image_label: str | None = None, **kwargs
    ) -> None:
        """
        Set the colormap for the image specified by image_label.

        Parameters
        ----------
        map_name
            The name of the colormap to set. This should be a
            valid colormap name from `Matplotlib <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_.
            Not all backends support all colormaps; a backend must raise
            a `ValueError` for a colormap name it does not support.
        image_label : optional
            The label of the image to set the colormap for. If not given and there is
            only one image loaded, the colormap for that image is set. If there are
            multiple images and no label is provided, an error is raised.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Raises
        ------
        ValueError
            If the ``map_name`` is not a colormap name the backend supports,
            if the ``image_label`` is not provided when there are multiple
            images loaded, or if the ``image_label`` does not correspond to a
            loaded image.

        Notes
        -----
        This should update the display of the image to reflect the new colormap.

        .. _Matplotlib: https://matplotlib.org/stable/gallery/color/colormap_reference.html
        """
        raise NotImplementedError

    @abstractmethod
    def get_colormap(self, image_label: str | None = None, **kwargs) -> str | None:
        """
        Get the current colormap for the image.

        Parameters
        ----------
        image_label : str, optional
            The label of the image to get the colormap for. If not given and there is
            only one image loaded, the colormap for that image is returned. If there are
            multiple images and no label is provided, an error is raised.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Returns
        -------
        map_name : str or None
            The name of the current colormap, or `None` if no colormap has
            been set for the image. A return of `None` means the backend is
            using whatever default colormap it chooses; a backend may instead
            return the name of that default.

        Raises
        ------
        ValueError
            If the ``image_label`` is not provided when there are multiple images loaded
            or if the ``image_label`` does not correspond to a loaded image.

        Notes
        -----
        This has no effect on the displayed image.
        """
        raise NotImplementedError

    # Saving contents of the view and accessing the view
    @abstractmethod
    def save(
        self, filename: str | os.PathLike, overwrite: bool = False, **kwargs
    ) -> None:
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

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Raises
        ------
        FileExistsError
            If the file already exists and ``overwrite`` is `False`.

        Notes
        -----
        This has no effect on the displayed image.

        Saving is only specified for a view that has an image in it: the
        compliance suite `~astro_image_display_api.api_test.ImageAPITest`
        loads an image before calling this method. What happens when
        nothing has been loaded is left to the implementation, which may
        raise; document whatever your implementation does.
        """
        raise NotImplementedError

    @abstractmethod
    def load_catalog(
        self,
        table: Table,
        x_colname: str = "x",
        y_colname: str = "y",
        skycoord_colname: str = "coord",
        use_skycoord: bool = False,
        catalog_label: str | None = None,
        catalog_style: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Add catalog entries to the viewer at positions given by the catalog.

        Loading a catalog using a ``catalog_label`` that already exists will
        overwrite the existing catalog with the new one.

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
            The name of the column containing the sky coordinates, as
            `~astropy.coordinates.SkyCoord` objects. The column is only
            used for marker positions when ``use_skycoord`` is `True`.
            Default is ``'coord'``.
        use_skycoord : bool, optional
            Whether marker positions come from sky or pixel coordinates.
            If `True`, positions are taken from the ``skycoord_colname``
            column, or computed from the pixel position columns and the
            image's WCS if the table has no sky-coordinate column. If
            `False` (the default), positions are taken from the
            ``x_colname``/``y_colname`` columns, or computed from the
            sky-coordinate column and the image's WCS if the table has no
            pixel position columns. An error is raised when the requested
            mode is impossible with the available columns and WCS.
        catalog_label : str, optional
            The name to use for the catalog. If not given, a single shared
            default label is used, so loading a catalog without a label
            repeatedly replaces the previously loaded unlabeled catalog.
        catalog_style : dict, optional
            A dictionary that specifies the style of the markers used to
            represent the catalog. See
            `~astro_image_display_api.interface_definition.ImageViewerInterface.set_catalog_style`
            for details.
        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Raises
        ------
        ValueError
            If pixel positions are requested (``use_skycoord=False``) but the
            table has no pixel position columns and they cannot be computed
            (error message starts with "Cannot use pixel coordinates without
            pixel columns"), or if sky positions are requested
            (``use_skycoord=True``) but the table has no sky-coordinate
            column and there is no WCS to compute one (error message starts
            with "Cannot use sky coordinates without"). Also raised if a
            pixel/sky conversion is required but the backend cannot
            determine which image's WCS to use (e.g. because several
            images are loaded).

        Notes
        -----
        This should display the markers on the image in addition to storing
        the marker positions in the viewer.
        """
        raise NotImplementedError

    @abstractmethod
    def set_catalog_style(
        self,
        catalog_label: str | None = None,
        shape: str = "circle",
        color: str = "red",
        size: float = 5.0,
        **kwargs,
    ):
        """
        Set the style of the catalog markers.

        Parameters
        ----------
        catalog_label : str, optional
            The label of the catalog whose style is being set. If not given
            and there is only one catalog loaded, the style for that catalog
            is set. If there are multiple catalogs and no label is provided,
            an error is raised.
        shape : str, optional
            The shape of the markers. Default is ``'circle'``. The set of
            supported shapes is listed below in the *Note* section below.
        color : str, optional
            The color of the markers. Default is ``'red'``. Permitted colors are
            any CSS4 color name. CSS4 also permits hex RGB or RGBA colors.
        size : float, optional
            The size of the markers. Default is ``5.0``.

        **kwargs
            Additional keyword arguments to pass to the marker style.

        Raises
        ------
        ValueError
            If there are multiple catalog styles set and the user has not
            specified a ``catalog_label`` for which to set the style, or if
            an style is set for a catalog that does not exist.

        Notes
        -----
        The following shapes are supported: "circle", "square", "crosshair", "plus",
        "diamond".

        Changing the style of the markers should update the display of the
        markers in the image.
        """
        raise NotImplementedError

    @abstractmethod
    def get_catalog_style(self, catalog_label: str | None = None, **kwargs) -> dict:
        """
        Get the style of the catalog markers.

        Parameters
        ----------
        catalog_label : str, optional
            The name of the catalog. If not given and there is
            only one catalog loaded, the style for that catalog is returned.
            If there are multiple catalogs and no label is provided, an error
            is raised. If no label is given and no catalogs are loaded, the
            default style is returned.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Returns
        -------
        dict
            The style of the markers. The dictionary always contains at
            least the keys ``shape``, ``color`` and ``size``, and a
            ``catalog_label`` key whose value is the label of the catalog
            the style belongs to. When the default style is returned
            because no catalogs are loaded, the value of the
            ``catalog_label`` key is `None`.

        Raises
        ------

        ValueError
            If there are multiple catalog labels defined and the user has
            not specified a ``catalog_label`` for which to get the style, or
            if the ``catalog_label`` does not correspond to a loaded catalog.

        Notes
        -----
        This has no effect on the displayed image.

        """
        raise NotImplementedError

    @abstractmethod
    def remove_catalog(self, catalog_label: str | None = None, **kwargs) -> None:
        """
        Remove markers from the image.

        Parameters
        ----------
        catalog_label : str, optional
            The name of the catalog to remove. The value ``'*'`` can be used to
            remove all catalogs. If not given and there is
            only one catalog loaded, that catalog is removed.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Raises
        ------
        ValueError
            If the ``catalog_label`` is not provided when there are multiple
            catalogs loaded, or if the ``catalog_label`` does not correspond to a
            loaded catalog.

        TypeError
            If the ``catalog_label`` is a list. Only a single catalog label
            or ``'*'`` may be given.
        """
        raise NotImplementedError

    @abstractmethod
    def get_catalog(
        self,
        x_colname: str = "x",
        y_colname: str = "y",
        skycoord_colname: str = "coord",
        catalog_label: str | None = None,
        **kwargs,
    ) -> Table:
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
        catalog_label : str, optional
            The name of the catalog set to get.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Returns
        -------
        table : `astropy.table.Table`
            The table containing the marker positions. If no catalogs are
            loaded and no ``catalog_label`` is given, an empty table is
            returned.

        Raises
        ------
        ValueError
            If the ``catalog_label`` is not provided when there are multiple catalogs
            loaded, or if the ``catalog_label`` does not correspond to a loaded
            catalog.

        Notes
        -----
        This has no effect on the displayed image.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def catalog_labels(self) -> tuple[str, ...]:
        """
        Names of the loaded catalogs.

        Returns
        -------
        tuple of str
            The names of the loaded catalogs.

        Notes
        -----
        This has no effect on the displayed image.
        """
        raise NotImplementedError

    # Methods that modify the view
    @abstractmethod
    def set_viewport(
        self,
        center: SkyCoord | tuple[float, float] | None = None,
        fov: Quantity | float | None = None,
        image_label: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Set the viewport of the image, which defines the center and field of view.

        Parameters
        ----------
        center : `astropy.coordinates.SkyCoord` or tuple of float, optional
            The center of the viewport. If not given, the current center is used.
        fov : `astropy.units.Quantity` or float, optional
            The field of view (FOV) of the viewport. If not given, the current FOV
            is used. If a float is given, it is interpreted as a size in pixels. For
            viewers that are not square, the FOV is interpreted as the size of the
            shorter side of the viewer such that the FOV is guaranteed to be entirely
            visible regardless of the aspect ratio of the viewer.
        image_label : str, optional
            The label of the image to set the viewport for. If not given and there is
            only one image loaded, the viewport for that image is set. If there are
            multiple images and no label is provided, an error is raised.

        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Raises
        ------
        TypeError
            If the ``center`` is not a `~astropy.coordinates.SkyCoord` object
            or a tuple of floats (error message starts with "Invalid value
            for center"), or if the ``fov`` is not an angular
            `~astropy.units.Quantity` or a float (error message starts with
            "Invalid value for fov"). Also raised if the image has no WCS
            and a sky value is given where the stored viewport is in pixels,
            or vice versa, so that no conversion is possible; those error
            messages start with "Center must be a tuple", "Center must be a
            SkyCoord" or "FOV must be a float".

        ValueError
            If ``image_label`` is not provided when there are multiple images loaded.

        `astropy.units.UnitTypeError`
            If the ``fov`` is a `~astropy.units.Quantity` but does not have an angular
            unit (error message starts with "Incorrect unit for fov").

        Notes
        -----
        Setting the viewport should update the display of the image to reflect the new
        viewport.
        """
        raise NotImplementedError

    @abstractmethod
    def get_viewport(
        self,
        sky_or_pixel: str | None = None,
        image_label: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get the current viewport of the image.

        Parameters
        ----------
        sky_or_pixel : str, optional
            If 'sky', the center will be returned as a `~astropy.coordinates.SkyCoord`
            object. If 'pixel', the center will be returned as a tuple of pixel
            coordinates.
            If `None`, the default behavior is to return the center as a
            `~astropy.coordinates.SkyCoord` if
            possible, or as a tuple of floats if the image is in pixel coordinates and
            has no WCS information.
        image_label : str, optional
            The label of the image to get the viewport for. If not given and there is
            only one image loaded, the viewport for that image is returned. If there
            are multiple images and no label is provided, an error is raised.
        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Returns
        -------
        dict
            A dictionary containing the current viewport settings.
            The keys are 'center', 'fov', and 'image_label'.
            - 'center' is an `~astropy.coordinates.SkyCoord` object or a tuple of
            floats.
            - 'fov' is an `~astropy.units.Quantity` object or a float.
            - 'image_label' is a string representing the label of the image
            (the default label if the image was loaded without one).

        Raises
        ------
        ValueError
            If the ``sky_or_pixel`` parameter is not one of 'sky', 'pixel', or `None`,
            or if the ``image_label`` is not provided when there are multiple images
            loaded, or if the ``image_label`` does not correspond to a loaded image.
            Also raised if a conversion between pixel and sky coordinates is
            needed to satisfy the requested ``sky_or_pixel`` but the image
            has no WCS; the error message contains "WCS is not set".

        Notes
        -----
        This has no effect on the displayed image.
        """
        raise NotImplementedError
