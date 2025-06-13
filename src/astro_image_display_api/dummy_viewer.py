import numbers
import os
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData, NDData
from astropy.table import Table, vstack
from astropy.units import Quantity
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import AsymmetricPercentileInterval, BaseInterval, BaseStretch, LinearStretch, ManualInterval
from numpy.typing import ArrayLike

from .interface_definition import ImageViewerInterface

@dataclass
class CatalogInfo:
    """
    Class to hold information about a catalog.
    """
    style: dict[str, Any] = field(default_factory=dict)
    data: Table | None = None

@dataclass
class ViewportInfo:
    """
    Class to hold image and viewport information.
    """
    center: SkyCoord | tuple[numbers.Real, numbers.Real] | None = None
    fov: float | Quantity | None = None
    wcs: WCS | None = None
    largest_dimension: int | None = None
    stretch: BaseStretch | None = None
    cuts: BaseInterval | tuple[numbers.Real, numbers.Real] | None = None
    colormap: str | None = None

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
    _cuts: BaseInterval | tuple[float, float] = AsymmetricPercentileInterval(upper_percentile=95)
    _stretch: BaseStretch = LinearStretch
    # viewer: Any

    # Allowed locations for cursor display
    ALLOWED_CURSOR_LOCATIONS: tuple = ImageViewerInterface.ALLOWED_CURSOR_LOCATIONS

    # some internal variable for keeping track of viewer state
    _wcs: WCS | None = None
    _center: tuple[numbers.Real, numbers.Real] = (0.0, 0.0)

    def __post_init__(self):
        # This is a dictionary of marker sets. The keys are the names of the
        # marker sets, and the values are the tables containing the markers.
        self._catalogs = defaultdict(CatalogInfo)
        self._catalogs[None].data = None
        self._catalogs[None].style = self._default_catalog_style.copy()

        self._images = defaultdict(ViewportInfo)
        self._images[None].center = None
        self._images[None].fov = None
        self._images[None].wcs = None


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

    @property
    def _default_catalog_style(self) -> dict[str, Any]:
        """
        The default style for the catalog markers.
        """
        return {
            "shape": "circle",
            "color": "red",
            "size": 5,
        }



    def get_stretch(self, image_label: str | None = None) -> BaseStretch:
        image_label = self._resolve_image_label(image_label)
        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")
        return self._images[image_label].stretch

    def set_stretch(self, value: BaseStretch, image_label: str | None = None) -> None:
        if not isinstance(value, BaseStretch):
            raise ValueError(f"Stretch option {value} is not valid. Must be an Astropy.visualization Stretch object.")
        image_label = self._resolve_image_label(image_label)
        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")
        self._images[image_label].stretch = value

    def get_cuts(self, image_label: str | None = None) -> tuple:
        image_label = self._resolve_image_label(image_label)
        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")
        return self._images[image_label].cuts

    def set_cuts(self, value: tuple[numbers.Real, numbers.Real] | BaseInterval, image_label: str | None = None) -> None:
        if isinstance(value, tuple) and len(value) == 2:
            self._cuts = ManualInterval(value[0], value[1])
        elif isinstance(value, BaseInterval):
            self._cuts = value
        else:
            raise ValueError("Cuts must be an Astropy.visualization Interval object or a tuple of two values.")
        image_label = self._resolve_image_label(image_label)
        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")
        self._images[image_label].cuts = self._cuts

    def set_colormap(self, map_name: str, image_label: str | None = None) -> None:
        image_label = self._resolve_image_label(image_label)
        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")
        self._images[image_label].colormap = map_name

    set_colormap.__doc__ = ImageViewerInterface.set_colormap.__doc__

    def get_colormap(self, image_label: str | None = None) -> str:
        image_label = self._resolve_image_label(image_label)
        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")
        return self._images[image_label].colormap

    get_colormap.__doc__ = ImageViewerInterface.get_colormap.__doc__

    @property
    def cursor(self) -> str:
        return self._cursor

    @cursor.setter
    def cursor(self, value: str) -> None:
        if value not in self.ALLOWED_CURSOR_LOCATIONS:
            raise ValueError(f"Cursor location {value} is not valid. Must be one of {self.ALLOWED_CURSOR_LOCATIONS}.")
        self._cursor = value

    # The methods, grouped loosely by purpose

    def get_catalog_style(self, catalog_label=None) -> dict[str, Any]:
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

        style = self._catalogs[catalog_label].style.copy()
        style["catalog_label"] = catalog_label
        return style

    def set_catalog_style(
            self,
            catalog_label: str | None = None,
            shape: str = "circle",
            color: str = "red",
            size: float = 5,
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
        catalog_label = self._resolve_catalog_label(catalog_label)

        if self._catalogs[catalog_label].data is None:
            raise ValueError("Must load a catalog before setting a catalog style.")

        self._catalogs[catalog_label].style = dict(
            shape=shape,
            color=color,
            size=size,
            **kwargs
        )

    # Methods for loading data
    def _user_image_labels(self) -> list[str]:
        """
        Get the list of user-defined image labels.

        Returns
        -------
        list of str
            The list of user-defined image labels.
        """
        return [label for label in self._images if label is not None]

    def _resolve_image_label(self, image_label: str | None) -> str:
        """
        Figure out the image label if the user did not specify one. This
        is needed so that the user gets what they expect in the simple case
        where there is only one image loaded. In that case the user may
        or may not have actually specified a image label.
        """
        user_keys = self._user_image_labels()
        if image_label is None:
            match len(user_keys):
                case 0:
                    # No user-defined image labels, so return the default label.
                    image_label = None
                case 1:
                    # The user must have loaded a image, so return that instead of
                    # the default label, which live in the key None.
                    image_label = user_keys[0]
                case _:
                    raise ValueError(
                        "Multiple image labels defined. Please specify a image_label to get the style."
                    )

        return image_label

    def load_image(self, file: str | os.PathLike | ArrayLike | NDData, image_label: str | None = None) -> None:
        """
        Load a FITS file into the viewer.

        Parameters
        ----------
        file : str or `astropy.io.fits.HDU`
            The FITS file to load. If a string, it can be a URL or a
            file path.

        image_label : str, optional
            A label for the image.
        """
        image_label = self._resolve_image_label(image_label)

        # Delete the current viewport if it exists
        if image_label in self._images:
            del self._images[image_label]

        if isinstance(file, (str, os.PathLike)):
            if isinstance(file, str):
                is_adsf = file.endswith(".asdf")
            else:
                is_asdf = file.suffix == ".asdf"
            if is_asdf:
                self._load_asdf(file, image_label)
            else:
                self._load_fits(file, image_label)
        elif isinstance(file, NDData):
            self._load_nddata(file, image_label)
        else:
            # Assume it is a 2D array
            self._load_array(file, image_label)

        # This may eventually get pulled, but for now is needed to keep markers
        # working with the new image.
        self._wcs = self._images[image_label].wcs

    def _determine_largest_dimension(self, shape: tuple[int, int]) -> int:
        """
        Determine which index is the largest dimension.

        Parameters
        ----------
        shape : tuple of int
            The shape of the image.

        Returns
        -------
        int
            The index of the largest dimension of the image, or 0 if square.
        """
        return int(shape[1] > shape[0])

    def _initialize_image_viewport_stretch_cuts(
        self,
        image_data: ArrayLike | NDData | CCDData,
        image_label: str | None,
    ) -> None:
        """
        Initialize the viewport, stretch and cuts for an image.

        Parameters
        ----------
        image_data : ArrayLike
            The image data to initialize the viewport for.
        image_label : str or None
            The label for the image. If None, the default label will be used.

        Note
        ----
        This method is called internally to set up the initial viewport,
        stretch, and cuts for the image. It should be called AFTER setting
        the WCS.
        """

        # Deal with the viewport first
        height, width = image_data.shape
        # Center the image in the viewport and show the whole image.
        center = (width / 2, height / 2)
        fov = max(image_data.shape)
        self._images[image_label].largest_dimension = self._determine_largest_dimension(image_data.shape)

        wcs = self._images[image_label].wcs
        # Is there a WCS set? If yes, make center a SkyCoord and fov a Quantity,
        # otherwise leave them as pixels.
        if wcs is not None:
            center = wcs.pixel_to_world(center[0], center[1])
            fov = fov * u.degree / proj_plane_pixel_scales(wcs)[self._images[image_label].largest_dimension]

        self.set_viewport(
            center=center,
            fov=fov,
            image_label=image_label
        )

        # Now set the stretch and cuts
        self.set_cuts(AsymmetricPercentileInterval(1, 95), image_label=image_label)
        self.set_stretch(LinearStretch(), image_label=image_label)

    def _load_fits(self, file: str | os.PathLike, image_label: str | None) -> None:
        ccd = CCDData.read(file)
        self._images[image_label].wcs = ccd.wcs
        self._initialize_image_viewport_stretch_cuts(ccd.data, image_label)

    def _load_array(self, array: ArrayLike, image_label: str | None) -> None:
        """
        Load a 2D array into the viewer.

        Parameters
        ----------
        array : array-like
            The array to load.
        """
        self._images[image_label].wcs = None  # No WCS for raw arrays
        self._images[image_label].largest_dimension = self._determine_largest_dimension(array.shape)
        self._initialize_image_viewport_stretch_cuts(array, image_label)

    def _load_nddata(self, data: NDData, image_label: str | None) -> None:
        """
        Load an `astropy.nddata.NDData` object into the viewer.

        Parameters
        ----------
        data : `astropy.nddata.NDData`
            The NDData object to load.
        """
        self._images[image_label].wcs = data.wcs
        self._images[image_label].largest_dimension = self._determine_largest_dimension(data.data.shape)
        # Not all NDDData objects have a shape, apparently
        self._initialize_image_viewport_stretch_cuts(data.data, image_label)

    def _load_asdf(self, asdf_file: str | os.PathLike, image_label: str | None) -> None:
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
                    skycoord_colname: str = 'coord', use_skycoord: bool = False,
                    catalog_label: str | None = None,
                    catalog_style: dict | None = None) -> None:
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

        # Either set new data or append to existing data
        if (
            catalog_label in self._catalogs
            and self._catalogs[catalog_label].data is not None
        ):
            # If the catalog already exists, we append to it
            old_table = self._catalogs[catalog_label].data
            self._catalogs[catalog_label].data = vstack([old_table, to_add])
        else:
            # If the catalog does not exist, we create a new one
            self._catalogs[catalog_label].data = to_add

        # Ensure a catalog always has a style
        if catalog_style is None:
            if not self._catalogs[catalog_label].style:
                catalog_style = self._default_catalog_style.copy()

        self._catalogs[catalog_label].style = catalog_style

    load_catalog.__doc__ = ImageViewerInterface.load_catalog.__doc__

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
            raise ValueError(f"Catalog label {catalog_label} not found.")

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
    def set_viewport(
                self, center: SkyCoord | tuple[numbers.Real, numbers.Real] | None = None,
        fov: Quantity | numbers.Real | None = None,
        image_label: str | None = None
    ) -> None:
        image_label = self._resolve_image_label(image_label)

        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")

        # Get current center/fov, if any, so that the user may input only one of them
        # after the initial setup if they wish.
        current_viewport = copy(self._images[image_label])
        if center is None:
            center = current_viewport.center
        if fov is None:
            fov = current_viewport.fov

        # If either center or fov is None these checks will raise an appropriate error
        if not isinstance(center, (SkyCoord, tuple)):
            raise TypeError("Invalid value for center. Center must be a SkyCoord or tuple of (X, Y).")
        if not isinstance(fov, (Quantity, numbers.Real)):
            raise TypeError("Invalid value for fov. fov must be an angular Quantity or float.")

        if isinstance(fov, Quantity) and not fov.unit.is_equivalent(u.deg):
            raise u.UnitTypeError("Incorrect unit for fov. fov must be an angular Quantity or float.")

        # Check that the center and fov are compatible with the current image
        if self._images[image_label].wcs is None:
            if current_viewport.center is not None:
                # If there is a WCS either input is fine. If there is no WCS then we only
                # check wther the new center is the same type as the current center.
                if isinstance(center, SkyCoord) and not isinstance(current_viewport.center, SkyCoord):
                    raise TypeError("Center must be a tuple for this image when WCS is not set.")
                elif isinstance(center, tuple) and not isinstance(current_viewport.center, tuple):
                    raise TypeError("Center must be a SkyCoord for this image when WCS is not set.")
            if current_viewport.fov is not None:
                if isinstance(fov, Quantity) and not isinstance(current_viewport.fov, Quantity):
                    raise TypeError("FOV must be a float for this image when WCS is not set.")
                elif isinstance(fov, numbers.Real) and not isinstance(current_viewport.fov, numbers.Real):
                    raise TypeError("FOV must be a float for this image when WCS is not set.")

        # 😅 if we made it this far we should be able to handle the actual setting
        self._images[image_label].center = center
        self._images[image_label].fov = fov


    set_viewport.__doc__ = ImageViewerInterface.set_viewport.__doc__

    def get_viewport(
        self, sky_or_pixel: str | None = None, image_label: str | None = None
    ) -> dict[str, Any]:
        if sky_or_pixel not in (None, "sky", "pixel"):
            raise ValueError("sky_or_pixel must be 'sky', 'pixel', or None.")
        image_label = self._resolve_image_label(image_label)

        if image_label not in self._images:
            raise ValueError(f"Image label '{image_label}' not found. Please load an image first.")

        viewport = self._images[image_label]

        # Figure out what to return if the user did not specify sky_or_pixel
        if sky_or_pixel is None:
            if isinstance(viewport.center, SkyCoord):
                # Somebody set this to sky coordinates, so return sky coordinates
                sky_or_pixel = "sky"
            elif isinstance(viewport.center, tuple):
                # Somebody set this to pixel coordinates, so return pixel coordinates
                sky_or_pixel = "pixel"

        center = None
        fov = None
        if sky_or_pixel == "sky":
            if isinstance(viewport.center, SkyCoord):
                center = viewport.center

            if isinstance(viewport.fov, Quantity):
                fov = viewport.fov

            if center is None or fov is None:
                # At least one of center or fov is not set, which means at least one
                # was not already sky, so we need to convert them or fail
                if viewport.wcs is None:
                    raise ValueError("WCS is not set. Cannot convert pixel coordinates to sky coordinates.")
                else:
                    center = viewport.wcs.pixel_to_world(viewport.center[0], viewport.center[1])
                    pixel_scale = proj_plane_pixel_scales(viewport.wcs)[viewport.largest_dimension]
                    fov = pixel_scale * viewport.fov * u.degree
        else:
            # Pixel coordinates
            if isinstance(viewport.center, SkyCoord):
                if viewport.wcs is None:
                    raise ValueError("WCS is not set. Cannot convert sky coordinates to pixel coordinates.")
                center = viewport.wcs.world_to_pixel(viewport.center)
            else:
                center = viewport.center
            if isinstance(viewport.fov, Quantity):
                if viewport.wcs is None:
                    raise ValueError("WCS is not set. Cannot convert FOV to pixel coordinates.")
                pixel_scale = proj_plane_pixel_scales(viewport.wcs)[viewport.largest_dimension]
                fov = viewport.fov.value / pixel_scale
            else:
                fov = viewport.fov

        return dict(
            center=center,
            fov=fov,
            image_label=image_label
        )


    get_viewport.__doc__ = ImageViewerInterface.get_viewport.__doc__
