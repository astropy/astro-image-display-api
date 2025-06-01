import numbers
import os
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData, NDData
from astropy.table import Table, vstack
from astropy.units import Quantity, get_physical_type
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import AsymmetricPercentileInterval, BaseInterval, BaseStretch, LinearStretch, ManualInterval
from numpy.typing import ArrayLike

from .interface_definition import ImageViewerInterface

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
    _center: tuple[numbers.Real, numbers.Real] = (0.0, 0.0)


    def __post_init__(self):
        # Set up the initial state of the viewer
        self._images = defaultdict(ViewportInfo)
        self._images[None].center = None
        self._images[None].fov = None
        self._images[None].wcs = None

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

    def _intialize_image_viewport_stretch_cuts(
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
        self._intialize_image_viewport_stretch_cuts(ccd.data, image_label)

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
        self._intialize_image_viewport_stretch_cuts(array, image_label)

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
        self._intialize_image_viewport_stretch_cuts(data.data, image_label)

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
