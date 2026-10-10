import contextlib
import numbers
import os
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData, NDData
from astropy.table import Table
from astropy.units import Quantity
from astropy.visualization import (
    AsymmetricPercentileInterval,
    BaseInterval,
    BaseStretch,
    LinearStretch,
    ManualInterval,
)
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from numpy.typing import ArrayLike

from .interface_definition import ImageViewerInterface

__all__ = ["ImageViewerLogic", "docs_from_image_viewer_logic_if_missing"]

#: Label used for an image or catalog that is loaded without an explicit
#: label. There is only ever one unlabeled image and one unlabeled catalog,
#: so loading without a label repeatedly replaces the previous unlabeled
#: image or catalog rather than accumulating new entries. The value is
#: deliberately unlikely to collide with a label a user would choose
#: themselves.
DEFAULT_LABEL = "_internal_default_label"


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
    data: ArrayLike | NDData | CCDData | None = None


def _copy_missing_docs(cls, source):
    """
    Copy docstrings from ``source`` to the same-named public methods of
    ``cls`` that lack a docstring of their own. Methods that already have
    a docstring keep it.
    """
    for name, method in cls.__dict__.items():
        if not name.startswith("_") and not method.__doc__:
            source_method = getattr(source, name, None)
            if source_method is not None:
                method.__doc__ = source_method.__doc__
    return cls


def docs_from_interface(cls):
    """
    Decorator to copy the docstrings from the interface methods to the
    methods in the class. Methods that already have a docstring of their
    own keep it.
    """
    return _copy_missing_docs(cls, ImageViewerInterface)


@dataclass
@docs_from_interface
class ImageViewerLogic:
    """
    This viewer does not do anything except making changes to its internal
    state to simulate the behavior of a real viewer.
    """

    # some internal variable for keeping track of viewer state
    _center: tuple[numbers.Real, numbers.Real] = (0.0, 0.0)

    def __post_init__(self):
        self._set_up_catalog_image_dicts()
        # Labels of the images the viewer is currently displaying. Today a
        # viewer displays a single image at a time, so the tuple has at most
        # one element and ``load_image`` uses replace semantics, but a tuple
        # leaves room for multi-image display in the future. The ``_apply_*``
        # hooks are only invoked for labels that are members of this tuple.
        self._displayed_image_labels: tuple[str, ...] = ()

    def _set_up_catalog_image_dicts(self):
        # Keys are the user-visible labels of the loaded catalogs and images.
        # Entries are created only by the load_* methods, so reading state
        # can never create an entry.
        self._catalogs: dict[str, CatalogInfo] = {}
        self._images: dict[str, ViewportInfo] = {}

    def _resolve_label(
        self,
        label: str | None,
        kind: str,
        allow_new: bool = False,
    ) -> str:
        """
        Resolve a user-provided image or catalog label.

        Parameters
        ----------
        label : str or None
            The label the user provided, or None if they did not provide one.
        kind : str
            Either ``"image"`` or ``"catalog"``; selects which registry the
            label is resolved against and is used in error messages.
        allow_new : bool, optional
            If True the label is being resolved for a load operation, so an
            explicit label need not already exist and a missing label
            resolves to the single shared default label instead of raising.

        Returns
        -------
        str
            The resolved label.

        Raises
        ------
        ValueError
            If an explicit label does not correspond to loaded data, or, when
            no label is given, if nothing is loaded or if several labels
            exist so the choice is ambiguous. Never raised when ``allow_new``
            is True.

        Notes
        -----
        This is needed so that the user gets what they expect in the simple
        case where there is only one image or catalog loaded. In that case
        the user may or may not have actually specified a label.
        """
        registry, article = (
            (self._images, "an image")
            if kind == "image"
            else (self._catalogs, "a catalog")
        )

        if label is not None:
            if not allow_new and label not in registry:
                raise ValueError(
                    f"{kind.capitalize()} label '{label}' not found. "
                    f"Please load {article} first."
                )
            return label

        if allow_new:
            # A load without an explicit label always targets the single
            # shared default label, so repeated unlabeled loads replace the
            # previous unlabeled image or catalog rather than piling up new
            # ones.
            return DEFAULT_LABEL

        match len(registry):
            case 0:
                raise ValueError(f"No {kind} is loaded. Please load {article} first.")
            case 1:
                return list(registry)[0]
            case _:
                raise ValueError(
                    f"Multiple {kind} labels defined. Please specify a "
                    f"{kind}_label to select one."
                )

    def _resolve_catalog_label(
        self, catalog_label: str | None, allow_new: bool = False
    ) -> str:
        """
        Resolve a user-provided catalog label; see `_resolve_label`.
        """
        return self._resolve_label(catalog_label, "catalog", allow_new=allow_new)

    # Rendering hooks for backends
    #
    # The public API methods of this class are templates: they own all state
    # handling and label resolution, then call the hooks below so that a
    # backend can push the already-validated state into its display. Every
    # hook receives a *resolved* label (never None, never "*"). All hooks are
    # no-ops here, which keeps this class a valid headless reference
    # implementation of the interface.

    def _render_image(self, image_label: str) -> None:
        """
        Display the image stored under ``image_label``.

        Called by `load_image` after the image data, viewport, cuts, stretch
        and colormap for the new image have all been stored and
        ``image_label`` has become the displayed image.

        Parameters
        ----------
        image_label : str
            The resolved label of the image to display.
        """

    def _apply_cuts(self, image_label: str) -> None:
        """
        Push the stored cuts for ``image_label`` into the display.

        Called by `set_cuts` and `load_image`, and only when ``image_label``
        is one of the displayed images.

        Parameters
        ----------
        image_label : str
            The resolved label of the image whose cuts changed.
        """

    def _apply_stretch(self, image_label: str) -> None:
        """
        Push the stored stretch for ``image_label`` into the display.

        Called by `set_stretch` and `load_image`, and only when
        ``image_label`` is one of the displayed images.

        Parameters
        ----------
        image_label : str
            The resolved label of the image whose stretch changed.
        """

    def _apply_colormap(self, image_label: str) -> None:
        """
        Push the stored colormap for ``image_label`` into the display.

        Called by `set_colormap` and `load_image`, and only when
        ``image_label`` is one of the displayed images.

        Parameters
        ----------
        image_label : str
            The resolved label of the image whose colormap changed.
        """

    def _apply_viewport(self, image_label: str) -> None:
        """
        Push the stored viewport for ``image_label`` into the display.

        Called by `set_viewport` and `load_image`, and only when
        ``image_label`` is one of the displayed images.

        Parameters
        ----------
        image_label : str
            The resolved label of the image whose viewport changed.
        """

    def _draw_catalog(self, catalog_label: str) -> None:
        """
        Draw (or redraw) the markers for the catalog ``catalog_label``.

        Called by `load_catalog` and `set_catalog_style` after the catalog
        data and style have been stored.

        Parameters
        ----------
        catalog_label : str
            The resolved label of the catalog to draw.
        """

    def _remove_catalog_marks(self, catalog_label: str) -> None:
        """
        Remove the markers drawn for the catalog ``catalog_label``.

        Called by `remove_catalog` after the catalog has been removed from
        the stored state. `remove_catalog` expands ``"*"`` itself and calls
        this hook once per removed catalog, so ``catalog_label`` is always
        the label of a single catalog.

        Parameters
        ----------
        catalog_label : str
            The resolved label of the catalog whose markers to remove.
        """

    def _batch_update(self):
        """
        Context manager wrapping a group of display updates.

        `load_image` wraps its state changes and hook calls in this context
        manager. Backends can override it to suppress intermediate redraws
        (e.g. by holding widget synchronization) until the batch completes.

        Returns
        -------
        context manager
            By default `contextlib.nullcontext`, i.e. no batching.
        """
        return contextlib.nullcontext()

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

    def get_stretch(
        self,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> BaseStretch:
        image_label = self._resolve_image_label(image_label)
        return self._images[image_label].stretch

    def set_stretch(
        self,
        stretch: BaseStretch,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        if not isinstance(stretch, BaseStretch):
            raise TypeError(
                f"Stretch option {stretch} is not valid. Must be an "
                "`astropy.visualization` Stretch object."
            )
        image_label = self._resolve_image_label(image_label)
        self._images[image_label].stretch = stretch
        if image_label in self._displayed_image_labels:
            self._apply_stretch(image_label)

    def get_cuts(
        self,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> BaseInterval:
        image_label = self._resolve_image_label(image_label)
        return self._images[image_label].cuts

    def set_cuts(
        self,
        cuts: tuple[numbers.Real, numbers.Real] | BaseInterval,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        if isinstance(cuts, tuple) and len(cuts) == 2:
            cuts = ManualInterval(cuts[0], cuts[1])
        elif not isinstance(cuts, BaseInterval):
            raise TypeError(
                "Cuts must be an Astropy.visualization Interval object or a tuple "
                "of two values."
            )
        image_label = self._resolve_image_label(image_label)
        self._images[image_label].cuts = cuts
        if image_label in self._displayed_image_labels:
            self._apply_cuts(image_label)

    @staticmethod
    def _validate_colormap_name(map_name: str) -> None:
        """
        Check that ``map_name`` is a valid matplotlib colormap name.

        Parameters
        ----------
        map_name : str
            The name of the colormap to validate.

        Raises
        ------
        ValueError
            If ``map_name`` is not the name of a matplotlib colormap.

        Notes
        -----
        matplotlib is not a dependency of this package; if it is not
        installed the name is accepted without validation.
        """
        try:
            from matplotlib import colormaps
        except ImportError:  # pragma: no cover
            return
        if map_name not in colormaps:
            raise ValueError(f"Colormap '{map_name}' is not a valid colormap name.")

    def set_colormap(
        self,
        map_name: str,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        self._validate_colormap_name(map_name)
        image_label = self._resolve_image_label(image_label)
        self._images[image_label].colormap = map_name
        if image_label in self._displayed_image_labels:
            self._apply_colormap(image_label)

    def get_colormap(
        self,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> str | None:
        image_label = self._resolve_image_label(image_label)
        return self._images[image_label].colormap

    # The methods, grouped loosely by purpose

    def get_catalog_style(
        self,
        catalog_label=None,
        **kwargs,  # noqa: ARG002
    ) -> dict[str, Any]:
        if catalog_label is None and not self._catalogs:
            # Nothing is loaded, so report the default style that a new
            # catalog would get.
            style = self._default_catalog_style
            style["catalog_label"] = None
            return style

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
        **kwargs,
    ) -> None:
        if not self._catalogs:
            raise ValueError("Must load a catalog before setting a catalog style.")

        catalog_label = self._resolve_catalog_label(catalog_label)

        self._catalogs[catalog_label].style = dict(
            shape=shape, color=color, size=size, **kwargs
        )

        self._draw_catalog(catalog_label)

    # Methods for loading data
    def _resolve_image_label(
        self, image_label: str | None, allow_new: bool = False
    ) -> str:
        """
        Resolve a user-provided image label; see `_resolve_label`.
        """
        return self._resolve_label(image_label, "image", allow_new=allow_new)

    def load_image(
        self,
        data: str | os.PathLike | ArrayLike | NDData,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """
        Load a FITS file, 2D array or `~astropy.nddata.NDData` object into
        the viewer and display it.

        Parameters
        ----------
        data : str, `os.PathLike`, array-like or `~astropy.nddata.NDData`
            The data to load. A string or path is interpreted as the name
            of a FITS file (or of an ASDF file when it ends in ``.asdf``).
        image_label : str, optional
            The label for the image. If not given, a single shared default
            label is used, so loading an image without a label repeatedly
            replaces the previously loaded unlabeled image.
        **kwargs
            Additional keyword arguments that may be used by the viewer.

        Notes
        -----
        Loading an image sets an appropriate viewport for that image and
        makes it the displayed image, replacing the image that was
        displayed before.

        Loading data under an ``image_label`` that already exists replaces
        that label's data but keeps the cuts, stretch and colormap that
        were set for the label; the viewport is reset to fit the new data.
        An image loaded under a new label starts from the default settings.
        """
        image_label = self._resolve_image_label(image_label, allow_new=True)

        # When data is loaded under an existing label, that label keeps the
        # cuts/stretch/colormap the user attached to it; the data, WCS and
        # viewport come from the new image. A new label starts from the
        # default settings.
        previous = self._images.get(image_label)
        previous_displayed = self._displayed_image_labels

        with self._batch_update():
            # Nothing is displayed while the new image's state is being set
            # up, so the set_* calls made during initialization below do not
            # fire any of the _apply_* hooks; the hooks are called once, in a
            # fixed order, at the end of this method.
            self._displayed_image_labels = ()

            # Start from a fresh entry; the settings an existing label keeps
            # are restored after loading.
            self._images[image_label] = ViewportInfo()

            try:
                if isinstance(data, str | os.PathLike):
                    # os.fsdecode normalizes str, bytes and any os.PathLike
                    # (which need not be a pathlib.Path with a .suffix).
                    if os.fsdecode(data).endswith(".asdf"):
                        self._load_asdf(data, image_label)
                    else:
                        self._load_fits(data, image_label)
                elif isinstance(data, NDData):
                    self._load_nddata(data, image_label)
                else:
                    # Assume it is a 2D array
                    self._load_array(data, image_label)
            except Exception:
                # The load failed, so the viewer is still displaying whatever
                # it displayed before. Put back the previous entry for this
                # label (or remove the fresh one if the label was new) and
                # the displayed-image tracking, so the _apply_* hooks keep
                # firing for the image that is still on screen.
                if previous is not None:
                    self._images[image_label] = previous
                else:
                    del self._images[image_label]
                self._displayed_image_labels = previous_displayed
                raise

            # Restore the settings this label already had, overriding the
            # defaults set while loading.
            if previous is not None:
                self._images[image_label].cuts = previous.cuts
                self._images[image_label].stretch = previous.stretch
                self._images[image_label].colormap = previous.colormap

            # The new image replaces whatever was displayed before.
            self._displayed_image_labels = (image_label,)

            self._render_image(image_label)
            self._apply_cuts(image_label)
            self._apply_stretch(image_label)
            self._apply_colormap(image_label)
            self._apply_viewport(image_label)

    def get_image(
        self,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> ArrayLike | NDData | CCDData:
        image_label = self._resolve_image_label(image_label)
        return self._images[image_label].data

    @property
    def image_labels(self) -> tuple[str, ...]:
        return tuple(self._images)

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
        # With 0-indexed pixel-center coordinates the center of the
        # image is at ((width - 1) / 2, (height - 1) / 2).
        center = ((width - 1) / 2, (height - 1) / 2)
        fov = max(image_data.shape)
        self._images[image_label].largest_dimension = self._determine_largest_dimension(
            image_data.shape
        )

        wcs = self._images[image_label].wcs
        # Is there a WCS set? If yes, make center a SkyCoord and fov a Quantity,
        # otherwise leave them as pixels.
        if wcs is not None:
            center = wcs.pixel_to_world(center[0], center[1])
            # largest_dimension indexes the numpy shape tuple (0 = y, 1 = x),
            # but proj_plane_pixel_scales is ordered by WCS pixel axis
            # (0 = x, 1 = y), so flip the index.
            fov = (
                fov
                * u.degree
                * proj_plane_pixel_scales(wcs)[
                    1 - self._images[image_label].largest_dimension
                ]
            )

        self.set_viewport(center=center, fov=fov, image_label=image_label)

        # Now set the stretch and cuts
        self.set_cuts(AsymmetricPercentileInterval(1, 95), image_label=image_label)
        self.set_stretch(LinearStretch(), image_label=image_label)

    def _load_fits(self, file: str | os.PathLike, image_label: str | None) -> None:
        ccd = CCDData.read(file)
        self._images[image_label].wcs = ccd.wcs
        self._images[image_label].data = ccd
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
        self._images[image_label].largest_dimension = self._determine_largest_dimension(
            array.shape
        )
        self._images[image_label].data = array
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
        self._images[image_label].data = data
        self._images[image_label].largest_dimension = self._determine_largest_dimension(
            data.data.shape
        )
        # Not all NDDData objects have a shape, apparently
        self._initialize_image_viewport_stretch_cuts(data.data, image_label)

    def _load_asdf(self, asdf_file: str | os.PathLike, image_label: str | None) -> None:
        """
        Not implementing some load types is fine.
        """
        raise NotImplementedError(
            "ASDF loading is not implemented in this dummy viewer."
        )

    # Saving contents of the view and accessing the view
    def save(
        self,
        filename: str | os.PathLike,
        overwrite: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        p = Path(filename)
        if p.exists() and not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Use overwrite=True to overwrite it."
            )

        p.write_text("This is a dummy file. The viewer does not save anything.")

    # Marker-related methods
    def _catalog_conversion_wcs(self, conversion_required: bool) -> WCS | None:
        """
        Return the WCS to use for catalog pixel↔sky coordinate conversion.

        Parameters
        ----------
        conversion_required : bool
            Whether a pixel↔sky conversion is required to satisfy the
            ``load_catalog`` call, as opposed to being merely opportunistic.

        Returns
        -------
        `astropy.wcs.WCS` or None
            The WCS of the single loaded or single displayed image, or
            None.

        Raises
        ------
        ValueError
            If a conversion is required, several images are loaded, and
            the displayed image does not disambiguate the choice.

        Notes
        -----
        The WCS is chosen with the same defaulting rule used for labels
        when no label is given: if exactly one image is loaded, that
        image's WCS is used. With no image loaded there is no WCS. With
        several images loaded, the image the viewer is currently
        displaying disambiguates: if exactly one image is displayed, its
        WCS is used. Otherwise the choice is ambiguous, so if a
        conversion is actually required an error is raised; if not, no
        WCS is used, i.e. the optional enrichment of the catalog with
        the coordinates that are not in the table is skipped rather than
        done with an arbitrary image's WCS.
        """
        match len(self._images):
            case 0:
                return None
            case 1:
                return list(self._images.values())[0].wcs
            case _:
                if len(self._displayed_image_labels) == 1:
                    return self._images[self._displayed_image_labels[0]].wcs
                if conversion_required:
                    raise ValueError(
                        "Multiple image labels defined. Cannot determine "
                        "which image's WCS to use to convert catalog "
                        "coordinates."
                    )
                return None

    def load_catalog(
        self,
        table: Table,
        x_colname: str = "x",
        y_colname: str = "y",
        skycoord_colname: str = "coord",
        use_skycoord: bool = False,
        catalog_label: str | None = None,
        catalog_style: dict | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        try:
            coords = table[skycoord_colname]
        except KeyError:
            coords = None

        try:
            xy = (table[x_colname], table[y_colname])
        except KeyError:
            xy = None

        # A conversion is required, not just opportunistic, when the pixel
        # columns must be computed from the sky coordinates or when sky
        # coordinates were requested but are not in the table.
        wcs = self._catalog_conversion_wcs(
            conversion_required=(xy is None and coords is not None)
            or (coords is None and use_skycoord)
        )

        to_add = table.copy()
        if xy is None:
            if wcs is not None and coords is not None:
                x, y = wcs.world_to_pixel(coords)
                to_add[x_colname] = x
                to_add[y_colname] = y
                xy = (x, y)
            else:
                to_add[x_colname] = to_add[y_colname] = None

        if not use_skycoord and xy is None:
            raise ValueError(
                "Cannot use pixel coordinates without pixel columns or both "
                "coordinates and a WCS."
            )

        if coords is None:
            if use_skycoord and wcs is None:
                raise ValueError(
                    "Cannot use sky coordinates without a SkyCoord column or WCS."
                )
            elif xy is not None and wcs is not None:
                # If we have xy coordinates, convert them to sky coordinates
                coords = wcs.pixel_to_world(xy[0], xy[1])
                to_add[skycoord_colname] = coords
            else:
                to_add[skycoord_colname] = None

        # Store the position columns under canonical internal names so that
        # get_catalog can return them under any requested names.
        to_add.rename_columns(
            [x_colname, y_colname, skycoord_colname], ["x", "y", "coord"]
        )

        catalog_label = self._resolve_catalog_label(catalog_label, allow_new=True)

        if catalog_label not in self._catalogs:
            self._catalogs[catalog_label] = CatalogInfo()

        # Set the new data
        self._catalogs[catalog_label].data = to_add

        # Ensure a catalog always has a style
        if catalog_style is None:
            if not self._catalogs[catalog_label].style:
                # No style has been set, so use the default style
                catalog_style = self._default_catalog_style.copy()
            else:
                # Use the existing style
                catalog_style = self._catalogs[catalog_label].style.copy()

        self._catalogs[catalog_label].style = catalog_style

        self._draw_catalog(catalog_label)

    def remove_catalog(
        self,
        catalog_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        if isinstance(catalog_label, list):
            raise TypeError(
                "Cannot remove multiple catalogs from a list. Please specify "
                "a single catalog label or use '*' to remove all catalogs."
            )
        elif catalog_label == "*":
            # If the user wants to remove all catalogs, we reset the
            # catalogs dictionary to an empty one, which is exactly the
            # state a fresh viewer starts in. The "*" is expanded here,
            # so the _remove_catalog_marks hook is called once per catalog
            # and never sees the "*" itself.
            removed_labels = tuple(self._catalogs)
            self._catalogs = {}
            with self._batch_update():
                for removed_label in removed_labels:
                    self._remove_catalog_marks(removed_label)
            return

        # Special cases are done, so we can resolve the catalog label.
        # Resolution raises a ValueError if the label is not found.
        catalog_label = self._resolve_catalog_label(catalog_label)

        del self._catalogs[catalog_label]

        self._remove_catalog_marks(catalog_label)

    def get_catalog(
        self,
        x_colname: str = "x",
        y_colname: str = "y",
        skycoord_colname: str = "coord",
        catalog_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> Table:
        # Docstring is copied from the interface definition, so it is not
        # duplicated here.
        catalog_label = self._resolve_catalog_label(catalog_label)

        # Return a copy so that modifying the returned table cannot corrupt
        # the stored catalog, renamed from the canonical internal column
        # names to the requested ones.
        result = self._catalogs[catalog_label].data.copy()
        result.rename_columns(
            ["x", "y", "coord"], [x_colname, y_colname, skycoord_colname]
        )

        return result

    @property
    def catalog_labels(self) -> tuple[str, ...]:
        return tuple(self._catalogs)

    # Methods that modify the view
    def set_viewport(
        self,
        center: SkyCoord | tuple[numbers.Real, numbers.Real] | None = None,
        fov: Quantity | numbers.Real | None = None,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        image_label = self._resolve_image_label(image_label)

        # Get current center/fov, if any, so that the user may input only one of them
        # after the initial setup if they wish.
        current_viewport = copy(self._images[image_label])
        if center is None:
            center = current_viewport.center
        if fov is None:
            fov = current_viewport.fov

        # If either center or fov is None these checks will raise an appropriate error
        if not isinstance(center, SkyCoord | tuple):
            raise TypeError(
                "Invalid value for center. Center must be a SkyCoord or tuple "
                "of (X, Y)."
            )
        if not isinstance(fov, Quantity | numbers.Real):
            raise TypeError(
                "Invalid value for fov. fov must be an angular Quantity or float."
            )

        if isinstance(fov, Quantity) and not fov.unit.is_equivalent(u.deg):
            raise u.UnitTypeError(
                "Incorrect unit for fov. fov must be an angular Quantity or float."
            )

        # Check that the center and fov are compatible with the current image
        if self._images[image_label].wcs is None:
            if current_viewport.center is not None:
                # If there is a WCS either input is fine. If there is no WCS then we
                # only check wther the new center is the same type as the
                # current center.
                if isinstance(center, SkyCoord) and not isinstance(
                    current_viewport.center, SkyCoord
                ):
                    raise TypeError(
                        "Center must be a tuple for this image when WCS is not set."
                    )
                elif isinstance(center, tuple) and not isinstance(
                    current_viewport.center, tuple
                ):
                    raise TypeError(
                        "Center must be a SkyCoord for this image when WCS is not set."
                    )
            if current_viewport.fov is not None:
                if isinstance(fov, Quantity) and not isinstance(
                    current_viewport.fov, Quantity
                ):
                    raise TypeError(
                        "FOV must be a float for this image when WCS is not set."
                    )
                elif isinstance(fov, numbers.Real) and not isinstance(
                    current_viewport.fov, numbers.Real
                ):
                    raise TypeError(
                        "FOV must be a float for this image when WCS is not set."
                    )

        # 😅 if we made it this far we should be able to handle the actual setting
        self._images[image_label].center = center
        self._images[image_label].fov = fov
        if image_label in self._displayed_image_labels:
            self._apply_viewport(image_label)

    def get_viewport(
        self,
        sky_or_pixel: str | None = None,
        image_label: str | None = None,
        **kwargs,  # noqa: ARG002
    ) -> dict[str, Any]:
        if sky_or_pixel not in (None, "sky", "pixel"):
            raise ValueError("sky_or_pixel must be 'sky', 'pixel', or None.")
        image_label = self._resolve_image_label(image_label)

        viewport = self._images[image_label]

        # Figure out what to return if the user did not specify sky_or_pixel.
        # The interface definition for get_viewport says that if the image has a WCS,
        # then the return should be in world coordinates, otherwise it should
        # be in pixel coordinates.
        if sky_or_pixel is None:
            if viewport.wcs is not None:
                # The requested image has a WCS, so return sky coordinates
                sky_or_pixel = "sky"
            else:
                # The requested image has no WCS, so return pixel coordinates
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
                    raise ValueError(
                        "WCS is not set. Cannot convert pixel coordinates to "
                        "sky coordinates."
                    )
                else:
                    if center is None:
                        center = viewport.wcs.pixel_to_world(
                            viewport.center[0], viewport.center[1]
                        )
                    if fov is None:
                        # largest_dimension indexes the numpy shape tuple
                        # (0 = y, 1 = x), but proj_plane_pixel_scales is
                        # ordered by WCS pixel axis (0 = x, 1 = y), so flip
                        # the index.
                        pixel_scale = proj_plane_pixel_scales(viewport.wcs)[
                            1 - viewport.largest_dimension
                        ]
                        fov = pixel_scale * viewport.fov * u.degree
        else:
            # Pixel coordinates
            if isinstance(viewport.center, SkyCoord):
                if viewport.wcs is None:
                    raise ValueError(
                        "WCS is not set. Cannot convert sky coordinates to "
                        "pixel coordinates."
                    )
                center = viewport.wcs.world_to_pixel(viewport.center)
            else:
                center = viewport.center
            if isinstance(viewport.fov, Quantity):
                if viewport.wcs is None:
                    raise ValueError(
                        "WCS is not set. Cannot convert FOV to pixel coordinates."
                    )
                # See comment above about flipping the index into
                # proj_plane_pixel_scales.
                pixel_scale = proj_plane_pixel_scales(viewport.wcs)[
                    1 - viewport.largest_dimension
                ]
                # proj_plane_pixel_scales returns degrees for a celestial WCS
                # (wcslib normalizes CUNIT to degrees), so convert the fov to
                # degrees rather than assuming it already is in degrees.
                fov = viewport.fov.to_value(u.degree) / pixel_scale
            else:
                fov = viewport.fov

        return dict(center=center, fov=fov, image_label=image_label)


def docs_from_image_viewer_logic_if_missing(cls):
    """
    Class decorator that fills in missing docstrings from `ImageViewerLogic`.

    Public methods of the decorated class that lack a docstring receive the
    docstring of the same-named method on
    `~astro_image_display_api.image_viewer_logic.ImageViewerLogic`. Backends
    that override interface methods can use this so that the overrides keep
    the documented API without duplicating the docstrings.

    Parameters
    ----------
    cls : type
        The class being decorated.

    Returns
    -------
    type
        The same class, with missing docstrings filled in.
    """
    return _copy_missing_docs(cls, ImageViewerLogic)
