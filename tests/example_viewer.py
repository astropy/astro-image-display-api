"""
A worked example of an `~astro_image_display_api.image_viewer_logic.ImageViewerLogic`
subclass, used in the documentation to show how a real viewer backend plugs into
the reference implementation.

The public API methods of ``ImageViewerLogic`` are templates: they validate their
arguments, resolve labels and store state, then call a small set of private
rendering hooks with the already-resolved label. A backend overrides those hooks
to push the stored state into its display. Instead of drawing anything,
:class:`RecordingViewer` overrides every hook to record, in order, the display
operations it was asked to perform. That makes it possible to write tests (and
documentation examples) that assert on *what* would have been drawn and in what
order, without needing an actual display backend.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from astro_image_display_api.image_viewer_logic import ImageViewerLogic

__all__ = ["RecordingViewer"]


class RecordingViewer(ImageViewerLogic):
    """
    An `~astro_image_display_api.image_viewer_logic.ImageViewerLogic` that records
    display operations instead of drawing them.

    Every rendering hook appends an entry to :attr:`display_calls` describing what
    was requested. A real viewer would instead update whatever GUI or plotting
    library it wraps; the state to push comes from the public getters, called with
    the label the hook received.
    """

    def __post_init__(self):
        # Let the base class set up its internal ``_images``/``_catalogs``
        # dictionaries first, then start our own call log.
        super().__post_init__()
        self.display_calls = []

    def _record(self, operation: str, **details: Any) -> None:
        """Append one (operation, details) entry to the display call log."""
        self.display_calls.append((operation, details))

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------
    @contextmanager
    def _batch_update(self):
        """Record the start and end of a batch of display updates."""
        # A real backend would hold off on redrawing until the block ends.
        # The ``finally`` matters: load_image re-raises loader errors from
        # inside the batch, and the batch must still be closed.
        self._record("batch_enter")
        try:
            yield
        finally:
            self._record("batch_exit")

    # ------------------------------------------------------------------
    # Image hooks
    # ------------------------------------------------------------------
    def _render_image(self, image_label: str) -> None:
        """Record that the image ``image_label`` is now displayed."""
        # A real backend would hand the pixel values to its plotting library
        # here. get_image() may return an NDData/CCDData rather than an
        # array, so use np.asarray(getattr(image, "data", image)).
        self._record("render_image", image_label=image_label)

    def _apply_cuts(self, image_label: str) -> None:
        """Record the cuts to apply to the displayed image."""
        self._record(
            "apply_cuts",
            image_label=image_label,
            cuts=self.get_cuts(image_label=image_label),
        )

    def _apply_stretch(self, image_label: str) -> None:
        """Record the stretch to apply to the displayed image."""
        self._record(
            "apply_stretch",
            image_label=image_label,
            stretch=self.get_stretch(image_label=image_label),
        )

    def _apply_colormap(self, image_label: str) -> None:
        """Record the colormap to apply to the displayed image."""
        self._record(
            "apply_colormap",
            image_label=image_label,
            colormap=self.get_colormap(image_label=image_label),
        )

    def _apply_viewport(self, image_label: str) -> None:
        """Record the viewport to apply to the displayed image."""
        self._record(
            "apply_viewport",
            image_label=image_label,
            viewport=self.get_viewport(image_label=image_label),
        )

    # ------------------------------------------------------------------
    # Catalog hooks
    # ------------------------------------------------------------------
    def _draw_catalog(self, catalog_label: str) -> None:
        """Record the markers to draw for ``catalog_label``."""
        self._record(
            "draw_catalog",
            catalog_label=catalog_label,
            n_rows=len(self.get_catalog(catalog_label=catalog_label)),
            style=self.get_catalog_style(catalog_label=catalog_label),
        )

    def _remove_catalog_marks(self, catalog_label: str) -> None:
        """Record that the markers for ``catalog_label`` were removed."""
        self._record("remove_catalog_marks", catalog_label=catalog_label)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save(
        self,
        filename: str | os.PathLike,
        overwrite: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Write the recorded display calls to ``filename`` instead of drawing."""
        # Unlike the hooks, this is a public method, so it needs a docstring
        # of its own. It does NOT call super() -- the base implementation just
        # writes a dummy placeholder, which is not useful here. Instead, write
        # out the log of everything that was "displayed".
        p = Path(filename)
        if p.exists() and not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Use overwrite=True to overwrite it."
            )

        p.write_text("\n".join(str(call) for call in self.display_calls))
