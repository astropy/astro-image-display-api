import numpy as np

from astro_image_display_api import ImageViewerInterface
from astro_image_display_api.api_test import ImageAPITest

from .example_viewer import RecordingViewer


def _operations(viewer):
    return [op for op, _ in viewer.display_calls]


def test_instance():
    assert isinstance(RecordingViewer(), ImageViewerInterface)


def test_load_image_hook_sequence():
    # load_image stores all of the new image's state first, then calls
    # each rendering hook once, in a fixed order, inside one batch, with
    # the resolved label. See test_image_viewer_logic_implementation.py
    # for the full set of hook behavior tests.
    viewer = RecordingViewer()
    viewer.load_image(np.zeros((4, 6)), image_label="a")

    assert _operations(viewer) == [
        "batch_enter",
        "render_image",
        "apply_cuts",
        "apply_stretch",
        "apply_colormap",
        "apply_viewport",
        "batch_exit",
    ]
    labels = {d["image_label"] for op, d in viewer.display_calls if "image_label" in d}
    assert labels == {"a"}


def test_apply_hooks_only_for_displayed_image():
    # The _apply_* hooks fire only for the displayed image; settings for
    # any other loaded image are stored without touching the display.
    viewer = RecordingViewer()
    viewer.load_image(np.zeros((4, 6)), image_label="first")
    viewer.load_image(np.ones((4, 6)), image_label="second")  # now displayed
    viewer.display_calls.clear()

    viewer.set_cuts((0, 1), image_label="first")
    assert viewer.display_calls == []

    viewer.set_cuts((0, 1), image_label="second")
    assert _operations(viewer) == ["apply_cuts"]
    assert viewer.display_calls[0][1]["image_label"] == "second"


class TestRecordingViewer(ImageAPITest):
    image_widget_class = RecordingViewer
