from astro_image_display_api import ImageViewerInterface, ImageWidgetAPITest
from astro_image_display_api.dummy_viewer import ImageViewer


def test_instance():
    # Make sure that the ImageViewer class implements the ImageViewerInterface
    image = ImageViewer()
    assert isinstance(image, ImageViewerInterface)


class TestDummyViewer(ImageWidgetAPITest):
    """
    Test functionality of the ImageViewer class."""

    image_widget_class = ImageViewer
