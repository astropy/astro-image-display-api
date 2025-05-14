from astro_image_display_api.dummy_viewer import ImageViewer
from astro_image_display_api.interface_definition import ImageViewerInterface
from astro_image_display_api.widget_api_test import ImageWidgetAPITest


def test_instance():
    # Make sure that the ImageViewer class implements the ImageViewerInterface
    image = ImageViewer()
    assert isinstance(image, ImageViewerInterface)


class TestDummyViewer(ImageWidgetAPITest):
    """
    Test functionality of the ImageViewer class."""
    image_widget_class = ImageViewer
