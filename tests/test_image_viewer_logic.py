# All implementations of the ImageViewerInterface will need to import
# these to carry out the tests.
from astro_image_display_api import ImageViewerInterface, ImageAPITest  # noqa: I001

# This import should be replaced with an import of your specific
# implementation of the ImageViewerInterface. If you keep the
# "as ImageViewer" part, then the test below will work without modification.
from astro_image_display_api.image_viewer_logic import ImageViewerLogic as ImageViewer

# You should not need to change the test below.


def test_instance():
    # Make sure that the ImageViewer has all required methods and attributes
    # defined in the ImageViewerInterface.
    image = ImageViewer()
    assert isinstance(image, ImageViewerInterface)


class TestViewer(ImageAPITest):
    """
    Test whether the non-display aspects of the ImageViewer
    implementation are correct.
    """

    image_widget_class = ImageViewer
