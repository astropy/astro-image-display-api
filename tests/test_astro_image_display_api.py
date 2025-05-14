from astropy.utils.data import get_pkg_data_contents

from astro_image_display_api.interface_definition import ImageViewerInterface

def test_api_test_class_completeness():
    """
    Test that the ImageWidgetAPITest class is complete and has tests
    for all of the required methods and attributes.
    """
    # Get the attributes on the protocol
    required_attributes = ImageViewerInterface.__protocol_attrs__

    # Get the text of the api tests
    widget_api_test_content = get_pkg_data_contents("widget_api_test.py", package="astro_image_display_api")
    # Loop over the attributes and check that the test class has a method
    # for each one whose name starts with test_ and ends with the attribute
    # name.
    attr_present = []
    image_viewer_name = "self.image"
    for attr in required_attributes:
        attr_present.append(f"{image_viewer_name}.{attr}" in widget_api_test_content)



    assert all(attr_present), (
        "ImageWidgetAPITest does not access these attributes/methods:\n "
        f"{"\n".join(attr for attr, present in zip(required_attributes, attr_present) if not present)}. "
    )
