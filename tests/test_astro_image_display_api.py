import re

from astropy.utils.data import get_pkg_data_contents

from astro_image_display_api import ImageViewerInterface


def test_api_test_class_covers_all_attributes_and_only_those_attributes():
    """
    Test that the ImageWidgetAPITest class is complete and has tests
    for all of the required methods and attributes and does not access
    any attributes or methods that are not part of the ImageViewerInterface.
    """
    # Get the attributes on the protocol
    required_attributes = ImageViewerInterface.__protocol_attrs__

    # Get the text of the api tests
    api_test_content = get_pkg_data_contents(
        "api_test.py", package="astro_image_display_api"
    )

    # This is the way the test class is accessed in the api_test.py file.
    image_viewer_name = "self.image"

    # Get all of the methods and attributes that are accessed
    # in the api_test.py file.
    # We use a regex to find all occurrences of the image_viewer_name
    # followed by a dot and then an attribute name.
    # This will match both attributes and methods.
    attributes_accessed_in_test_class = re.findall(
        rf"{image_viewer_name.replace(".", r"\.")}\.([a-zA-Z_][a-zA-Z0-9_]*)",
        api_test_content,
    )

    # Get the attribute/method names as a set
    attributes_accessed_in_test_class = list(set(attributes_accessed_in_test_class))

    # Make sure that the test class does not access any attributes
    # or methods that are not part of the ImageViewerInterface.
    attr_in_test_class_is_in_interface = []
    for attr in attributes_accessed_in_test_class:
        attr_in_test_class_is_in_interface.append(attr in required_attributes)

    attr_not_present_in_interface = [
        attr
        for attr, present in zip(
            attributes_accessed_in_test_class,
            attr_in_test_class_is_in_interface,
            strict=True,
        )
        if not present
    ]

    assert all(attr_in_test_class_is_in_interface), (
        f"ImageWidgetAPITest accesses these attributes/methods that are not part of "
        f"the ImageViewerInterface:\n{', '.join(attr_not_present_in_interface)}\n"
    )
    # Loop over the attributes and check that the test class has a method
    # for each one whose name starts with test_ and ends with the attribute
    # name.
    attr_present = []
    for attr in required_attributes:
        attr_present.append(f"{image_viewer_name}.{attr}" in api_test_content)

    missing_attributes = [
        attr
        for attr, present in zip(required_attributes, attr_present, strict=False)
        if not present
    ]
    missing_attributes_msg = "\n".join(missing_attributes)
    assert all(attr_present), (
        "ImageWidgetAPITest does not access these "
        f"attributes/methods:\n{missing_attributes_msg}\n"
    )


def test_every_method_attribute_has_docstring():
    """
    Check that every method and attribute in the protocol has a docstring.
    """
    from astro_image_display_api import ImageViewerInterface

    all_methods_and_attributes = ImageViewerInterface.__protocol_attrs__

    method_attrs_no_docs = []

    for method in all_methods_and_attributes:
        attr = getattr(ImageViewerInterface, method)
        # Make list of methods and attributes that have no docstring
        # and assert that list is empty at the end of the test.
        if not attr.__doc__:
            method_attrs_no_docs.append(method)

    assert not method_attrs_no_docs, (
        "The following methods and attributes have no docstring:\n\t"
        f"{'\n\t'.join(method_attrs_no_docs)}"
    )
