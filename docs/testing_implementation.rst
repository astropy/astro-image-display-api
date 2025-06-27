.. _testing_AIDA_implementation:

Testing
=======

Our goal is to make it easy for you to test your implementation of the
Astronomical Image Display API (AIDA). There are two things you need to test:

1. An instance of your class should pass the test
   ``isinstance(your_instance, ImageDisplayInterface)``.
   This ensures that your class has all of the attributes and methods in the interface.
2. To test the functionality of your implementation, we provide the class
   :py:class:`~astro_image_display_api.api_test.ImageAPITest`.
   To use it, create a subclass of :py:class:`~astro_image_display_api.api_test.ImageAPITest` in your test suite, and define
   a single class attribute ``image_widget_class``. See
   :ref:`test_example` for an example of how to do this.

These tests *do not* test the actual image display functionality of your package, nor
do they test the behaviour of your package via your package's user interface. You should
check that behavior in whatever way is appropriate for your package.

.. _test_example:

Example of using the test class
###############################

.. literalinclude:: ../tests/test_image_viewer_logic.py
  :language: python

.. automodapi:: astro_image_display_api.api_test
    :no-inheritance-diagram:
