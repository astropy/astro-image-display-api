.. astro-image-display-api documentation master file, created by
   sphinx-quickstart on Sun Jun 15 17:17:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Astronomical Image Display API
==============================

Who is this for?
################

The purpose of this API is to provide a standard interface for displaying
astronomical images. This package and API is aimed at *developers* who
want to write a package that displays astronomical images and provide a
uniform interface for users to interact with these images programmatically.

If you are a user looking for a way to display astronomical images, please
see the :ref:`aida_backends` page, which has a list of packages that implement
this API.

How to implement this API in your package
#########################################

1. The API is described in the :ref:`api_reference`. It consists of a set of methods with
   type annotations and extensive docstrings that describes the expected behavior.
   Note that you do **not** need to subclass the
   :py:class:`~astro_image_display_api.interface_definition.ImageViewerInterface`. The next
   step explains how to assert that your package implements the API correctly.
2. :ref:`testing_AIDA_implementation` describes how to test your
   package against the API.
3. There is a :ref:`sample implementation <reference_implementation>` of the API that you can use as a
   starting point for your own package. This reference implementation does not
   do any image display itself, but provides a set of methods that you can
   override to implement your own image display logic on top of the management
   of image and catalog labels. You are **not** required to use this reference
   implementation; it is just a convenience you can use to get started if you
   want to.

.. toctree::
  :maxdepth: 1
  :hidden:

  api.rst
  testing_implementation.rst
  reference_implementation.rst
  aida_backends.rst
