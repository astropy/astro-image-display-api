.. _reference_implementation:

Building on the reference implementation
=========================================

The class :py:class:`~astro_image_display_api.image_viewer_logic.ImageViewerLogic`
is provided as an *example* of how to implement the non-display logic of the
Astronomical Image Display API (AIDA). You do not need to use this class, but
it is provided as a convenience to help you get started with your own
implementation: subclass it, override a handful of private *rendering
hooks* that push state into an actual display, and let the base class handle
validation, label bookkeeping, and state storage for you.

This page walks through that pattern in detail: what the base class does and
does not do, how it stores state, which hooks there are and when each one is
called, the order in which the hooks fire while an image is loading, when
(rarely) to override a public method instead, and the couple of gotchas that
trip people up (constructing a dataclass subclass, and docstrings on
overridden public methods). It ends with a small, fully tested worked
example.

What ``ImageViewerLogic`` does -- and does not do
--------------------------------------------------

``ImageViewerLogic`` implements everything in
:py:class:`~astro_image_display_api.interface_definition.ImageViewerInterface`
*except* actually drawing anything. Concretely, it does:

- Validate its arguments and raise the exact exceptions and messages that
  :py:class:`~astro_image_display_api.api_test.ImageAPITest` checks for (see
  `Error messages are part of the contract`_ below).
- Keep track of per-image and per-catalog state (viewport, cuts, stretch,
  colormap, catalog style and data), keyed by label.
- Resolve labels for you when a method is called without one, including
  raising an error when the label is ambiguous.
- Set sensible default cuts (``AsymmetricPercentileInterval(1, 95)``) and
  stretch (``LinearStretch``) when an image is loaded under a new label.
  Reloading under an existing label keeps that label's cuts, stretch, and
  colormap; see `What happens during load_image`_.
- Load FITS files, 2D arrays, and ``astropy.nddata.NDData`` objects.

It does **not**:

- Draw or render anything. There is no plotting library, canvas, or widget
  anywhere in this class.
- Actually save an image. Its
  :py:meth:`~astro_image_display_api.image_viewer_logic.ImageViewerLogic.save`
  method just writes a placeholder text file; see `Saving the view`_.
- Support ``.asdf`` files -- loading one raises ``NotImplementedError``.
- Validate colormap names against anything but Matplotlib. ``set_colormap``
  rejects a name Matplotlib does not know, if Matplotlib is installed, and
  otherwise stores whatever string it is given. A backend whose plotting
  library has its own colormap names has to check them itself; see
  `When to override a public method instead`_.

The state model
----------------

Two private dictionaries hold all of the state:

- ``self._images`` maps an image label to a ``ViewportInfo`` object, which
  holds ``center``, ``fov``, ``wcs``, ``largest_dimension``, ``stretch``,
  ``cuts``, ``colormap``, and ``data``.
- ``self._catalogs`` maps a catalog label to a ``CatalogInfo`` object, which
  holds ``style`` and ``data``.

There is no longer a special ``None`` key. Instead, loading an image or
catalog without an explicit label stores it under a shared sentinel string
(the module-level ``DEFAULT_LABEL`` constant), so repeated unlabeled loads
replace the same "unlabeled" entry rather than accumulating new ones. Unlike
in earlier versions, that default label is **not** hidden from the public
:py:attr:`~astro_image_display_api.image_viewer_logic.ImageViewerLogic.image_labels`
and
:py:attr:`~astro_image_display_api.image_viewer_logic.ImageViewerLogic.catalog_labels`
properties -- once something has been loaded without a label, it shows up in
those tuples just like any other label. If you need to refer to it
explicitly (most callers do not; omitting ``image_label``/``catalog_label``
already resolves to it when nothing else is ambiguous), find it by set
difference against the labels you *did* choose yourself, e.g.
``(set(viewer.image_labels) - {"a", "b"}).pop()``, rather than depending on
the exact sentinel value, which is a private implementation detail.

Label resolution -- implemented by the private ``_resolve_label`` helper
(via ``_resolve_image_label``/``_resolve_catalog_label``) -- follows the
same rule everywhere an ``image_label`` or ``catalog_label`` argument is
accepted:

- If a label is given explicitly to ``load_image``/``load_catalog``, it is
  used as-is, and need not already exist -- loading under a brand-new label
  creates it.
- If no label is given to ``load_image``/``load_catalog``, the shared
  default label above is used.
- For every other method (the getters, ``set_viewport``, ``set_cuts``,
  ``set_stretch``, ``set_colormap``, ``set_catalog_style``,
  ``remove_catalog``), an explicit label must already correspond to loaded
  data, or a ``ValueError`` is raised.
- If no label is given to one of those other methods and nothing is loaded,
  a ``ValueError`` ("No image/catalog is loaded...") is raised, with the
  catalog exceptions listed below.
- If no label is given to one of those other methods and exactly one label
  is loaded -- whether it is the shared default label or one the caller
  chose -- that label is used.
- If no label is given to one of those other methods and more than one
  label exists, a ``ValueError`` is raised asking the caller to
  disambiguate.

Four catalog calls step outside these rules, and ``ImageAPITest``
exercises each of them:

- ``get_catalog()`` with no label on a viewer with no catalogs returns an
  empty table with the requested column names instead of raising.
- ``get_catalog_style()`` with no label on a viewer with no catalogs
  returns the default style, with ``catalog_label`` set to ``None``.
- ``set_catalog_style`` on a viewer with no catalogs raises
  ``ValueError("Must load a catalog before setting a catalog style.")``
  before any label is resolved, whether or not a label was given.
- ``remove_catalog("*")`` removes every catalog. ``"*"`` is handled before
  label resolution and is never treated as a label.

Treat ``_images`` and ``_catalogs`` as private. Read state back through the
public getters (``get_viewport``, ``get_cuts``, ``get_stretch``,
``get_colormap``, ``get_catalog``, ``get_catalog_style``, ``get_image``)
rather than reaching into these dictionaries directly -- the getters apply
the same label resolution and error handling that the rest of the API relies
on.

Override the rendering hooks, not the public methods
-----------------------------------------------------

Every public method of ``ImageViewerLogic`` that changes what is displayed
is a *template*: it validates its arguments, resolves the label, stores the
new state in ``_images`` or ``_catalogs``, and then calls one private
*rendering hook* with the already-resolved label. The hooks are no-ops in
the base class. A backend overrides the hooks, reads the state it needs
back through the public getters (``get_image``, ``get_cuts``,
``get_viewport``, ``get_catalog``, and so on, called with the label the
hook received), and pushes it into its display. It does not override the
public methods themselves.

This split is what makes the base class useful: all of the argument
validation, error messages, label resolution, and state bookkeeping live in
the public methods, so a hook never has to think about any of it. By the
time a hook runs, the label is a real, resolved label (never ``None`` and
never ``"*"``), the state for that label is complete, and any error that
was going to be raised has already been raised.

There are eight hooks. The five ``_apply_*``/``_render_image`` hooks deal
with images, two deal with catalogs, and one is a context manager for
batching.

.. list-table:: The rendering hooks
   :header-rows: 1
   :widths: 22 30 48

   * - Hook
     - Called by
     - What a backend does there
   * - ``_render_image(image_label)``
     - ``load_image``, once, after the new image's data, WCS, viewport,
       cuts, stretch, and colormap have all been stored and
       ``image_label`` has become the displayed image.
     - Get the image with ``get_image(image_label=...)`` and hand its pixel
       values to the plotting library, replacing whatever image was shown
       before. ``get_image`` returns what was loaded, so it may be a plain
       array or an ``NDData``/``CCDData``. With ``image`` as that return
       value, ``np.asarray(getattr(image, "data", image))`` yields an array
       in every case.
   * - ``_apply_cuts(image_label)``
     - ``set_cuts``, and ``load_image`` (once, after ``_render_image``).
       Only when ``image_label`` is the displayed image.
     - Re-scale the displayed pixels using ``get_cuts(image_label=...)``.
   * - ``_apply_stretch(image_label)``
     - ``set_stretch``, and ``load_image`` (once, after ``_apply_cuts``).
       Only when ``image_label`` is the displayed image.
     - Re-scale the displayed pixels using ``get_stretch(image_label=...)``.
   * - ``_apply_colormap(image_label)``
     - ``set_colormap``, and ``load_image`` (once, after
       ``_apply_stretch``). Only when ``image_label`` is the displayed
       image.
     - Apply ``get_colormap(image_label=...)`` to the rendered image. Note
       that this is ``None`` for an image whose colormap was never set.
   * - ``_apply_viewport(image_label)``
     - ``set_viewport``, and ``load_image`` (once, last). Only when
       ``image_label`` is the displayed image.
     - Pan/zoom the display to the center and field of view from
       ``get_viewport(image_label=...)``.
   * - ``_draw_catalog(catalog_label)``
     - ``load_catalog`` and ``set_catalog_style``, after the catalog's
       data and style have been stored.
     - Draw (or redraw) markers at the positions in
       ``get_catalog(catalog_label=...)`` using the shape, color, and size
       from ``get_catalog_style(catalog_label=...)``. The catalog's pixel
       and sky columns have already been filled in from the WCS where
       possible.
   * - ``_remove_catalog_marks(catalog_label)``
     - ``remove_catalog``, after the catalog has been removed from the
       stored state. ``remove_catalog("*")`` is expanded by the base
       class, which calls this hook once per removed catalog inside one
       ``_batch_update`` block, so the hook never sees ``"*"``.
     - Remove the markers drawn for that catalog.
   * - ``_batch_update()``
     - ``load_image`` wraps its state changes and hook calls in it, and
       ``remove_catalog("*")`` wraps its loop over the catalogs in it. The
       base implementation returns ``contextlib.nullcontext()``.
     - Return a context manager that suppresses intermediate redraws until
       the block exits, for backends where each hook call would otherwise
       trigger a repaint or a round trip to a front end. The context
       manager must be exception-safe: ``load_image`` re-raises loader
       errors from inside the block, so release whatever you acquired in a
       ``finally`` clause.

The gating of the ``_apply_*`` hooks is worth spelling out. ``ImageViewerLogic``
keeps track of which image is displayed (today, at most one: the most
recently loaded). Calling ``set_cuts``, ``set_stretch``, ``set_colormap``,
or ``set_viewport`` for any *other* loaded image stores the new value, so
that ``get_cuts`` and friends report it and it is used when that image is
next displayed, but does not call the hook, because there is nothing on
screen to update. A backend therefore never has to check whether the label
it was handed is the one on screen.

Because the hooks are only ever called with the state already stored, the
override does not need to call ``super()``, and it does not need a
docstring: the hooks are private, and the docstring check in
``ImageAPITest`` (see `Docstrings on overridden methods`_) covers only the
public methods of your class.

The worked example below overrides every hook. Its ``_apply_cuts`` is
typical:

.. literalinclude:: ../tests/example_viewer.py
  :language: python
  :pyobject: RecordingViewer._apply_cuts

When to override a public method instead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Occasionally a backend has to step in *before* the base class stores
state, which no hook allows. The astrowidgets ginga backend is the
motivating case: ginga only logs a warning on an unknown colormap name
rather than raising, which would leave the stored colormap disagreeing
with the display. Its ``set_colormap`` override therefore checks the name
against ginga's list, raises ``ValueError`` if it is not there, and only
then calls ``super().set_colormap(...)``. Doing that in ``_apply_colormap``
would not be enough, because that hook never runs for an image that is not
displayed, and the bad name would be stored for later. The other standard
exception is
:py:meth:`~astro_image_display_api.image_viewer_logic.ImageViewerLogic.save`,
which is a public method with no hook behind it; see `Saving the view`_.

When you do override a public method, follow these rules:

- Use the same signature and keyword names as the base method, and forward
  any ``**kwargs`` you receive. ``test_parameter_names_match_interface``
  inspects each method's signature (rather than calling it) to check that
  your parameter names match the interface's, in order. That check exists
  because every method also accepts ``**kwargs``, so a wrongly named
  parameter does not raise a ``TypeError`` when called by keyword -- it
  just silently falls into ``**kwargs`` and is never used.
- Call ``super().<method>(...)`` and let it do the validation, label
  resolution, storage, and hook dispatch. Anything you do before the
  ``super()`` call runs before validation, so it sees raw, unresolved
  arguments (including a possibly ``None`` label); anything after it runs
  after the hooks have already fired. ``test_all_methods_accept_additional_kwargs``
  calls every method with extra, unrecognized keyword arguments to make
  sure they are silently accepted, which forwarding ``**kwargs`` to
  ``super()`` takes care of.
- Give the override its own docstring, or use one of the helpers in
  `Docstrings on overridden methods`_, since it is a public method.
- Be aware that ``load_image`` calls ``set_viewport``, ``set_cuts``, and
  ``set_stretch`` through ``self`` while it is initializing a new image, so
  an override of any of those will run part way through a load, before the
  new image is marked as displayed. That is one more reason to put display
  work in the hooks, which ``load_image`` calls directly at the end.

Every accessor -- ``get_image``, ``get_viewport``, ``get_cuts``,
``get_stretch``, ``get_colormap``, ``get_catalog``, ``get_catalog_style``,
and the ``image_labels``/``catalog_labels`` properties -- is inherited
unchanged. Override one only if you need to change what state it returns
(the astrowidgets ginga backend overrides ``get_viewport`` to report the
view the user has panned or zoomed to), never in order to draw something.

What happens during ``load_image``
------------------------------------

Calling ``load_image(data, image_label=...)`` does the following, in order.
The image label is resolved first; every state change and hook call after
that happens inside a single ``_batch_update`` block.

#. The image label is resolved (a load accepts a brand-new label, or falls
   back to the shared default label described in `The state model`_
   above). This happens before the batch opens, as does noting the label's
   existing entry, if any, and which image is currently displayed, so that
   both can be put back if the load fails.
#. The ``_batch_update`` block opens. The viewer is temporarily marked as
   displaying nothing, and the entry for the label is replaced with a
   fresh, empty one.
#. A format-specific loader (for FITS, arrays, or ``NDData``) stores the
   new data and WCS, then establishes the default viewport (centered, whole
   image visible), cuts (``AsymmetricPercentileInterval(1, 95)``), and
   stretch (``LinearStretch``) for the new image. Because nothing is marked
   as displayed at this point, none of the ``_apply_*`` hooks fire during
   this step.
#. If the loader raised an exception, the label's previous entry (or its
   absence, if the label was new) and the previous displayed-image tracking
   are put back before the exception propagates. The viewer keeps showing
   whatever it showed before the failed load, the ``_apply_*`` hooks keep
   firing for that image, and no hook is called for the failed load. Only
   the loader in the previous step is covered by this rollback; the
   exception then leaves the ``_batch_update`` block, which is why that
   context manager has to release what it acquired in a ``finally`` clause.
#. If the label already existed before this load, its previous cuts,
   stretch, and colormap are restored now, replacing the default cuts and
   stretch the loader just established. Reloading data under an existing
   label keeps that label's display settings; only its data, WCS, and
   viewport come from the new image. A label that did not exist before keeps
   the defaults.
#. The label is marked as the (only) displayed image, and the hooks are
   called once each, with the resolved label, in this order:
   ``_render_image``, ``_apply_cuts``, ``_apply_stretch``,
   ``_apply_colormap``, ``_apply_viewport``.
#. The ``_batch_update`` block exits.

So, from a backend's point of view, a successful load is simply: the batch
opens, the five image hooks fire once each with complete state, the batch
closes. A failed load opens the batch and leaves it by way of the raised
exception, without calling any image hook. Apart from closing the batch on
that path, there is nothing to guard against, and nothing the backend has to
do in ``load_image`` itself. The worked example's log after loading one
image under the label ``"a"`` reads, in order, ``batch_enter``,
``render_image``, ``apply_cuts``, ``apply_stretch``, ``apply_colormap``,
``apply_viewport``, ``batch_exit``, with every hook entry carrying
``image_label="a"``; the tests in `Wiring up the tests`_ pin that sequence
down, and check that a failed load still ends with ``batch_exit``.

The one thing to keep in mind is which state each hook can rely on.
``_render_image`` runs first, so a backend that creates a plot artist or
canvas there can count on it existing in the four ``_apply_*`` hooks that
follow. Conversely, ``_apply_viewport`` runs last, so a backend whose
plotting library resets the view when new data is set (as many do) does not
have to worry about its viewport being clobbered after the fact.

Constructing your subclass
-----------------------------

``ImageViewerLogic`` is a ``dataclass``. Its generated ``__init__`` calls
``__post_init__``, which is where the ``_images`` and ``_catalogs``
dictionaries described above get created. Your subclass's constructor needs
to run that setup too, and it must work with **no arguments**, because
``ImageAPITest`` instantiates your class with ``image_widget_class()``. You
have a few options:

- Override ``__post_init__``, call ``super().__post_init__()`` first, and do
  your own setup (such as creating an empty call log or widget) afterward.
  This is the recommended approach, since it keeps the generated
  dataclass ``__init__`` and does not require you to repeat any of its
  argument handling.

  .. literalinclude:: ../tests/example_viewer.py
    :language: python
    :pyobject: RecordingViewer.__post_init__

- Write an explicit ``__init__`` that calls ``super().__init__()`` first,
  then does your own setup. You would only need this if you want
  constructor arguments beyond what the dataclass fields provide.
- If your viewer also needs to inherit from a widget base class (for
  example a Qt or Jupyter widget class), mixing that base class's ``__init__``
  with a dataclass-generated one usually does not work cleanly. In that case
  write an explicit ``__init__`` that calls both base class constructors
  yourself, rather than relying on ``super()`` chaining.

Docstrings on overridden methods
------------------------------------

This section only matters if you override a *public* method such as
``save`` or ``set_colormap``. The rendering hooks are private, and
``test_every_method_attribute_has_docstring`` does not look at them.

``ImageViewerLogic``'s methods get their docstrings copied over from
``ImageViewerInterface`` by an internal ``docs_from_interface`` decorator.
That decorator only rewrites the docstrings of names that are already
present in ``ImageViewerLogic.__dict__`` at class-definition time -- it does
not run again for subclasses. If you override a method and do not give it
its own docstring, the override ends up with no docstring at all, which
makes ``test_every_method_attribute_has_docstring`` fail. The test looks
attributes up on an *instance*, so it cannot see the docstring of an
overridden property such as ``image_labels``; an undocumented property
override passes silently. Give it a docstring anyway, or use the decorator
below, which does handle properties.

Three ways to fix this:

- Write a docstring on the override directly. This is the simplest option,
  and it is what the worked example below does for its ``save`` override.
- Apply
  :py:func:`~astro_image_display_api.image_viewer_logic.docs_from_image_viewer_logic_if_missing`
  to your own subclass. Unlike ``docs_from_interface``, which is an internal
  implementation detail, this helper *is* in
  ``astro_image_display_api.image_viewer_logic.__all__`` and is meant to be
  used this way: it fills in the docstring of any public method or property
  on your class that lacks its own, from the same-named attribute on
  ``ImageViewerLogic``.
- Assign ``__doc__`` on your override explicitly from the base method, e.g.
  ``MyViewer.load_image.__doc__ = ImageViewerLogic.load_image.__doc__``.

Saving the view
-------------------

There is no rendering hook behind
:py:meth:`~astro_image_display_api.image_viewer_logic.ImageViewerLogic.save`,
so it is the one method every backend overrides directly, and, unlike other
public-method overrides, it should generally **not** call
``super().save(...)``. The base implementation just writes a placeholder
text file -- it has nothing useful for a real viewer to reuse. Instead, your
override should do the whole job itself: render the current view to
``filename`` (with the output format determined by the file's suffix), and
raise ``FileExistsError`` unless ``overwrite=True`` is given.

The test suite loads an image, calls
:py:meth:`~astro_image_display_api.image_viewer_logic.ImageViewerLogic.save`,
and checks only that a file appears and that the overwrite behavior above
holds; it does not inspect the file's contents or format.

.. literalinclude:: ../tests/example_viewer.py
  :language: python
  :pyobject: RecordingViewer.save

Error messages are part of the contract
-------------------------------------------

:py:class:`~astro_image_display_api.api_test.ImageAPITest` checks not just
that the right *type* of exception is raised, but that its message matches a
specific pattern, using ``pytest.raises(..., match=...)``. If you call
``super().<method>(...)`` first, as recommended above, you get all of these
for free. If you replace the base class's validation entirely, your errors
need to match the same patterns.

.. list-table:: Errors ``ImageAPITest`` checks for
   :header-rows: 1
   :widths: 35 15 50

   * - Message pattern (regex)
     - Exception
     - Raised when
   * - ``[Ii]mage label.*not found``
     - ``ValueError``
     - An ``image_label`` is given that does not correspond to a loaded
       image (accessors, ``set_viewport``, ``set_cuts``, ``set_stretch``,
       ``set_colormap``, ``get_image``).
   * - ``(?i)catalog label.*not found``
     - ``ValueError``
     - A ``catalog_label`` is given that does not correspond to a loaded
       catalog (``get_catalog``, ``get_catalog_style``, and, when at least
       one catalog is loaded, ``set_catalog_style``).
   * - the missing label itself
     - ``ValueError``
     - ``remove_catalog`` is given a ``catalog_label`` that is not loaded.
       The suite only requires the message to contain the label; the base
       class uses the same "not found" message as the row above.
   * - ``[Nn]o image``
     - ``ValueError``
     - No image is loaded and no ``image_label`` is given (any image
       accessor, or ``set_viewport``, ``set_cuts``, ``set_stretch``,
       ``set_colormap``). With an explicit label the "not found" message
       above is raised instead, even on an empty viewer.
   * - ``[Nn]o catalog``
     - ``ValueError``
     - ``remove_catalog`` is called with no ``catalog_label`` and no
       catalog is loaded at all.
   * - ``Multiple image labels defined``
     - ``ValueError``
     - No ``image_label`` is given and more than one image is loaded. Also
       raised by ``load_catalog`` when a pixel/sky conversion is required
       and several images are loaded with none of them the displayed one.
   * - ``Multiple catalog labels defined``
     - ``ValueError``
     - No ``catalog_label`` is given and more than one catalog is loaded.
   * - ``[Ii]nvalid value for fov``
     - ``TypeError``
     - ``fov`` is not a float or an angular ``Quantity``.
   * - ``[Ii]ncorrect unit for fov``
     - ``astropy.units.UnitTypeError``
     - ``fov`` is a ``Quantity`` without an angular unit.
   * - ``[Ii]nvalid value for center``
     - ``TypeError``
     - ``center`` is not a ``SkyCoord`` or a tuple.
   * - ``Center must be a tuple``
     - ``TypeError``
     - A ``SkyCoord`` center is given for an image that has no WCS and
       whose current center is a tuple.
   * - ``FOV must be a float``
     - ``TypeError``
     - A ``Quantity`` fov is given for an image that has no WCS and whose
       current fov is a plain number.
   * - ``WCS is not set``
     - ``ValueError``
     - ``get_viewport`` is asked to convert between sky and pixel
       coordinates for an image with no WCS.
   * - ``[Ss]ky_or_pixel must be``
     - ``ValueError``
     - ``get_viewport``'s ``sky_or_pixel`` argument is not ``'sky'``,
       ``'pixel'``, or ``None``.
   * - ``Must load a catalog before setting a catalog style``
     - ``ValueError``
     - ``set_catalog_style`` is called when no catalog at all has been
       loaded yet.
   * - ``Cannot use pixel coordinates without pixel columns``
     - ``ValueError``
     - ``load_catalog`` is called with ``use_skycoord=False`` (the default)
       on a table with no x/y columns, and they cannot be computed either
       (no sky coordinate column, or no WCS to convert one with).
   * - ``Cannot use sky coordinates without``
     - ``ValueError``
     - ``load_catalog`` is called with ``use_skycoord=True`` but the table
       has no sky coordinate column and no WCS is loaded to compute one
       from the pixel columns.
   * - ``Cannot remove multiple catalogs from a list``
     - ``TypeError``
     - ``remove_catalog`` is given a list instead of a single label or
       ``'*'``.
   * - ``Stretch.*not valid.*``
     - ``TypeError``
     - ``set_stretch`` is given something that is not a
       ``BaseStretch``.
   * - ``[mM]ust be`` (cuts)
     - ``TypeError``
     - ``set_cuts`` is given something that is not a 2-tuple or a
       ``BaseInterval``.
   * - ``not a valid`` (colormap)
     - ``ValueError``
     - ``set_colormap`` is given a name that is not a valid Matplotlib
       colormap name (only checked when Matplotlib is installed).
   * - N/A (checked via file existence, not a message)
     - ``FileExistsError``
     - ``save`` is called for a file that already exists and
       ``overwrite=False``.

Worked example
------------------

The full example used throughout this page is reproduced below. It
subclasses ``ImageViewerLogic``, overrides every rendering hook and,
instead of drawing anything, records each display operation it is asked to
perform -- which is enough to write tests and documentation examples that
assert on *what* would have been drawn and in what order, without an actual
display backend. The only public method it overrides is ``save``.

.. dropdown:: tests/example_viewer.py

  .. literalinclude:: ../tests/example_viewer.py
    :language: python

Wiring up the tests
-----------------------

Testing a subclass of ``ImageViewerLogic`` works exactly the same way as
testing any other implementation of
:py:class:`~astro_image_display_api.interface_definition.ImageViewerInterface`;
see :ref:`testing_AIDA_implementation` for the general pattern of
subclassing :py:class:`~astro_image_display_api.api_test.ImageAPITest` and
setting ``image_widget_class``. Below is that pattern applied to the worked
example above, plus three extra tests: one pins down the hook order
described in `What happens during load_image`_, one checks that a failed
load still leaves the ``_batch_update`` block (``batch_enter`` followed by
``batch_exit``, with no hook in between), and one checks that the
``_apply_*`` hooks fire only for the displayed image. The full set of hook
behavior tests lives in ``tests/test_image_viewer_logic_implementation.py``
in the source tree.

.. literalinclude:: ../tests/test_example_viewer.py
  :language: python

When not to use ``ImageViewerLogic``
----------------------------------------

Subclassing ``ImageViewerLogic`` is a convenience, not a requirement, and it
is not always the right fit:

- If your viewer already keeps track of per-image or per-catalog state
  itself (for example, because it wraps a plotting library that has its own
  notion of loaded images and layers), subclassing ``ImageViewerLogic`` on
  top of that gives you two, potentially inconsistent, copies of the same
  bookkeeping.
- If your viewer needs to subclass a widget framework's base class, that
  base class's metaclass or constructor requirements may conflict with
  ``ImageViewerLogic`` being a ``dataclass``; see `Constructing your subclass`_
  above.

In either case, two alternatives are available:

- Implement :py:class:`~astro_image_display_api.interface_definition.ImageViewerInterface`
  directly, without subclassing ``ImageViewerLogic`` at all. Because the
  interface is a ``typing.Protocol`` decorated with ``runtime_checkable``,
  ``isinstance(your_instance, ImageViewerInterface)`` still works as long as
  your class defines the required methods and attributes -- there is no
  need to inherit from anything for that check to pass.
- Use composition instead of inheritance: hold an
  ``ImageViewerLogic`` instance as, e.g., ``self._state``, and delegate to it
  for argument validation and state bookkeeping, while your own class
  handles the actual display and whatever state it needs for that.
