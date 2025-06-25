# SPDX-FileCopyrightText: 2025-present Matt Craig <mattwcraig@gmail.com>
#
# SPDX-License-Identifier: MIT

try:
    from .version import version as __version__
except ImportError:
    __version__ = ""

from .api_test import *  # noqa: F403
from .interface_definition import *  # noqa: F403
