#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Adapted from huggingface/open-r1: https://github.com/huggingface/open-r1

from transformers.utils.import_utils import _is_package_available


# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available("e2b")


def is_e2b_available() -> bool:
    """Check if the e2b package is available."""
    return _e2b_available


_morph_available = _is_package_available("morphcloud")


def is_morph_available() -> bool:
    """Check if the morphcloud package is available."""
    return _morph_available

