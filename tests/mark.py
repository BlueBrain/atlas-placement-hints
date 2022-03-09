from distutils.spawn import find_executable

import pytest

skip_if_no_ultraliser = pytest.mark.skipif(
    find_executable("ultraVolume2Mesh") is None, reason="ultraVolume2Mesh is not present"
)
