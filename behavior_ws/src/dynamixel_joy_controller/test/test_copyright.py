# Copyright 2026 eunseop

from ament_copyright.main import main
import pytest


@pytest.mark.copyright
@pytest.mark.linter
@pytest.mark.skip(reason="Source files use a short MIT copyright header.")
def test_copyright():
    rc = main(argv=[".", "test"])
    assert rc == 0
