import pytest

from peekingduck.pipeline.nodes.model.fairmotv1.fairmot_model import (
    ensure_more_than_zero,
)


class TestFairMOTModel:
    def test_invalid_config_key(self):
        with pytest.raises(TypeError) as excinfo:
            ensure_more_than_zero({}, {})
        assert str(excinfo.value) == "'key' must be either 'str' or 'list'"

        with pytest.raises(TypeError) as excinfo:
            ensure_more_than_zero({}, 123)
        assert str(excinfo.value) == "'key' must be either 'str' or 'list'"
