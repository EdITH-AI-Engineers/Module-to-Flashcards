import pytest

from tests.factories import valid_clusters as make_valid_clusters


@pytest.fixture
def valid_clusters():
    return make_valid_clusters()

