import re

from api_server import app, health
from version import __version__


def test_release_version_is_semantic_1_1_0():
    assert __version__ == "1.1.0"
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_api_exposes_release_version():
    assert app.version == __version__
    assert health() == {"status": "ok", "version": __version__}
