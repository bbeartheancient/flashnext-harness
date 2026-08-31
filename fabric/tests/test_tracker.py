import pytest

from fabric.tracker import parse_sections


@pytest.fixture(scope="module")
def parsed():
    return parse_sections()


def test_all_sections_found(parsed):
    assert set(parsed["sections"]) == {
        "Device registry",
        "Department catalog",
        "Status summary",
    }


def test_registry_rows(parsed):
    reg = parsed["sections"]["Device registry"]
    assert reg["columns"][0] == "ID"
    ids = [r["cells"][0] for r in reg["rows"]]
    assert len(ids) >= 18 and ids[0] == "D01"


def test_status_rows_have_stages(parsed):
    st = parsed["sections"]["Status summary"]
    assert len(st["rows"]) >= 8
    assert all(r["stage"] for r in st["rows"])


def test_cells_cleaned(parsed):
    blob = str(parsed["sections"])
    assert "**" not in blob and "`" not in blob


def test_mtime_format(parsed):
    assert parsed["mtime"][:4].isdigit()
