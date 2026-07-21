"""Tests for SVG/DXF fabrication export."""

import xml.dom.minidom as minidom

from foldforge.geometry import examples
from foldforge.fabricate import to_svg, to_dxf


def test_svg_is_valid_xml_with_layers(tmp_path):
    p = tmp_path / "m.svg"
    to_svg(examples.miura(), p, size_mm=150)
    doc = minidom.parse(str(p))                       # raises if malformed
    labels = {g.getAttribute("inkscape:label") for g in doc.getElementsByTagName("g")}
    assert {"mountain", "valley", "cut"} <= labels    # fold types on their own layers
    assert "mm" in p.read_text()                       # sized for fabrication


def test_svg_line_count_matches_edges(tmp_path):
    pat = examples.miura()
    p = tmp_path / "m.svg"
    to_svg(pat, p)
    assert p.read_text().count("<line") == pat.n_edges


def test_dxf_has_line_entities_on_layers(tmp_path):
    p = tmp_path / "m.dxf"
    to_dxf(examples.miura(), p)
    text = p.read_text()
    assert text.count("\nLINE") == examples.miura().n_edges
    assert "mountain" in text and "valley" in text and "cut" in text
    assert text.strip().endswith("EOF")
