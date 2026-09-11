"""The standalone Blocks page.

The page is read with no venv, no terminal and possibly no network path back to
this repo, so the properties worth testing are the ones that would make it fail
silently in that setting: a missing block, an external request that a strict CSP
will drop, a colour that only exists in one theme, or a claim about the bot that
is no longer true.
"""
from __future__ import annotations

import html
import json
import re

import pytest

from bench import artifact, blocks as B


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("artifact") / "blocks.html"
    return artifact.build(out).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def payload(page) -> dict:
    m = re.search(r'id="payload">(.*?)</script>', page, re.S)
    assert m, "the embedded data payload is missing"
    return json.loads(html.unescape(m.group(1)))


# -- completeness ------------------------------------------------------------

def test_every_block_is_on_the_page(payload):
    assert {b["id"] for b in payload["blocks"]} == \
           {b.id for b in B.load_catalog().blocks}


def test_every_engine_field_is_represented(payload):
    from bench import variant as V
    claimed = {k for b in payload["blocks"] for k in b["knobs"]}
    assert claimed == set(V.wheel_fields())
    assert payload["nFields"] == len(V.wheel_fields())


def test_the_live_on_state_matches_the_master_config(payload):
    from bench import blocks
    live = {s.block.id: s.on for s in blocks.state_for("frozen")}
    assert {b["id"]: b["liveOn"] for b in payload["blocks"]} == live


def test_a_broken_catalog_puts_a_warning_on_the_page(tmp_path, monkeypatch):
    """The page must never look complete while being incomplete."""
    monkeypatch.setattr(B, "coverage_problems",
                        lambda *a, **k: ["field `put_delta` belongs to NO block"])
    text = artifact.build(tmp_path / "b.html").read_text()
    assert "This page is incomplete" in text
    assert "put_delta" in text


# -- self-containment --------------------------------------------------------

def test_no_external_requests(page):
    """A strict CSP drops every one of these, and the failure is silent."""
    for probe in ("http://", "https://", 'src="//', "@import", "cdn."):
        assert probe not in page, f"external reference: {probe}"


def test_the_payload_is_parsed_not_inlined(page):
    """Prose containing a quote or a </script> would end the script if the data
    were an inline JS literal."""
    i, j = page.find('id="payload"'), page.find("const D = JSON.parse")
    assert 0 <= i < j, "the payload must appear before the code that reads it"


def test_the_page_is_pure_ascii(page):
    """It is served from a filesystem, a static server and an artifact host,
    and only one of those reliably declares a charset. Entities render the same
    under all three."""
    bad = {c for c in page if ord(c) > 127}
    assert not bad, f"non-ASCII characters would mojibake: {sorted(bad)[:10]}"


def test_charset_is_declared_first(page):
    assert page.lstrip().startswith('<meta charset="utf-8">')


# -- theming -----------------------------------------------------------------

def test_all_three_viewer_theme_states_are_covered(page):
    css = page[page.find("<style>"):page.find("</style>")]
    assert ":root{" in css
    assert '@media (prefers-color-scheme:dark)' in css
    assert ':root:not([data-theme="light"])' in css
    assert ':root[data-theme="dark"]' in css


def test_the_body_paints_its_own_background(page):
    """A transparent body borrows the host's ground and renders one theme's
    text on the other theme's surface."""
    css = page[page.find("<style>"):page.find("</style>")]
    body = re.search(r"body\{([^}]*)\}", css)
    assert body and "background:var(--" in body.group(1)


def test_wide_content_scrolls_inside_its_own_container(page):
    css = page[page.find("<style>"):page.find("</style>")]
    assert "overflow-x:auto" in css


# -- honesty about the values ------------------------------------------------

def test_a_parameter_knob_keeps_its_value_when_its_block_is_off(payload):
    """`exit_call_delta` is 0.80 whether or not the exit ladder runs. Printing
    "off" beside it states something false about the engine."""
    x = [b for b in payload["blocks"] if b["id"] == "assigned-stock-exit"][0]
    assert x["offValues"]["assignment_exit"] is False       # the switch moves
    assert x["offValues"]["exit_call_delta"] == 0.8         # the parameter does not
    assert x["offValues"]["exit_stop"] == "spot_below_strike"


def test_a_threshold_block_clears_every_knob_when_off(payload):
    """For an any_set block every knob IS a switch, so all of them clear."""
    liq = [b for b in payload["blocks"] if b["id"] == "liquidity-floors"][0]
    assert set(liq["offValues"].values()) == {None}


def test_the_ranking_block_switches_back_to_the_live_sort_key(payload):
    r = [b for b in payload["blocks"] if b["id"] == "ranking"][0]
    assert r["offValues"]["rank_by"] == "vol_pctile"


def test_solo_only_blocks_are_marked_unrunnable(payload):
    solo = {b["id"] for b in payload["blocks"] if b["solo"]}
    assert solo == {"regime-gates", "put-stop", "roll-tested-puts",
                    "liquidate-on-assignment"}


def test_the_page_records_which_engine_built_it(payload):
    assert len(payload["git"]["sha"]) == 12
    assert isinstance(payload["git"]["dirty"], bool)


def test_scorecard_deltas_ride_along_with_the_variants(payload):
    names = {v["name"] for v in payload["variants"]}
    assert "iv-ranker" in names
    iv = [v for v in payload["variants"] if v["name"] == "iv-ranker"][0]
    assert iv["evidence"] is False, "salvaged runs are never evidence"


def test_the_title_is_a_name_not_a_sentence(page):
    m = re.search(r"<title>(.*?)</title>", page)
    assert m and 1 <= len(m.group(1).split()) <= 5
    assert ":" not in m.group(1) and " - " not in m.group(1)
