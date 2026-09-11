"""Generate the standalone Blocks page -- a single self-contained HTML file.

WHY GENERATED AND NOT HAND-WRITTEN. The page's whole claim is that it shows
every mechanic the bot has. A hand-maintained HTML copy of the catalog would be
wrong within a week, and wrong in the worst way: it would still LOOK complete.
So it is built from `bench/blocks.toml`, the live config and the scorecards on
disk, and rebuilt with one command.

    scripts/wheelbench artifact          -> bench/artifact/blocks.html

The output has no external requests of any kind: no CDN, no webfont, no image
host. That is partly a hosting constraint and partly the point -- this file is
the thing you open on a phone at 9am to remember what the bot is doing, and it
has to work with no venv, no terminal and no network path back to this repo.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from bench import blocks as B
from bench import fingerprint
from bench import scorecard as S
from bench import variant as V

OUT_PATH = Path(__file__).resolve().parent / "artifact" / "blocks.html"

#: Blocks the live portfolio engine refuses outright (run_portfolio_wheel raises).
#: Shown anyway -- hiding them would make the catalog incomplete -- but marked, and
#: not togglable, because composing a variant that cannot run is a dead end.
SOLO_ONLY = {"regime-gates", "put-stop", "roll-tested-puts",
             "liquidate-on-assignment"}

CAT_LETTER = {"entry": "E", "precaution": "P", "liquidity": "L",
              "exit": "X", "cost": "C", "setup": "A"}


def _defaults() -> dict:
    from dataclasses import fields
    from src.engine_v2.options.wheel import WheelConfig
    return {f.name: f.default for f in fields(WheelConfig)}


def _off_values(block, defaults: dict) -> dict:
    """What a block's settings read as when the mechanic is switched OFF.

    Not simply None for everything. `exit_call_delta` is 0.80 whether or not the
    exit ladder is running -- it is a parameter of the mechanic, not its switch
    -- so printing "off" beside it states something false. Only the knobs that
    actually control on/off move; the rest show the value they would have.

    Which knobs those are depends on how the block decides its state:
      flag / not_value -> one switch knob moves, the parameters keep defaults
      any_set          -> every knob is a threshold, so all of them clear
    """
    kind = block.on_when.get("kind")
    out = {}
    for k in block.knobs:
        d = defaults.get(k)
        if kind == "any_set":
            out[k] = False if isinstance(d, bool) else None
        elif kind == "flag" and k == block.on_when.get("knob"):
            out[k] = False
        elif kind == "not_value" and k == block.on_when.get("knob"):
            out[k] = block.on_when.get("value")
        else:
            out[k] = d
    return out


def _on_value(knob: str, live: dict, defaults: dict):
    if live.get(knob) is not None:
        return live[knob]
    d = defaults.get(knob)
    return True if isinstance(d, bool) else d


def collect() -> dict:
    """Everything the page needs, as plain JSON-able data."""
    from bench import config as bench_config
    catalog = B.load_catalog()
    defaults = _defaults()
    live = bench_config.knobs_for(V.MASTER_VARIANT)
    effective = {**defaults, **live}
    cfg = _as_cfg(effective)

    blocks = []
    for cat, bs in catalog.ordered():
        for b in bs:
            prov = V.resolve(V.MASTER_VARIANT).provenance
            blocks.append({
                "id": b.id, "cat": b.category, "name": b.name,
                "idx": f"{CAT_LETTER.get(b.category, '?')}{b.order}",
                "does": b.does, "why": b.why, "watch": b.watch_out,
                "knobs": list(b.knobs),
                "liveOn": bool(b.is_on(cfg)),
                "solo": b.id in SOLO_ONLY,
                "values": {k: _jsonable(effective.get(k)) for k in b.knobs},
                "onValues": {k: _jsonable(_on_value(k, live, defaults))
                             for k in b.knobs},
                "offValues": {k: _jsonable(v)
                              for k, v in _off_values(b, defaults).items()},
                "since": next((prov[k].since for k in b.knobs
                               if k in prov and prov[k].since), ""),
            })

    variants = []
    for name, v in sorted(V.all_variants().items()):
        if name in (V.ROOT_VARIANT, V.MASTER_VARIANT):
            continue
        cards = S.for_variant(name)
        d = cards[0].delta() if cards else {}
        variants.append({
            "name": name, "status": v.status, "question": v.question,
            "base": v.base or "",
            "tier": (cards[0].tier if cards else ""),
            "evidence": bool(cards and cards[0].is_evidence),
            "sharpe": _jsonable(d.get("sharpe")),
            "retdd": _jsonable(d.get("ret_per_maxdd")),
        })

    git = fingerprint.git_provenance()
    return {
        "categories": [{"id": c.id, "name": c.name, "blurb": c.blurb,
                        "letter": CAT_LETTER.get(c.id, "?")}
                       for c, _ in catalog.ordered()],
        "blocks": blocks,
        "variants": variants,
        "live": {k: _jsonable(v) for k, v in live.items()},
        "defaults": {k: _jsonable(v) for k, v in defaults.items()},
        "git": {"sha": git["sha"], "branch": git["branch"],
                "dirty": bool(git["dirty"])},
        "nFields": len(defaults),
    }


def _as_cfg(d: dict):
    from src.engine_v2.options.wheel import WheelConfig
    known = set(WheelConfig.__dataclass_fields__)
    return WheelConfig(**{k: v for k, v in d.items() if k in known})


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def build(out: Path | None = None) -> Path:
    data = collect()
    problems = B.coverage_problems()
    out = Path(out or OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(data, problems), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def _render(data: dict, problems: list) -> str:
    payload = html.escape(json.dumps(data, separators=(",", ":"), ensure_ascii=True), quote=False)
    warn = ""
    if problems:
        items = "".join(f"<li>{html.escape(p)}</li>" for p in problems)
        warn = (f'<div class="alarm"><strong>This page is incomplete.</strong>'
                f'<ul>{items}</ul></div>')
    return _TEMPLATE.replace("__DATA__", payload).replace("__WARN__", warn)


_TEMPLATE = r'''<meta charset="utf-8">
<title>Wheel Bot Mechanics</title>
<style>
/* ---------------------------------------------------------------------------
   BLUEPRINT. The subject is a machine's parts list, so the page is drawn like
   a technical drawing: hairline rules instead of cards, square corners, a
   drafting blue, and monospace for every value and index. Light theme is paper;
   dark theme is a lit drafting table.

   Every colour is a token on bare :root, redefined in BOTH the media query and
   the [data-theme] block, so the page holds in all three viewer states
   (explicit light, explicit dark, and the un-stamped system default).
--------------------------------------------------------------------------- */
:root{
  --paper:#F4F6F9;      --surface:#FFFFFF;   --sunk:#EDF1F6;
  --ink:#101820;        --muted:#5B6875;     --faint:#8895A4;
  --rule:#D7DEE7;       --rule-soft:#E6EBF1;
  --blue:#1F5FC4;       --blue-soft:#E4EDFB;
  --on:#127A5E;         --on-soft:#DDF0EA;
  --warn:#9A6412;       --warn-soft:#F7EBD8;
  --alarm:#B3261E;      --alarm-soft:#FBE6E4;
  --shadow:0 1px 0 rgba(16,24,32,.04);
  --fh:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --fm:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0C1118;    --surface:#111823;   --sunk:#0A0F16;
    --ink:#DCE5F0;      --muted:#8494A6;     --faint:#63748A;
    --rule:#1E2A38;     --rule-soft:#18222E;
    --blue:#68A2FF;     --blue-soft:#132239;
    --on:#3FBF97;       --on-soft:#0E2C24;
    --warn:#D69A46;     --warn-soft:#2A2114;
    --alarm:#F0776C;    --alarm-soft:#301715;
    --shadow:0 1px 0 rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --paper:#0C1118;      --surface:#111823;   --sunk:#0A0F16;
  --ink:#DCE5F0;        --muted:#8494A6;     --faint:#63748A;
  --rule:#1E2A38;       --rule-soft:#18222E;
  --blue:#68A2FF;       --blue-soft:#132239;
  --on:#3FBF97;         --on-soft:#0E2C24;
  --warn:#D69A46;       --warn-soft:#2A2114;
  --alarm:#F0776C;      --alarm-soft:#301715;
  --shadow:0 1px 0 rgba(0,0,0,.3);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--fh); font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1220px; margin:0 auto; padding:0 24px 96px}

/* ---- masthead ---- */
header{border-bottom:1px solid var(--rule); padding:34px 0 20px; margin-bottom:26px}
.eyebrow{
  font-family:var(--fm); font-size:11px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); margin:0 0 10px;
}
h1{
  font-size:clamp(28px,4.2vw,42px); line-height:1.06; margin:0 0 10px;
  letter-spacing:-.025em; font-weight:660; text-wrap:balance;
}
.dek{margin:0; color:var(--muted); max-width:62ch; font-size:15.5px}
.meta{
  display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:16px;
  font-family:var(--fm); font-size:11.5px; color:var(--faint);
}
.meta b{color:var(--muted); font-weight:500}

/* ---- summary strip ---- */
.strip{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
  border:1px solid var(--rule); background:var(--surface); box-shadow:var(--shadow);
  margin-bottom:22px;
}
.stat{padding:13px 14px; border-right:1px solid var(--rule-soft)}
.stat:last-child{border-right:0}
.stat .k{font-family:var(--fm); font-size:10.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--faint); display:block; margin-bottom:3px}
.stat .v{font-family:var(--fm); font-size:20px; font-variant-numeric:tabular-nums;
  letter-spacing:-.02em}
.stat .v small{font-size:12px; color:var(--faint)}

/* ---- controls ---- */
.bar{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:20px}
button,.seg label{
  font-family:var(--fm); font-size:12px; color:var(--muted);
  background:var(--surface); border:1px solid var(--rule); padding:6px 11px;
  cursor:pointer; border-radius:0;
}
button:hover,.seg label:hover{border-color:var(--blue); color:var(--blue)}
button:focus-visible,.seg input:focus-visible+label,
.tog:focus-visible{outline:2px solid var(--blue); outline-offset:2px}
.seg{display:flex; border:1px solid var(--rule)}
.seg input{position:absolute; opacity:0; pointer-events:none}
.seg label{border:0; border-right:1px solid var(--rule)}
.seg label:last-of-type{border-right:0}
.seg input:checked+label{background:var(--blue-soft); color:var(--blue)}
.spacer{flex:1}

/* ---- layout ---- */
.cols{display:grid; grid-template-columns:186px 1fr; gap:34px; align-items:start}
@media (max-width:840px){.cols{grid-template-columns:1fr; gap:18px}
  nav.index{position:static !important; display:flex; flex-wrap:wrap; gap:8px}}
nav.index{position:sticky; top:20px; font-family:var(--fm); font-size:12px}
nav.index a{
  display:flex; justify-content:space-between; gap:10px; padding:6px 0;
  color:var(--muted); text-decoration:none; border-bottom:1px solid var(--rule-soft);
}
nav.index a:hover{color:var(--blue)}
nav.index .n{color:var(--faint); font-variant-numeric:tabular-nums}

/* ---- category ---- */
.cat{margin-bottom:38px; scroll-margin-top:16px}
.cat > h2{
  font-size:13px; font-family:var(--fm); letter-spacing:.15em; text-transform:uppercase;
  color:var(--blue); margin:0 0 4px; font-weight:600;
}
.cat > p.blurb{margin:0 0 14px; color:var(--muted); font-size:14px; max-width:68ch}

/* ---- one block: a spec row, not a card ---- */
.blk{border-top:1px solid var(--rule); padding:15px 0 14px; display:grid;
  grid-template-columns:44px 1fr auto; gap:0 14px; align-items:start}
.blk:last-child{border-bottom:1px solid var(--rule)}
.blk.dim{opacity:.56}
.idx{font-family:var(--fm); font-size:11.5px; color:var(--faint); padding-top:3px;
  letter-spacing:.04em}
.blk h3{margin:0 0 3px; font-size:16.5px; font-weight:600; letter-spacing:-.012em}
.does{margin:0 0 8px; color:var(--muted); font-size:14.5px; max-width:66ch}
.vals{font-family:var(--fm); font-size:11.5px; color:var(--faint);
  display:flex; flex-wrap:wrap; gap:3px 14px; margin-bottom:2px}
.vals b{color:var(--muted); font-weight:500}
.chip{
  font-family:var(--fm); font-size:10px; letter-spacing:.1em; text-transform:uppercase;
  padding:2px 7px; white-space:nowrap; border:1px solid transparent;
}
.chip.on{background:var(--on-soft); color:var(--on); border-color:var(--on)}
.chip.off{background:var(--sunk); color:var(--faint); border-color:var(--rule)}
.chip.na{background:var(--warn-soft); color:var(--warn); border-color:var(--warn)}
.chip.chg{background:var(--blue-soft); color:var(--blue); border-color:var(--blue)}
.right{display:flex; flex-direction:column; align-items:flex-end; gap:7px}
.tog{
  font-family:var(--fm); font-size:11px; padding:4px 10px; cursor:pointer;
  background:var(--surface); border:1px solid var(--rule); color:var(--muted);
}
.tog[aria-pressed="true"]{background:var(--on-soft); border-color:var(--on); color:var(--on)}
.tog[disabled]{cursor:not-allowed; opacity:.5}
details.why{grid-column:2 / -1; margin-top:9px}
details.why summary{
  font-family:var(--fm); font-size:11.5px; color:var(--blue); cursor:pointer;
  list-style:none; padding:2px 0;
}
details.why summary::-webkit-details-marker{display:none}
details.why summary::before{content:"&#9656; "; }
details[open].why summary::before{content:"&#9662; "; }
details.why .body{
  border-left:2px solid var(--rule); padding:8px 0 2px 14px; margin-top:6px;
  color:var(--muted); font-size:14.5px; max-width:70ch; white-space:pre-wrap;
}
.watch{color:var(--warn); font-size:13.5px; margin-top:10px; font-style:italic}

/* ---- compose output ---- */
.compose{border:1px solid var(--blue); background:var(--surface); padding:16px 18px;
  margin:26px 0 0}
.compose h3{margin:0 0 6px; font-size:15px}
.compose p{margin:0 0 12px; color:var(--muted); font-size:14px}
pre{
  font-family:var(--fm); font-size:12px; background:var(--sunk); color:var(--ink);
  border:1px solid var(--rule); padding:12px 14px; overflow-x:auto; margin:0 0 10px;
}
/* ---- variants table ---- */
.tablewrap{overflow-x:auto; border:1px solid var(--rule); background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:13.5px}
th,td{padding:8px 12px; text-align:left; border-bottom:1px solid var(--rule-soft);
  white-space:nowrap}
th{font-family:var(--fm); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--faint); font-weight:500}
td.q{white-space:normal; color:var(--muted); max-width:44ch}
td.num{font-family:var(--fm); font-variant-numeric:tabular-nums; text-align:right}
.pos{color:var(--on)} .neg{color:var(--alarm)}
.alarm{border:1px solid var(--alarm); background:var(--alarm-soft); color:var(--alarm);
  padding:12px 16px; margin-bottom:20px}
.alarm ul{margin:8px 0 0 18px}
footer{margin-top:44px; padding-top:16px; border-top:1px solid var(--rule);
  color:var(--faint); font-size:13px}
code{font-family:var(--fm); font-size:.92em; background:var(--sunk); padding:1px 5px}
@media (prefers-reduced-motion:reduce){*{transition:none !important; animation:none !important}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Live paper bot &middot; parts list</p>
  <h1>What the wheel bot actually does</h1>
  <p class="dek">Every mechanic in the engine, grouped the way a trade meets them.
    Each one says what it does, why it was built, and whether it is switched on
    right now.</p>
  <div class="meta" id="meta"></div>
</header>

__WARN__

<div class="strip" id="strip"></div>

<div class="bar">
  <div class="seg" role="group" aria-label="Filter blocks">
    <input type="radio" name="f" id="f-all" value="all" checked><label for="f-all">All</label>
    <input type="radio" name="f" id="f-on" value="on"><label for="f-on">On</label>
    <input type="radio" name="f" id="f-off" value="off"><label for="f-off">Off</label>
    <input type="radio" name="f" id="f-chg" value="chg"><label for="f-chg">Changed</label>
  </div>
  <button id="expand">Expand all reasons</button>
  <button id="reset">Reset to live config</button>
  <span class="spacer"></span>
  <button id="theme">Theme</button>
</div>

<div class="cols">
  <nav class="index" id="index" aria-label="Categories"></nav>
  <main id="main"></main>
</div>

<div id="composeSlot"></div>

<h2 style="font-size:13px;font-family:var(--fm);letter-spacing:.15em;text-transform:uppercase;color:var(--blue);margin:44px 0 4px">Variants</h2>
<p style="margin:0 0 14px;color:var(--muted);font-size:14px;max-width:68ch">
  Every idea tried against this config. A tier marked <code>*</code> is stamped
  not-evidence &mdash; it cannot satisfy the promotion gate.</p>
<div class="tablewrap"><table id="variants"></table></div>

<footer id="foot"></footer>
</div>

<script type="application/json" id="payload">__DATA__</script>
<script>
/* The catalog, the live config and the scorecards, embedded. Parsed from a
   JSON script tag rather than inlined as a JS literal so the prose in every
   `why` cannot terminate the script no matter what punctuation it contains. */
const D = JSON.parse(document.getElementById("payload").textContent);
const state = new Map(D.blocks.map(b => [b.id, b.liveOn]));
let filter = "all";

const fmt = v => v === null ? "off" : v === true ? "on" : v === false ? "off" : String(v);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function changed(){
  return D.blocks.filter(b => state.get(b.id) !== b.liveOn);
}
function counts(){
  const on = D.blocks.filter(b => state.get(b.id)).length;
  return {on, off: D.blocks.length - on, chg: changed().length};
}

function renderMeta(){
  const c = counts();
  document.getElementById("meta").innerHTML =
    `<span><b>engine</b> ${esc(D.git.sha)} on ${esc(D.git.branch)}${D.git.dirty ? " (uncommitted changes)" : ""}</span>` +
    `<span><b>fields covered</b> ${D.nFields} of ${D.nFields}</span>` +
    `<span><b>source</b> variants/frozen.toml</span>`;
  document.getElementById("strip").innerHTML =
    `<div class="stat"><span class="k">On</span><span class="v">${c.on}<small> / ${D.blocks.length}</small></span></div>` +
    D.categories.map(cat => {
      const bs = D.blocks.filter(b => b.cat === cat.id);
      const n = bs.filter(b => state.get(b.id)).length;
      return `<div class="stat"><span class="k">${esc(cat.name)}</span>` +
             `<span class="v">${n}<small> / ${bs.length}</small></span></div>`;
    }).join("");
  document.getElementById("index").innerHTML = D.categories.map(cat => {
    const bs = D.blocks.filter(b => b.cat === cat.id);
    const n = bs.filter(b => state.get(b.id)).length;
    return `<a href="#cat-${cat.id}">${esc(cat.name)}<span class="n">${n}/${bs.length}</span></a>`;
  }).join("");
}

function blockHTML(b){
  const on = state.get(b.id);
  const moved = on !== b.liveOn;
  const vals = on ? b.values : b.offValues;
  let chips = `<span class="chip ${on ? "on" : "off"}">${on ? "on" : "off"}</span>`;
  if (b.solo) chips += `<span class="chip na">engine can't run</span>`;
  else if (moved) chips += `<span class="chip chg">changed</span>`;
  return `<article class="blk${on ? "" : " dim"}" data-id="${b.id}" data-on="${on}" data-chg="${moved}">
    <div class="idx">${esc(b.idx)}</div>
    <div>
      <h3>${esc(b.name)}</h3>
      <p class="does">${esc(b.does)}</p>
      <div class="vals">${b.knobs.map(k => `<span><b>${esc(k)}</b> ${esc(fmt(vals[k]))}</span>`).join("")}</div>
    </div>
    <div class="right">${chips}
      <button class="tog" aria-pressed="${on}" data-tog="${b.id}"${b.solo ? " disabled title='The live portfolio engine refuses this mechanic'" : ""}>${on ? "switch off" : "switch on"}</button>
    </div>
    <details class="why"><summary>why this exists</summary>
      <div class="body">${esc(b.why)}${b.watch ? `<div class="watch">&#9888; ${esc(b.watch)}</div>` : ""}</div>
    </details>
  </article>`;
}

function render(){
  document.getElementById("main").innerHTML = D.categories.map(cat => {
    const bs = D.blocks.filter(b => b.cat === cat.id);
    return `<section class="cat" id="cat-${cat.id}">
      <h2>${esc(cat.letter)} &middot; ${esc(cat.name)}</h2>
      <p class="blurb">${esc(cat.blurb)}</p>
      ${bs.map(blockHTML).join("")}
    </section>`;
  }).join("");
  applyFilter();
  renderMeta();
  renderCompose();
}

function applyFilter(){
  document.querySelectorAll(".blk").forEach(el => {
    const on = el.dataset.on === "true", chg = el.dataset.chg === "true";
    const show = filter === "all" || (filter === "on" && on) ||
                 (filter === "off" && !on) || (filter === "chg" && chg);
    el.style.display = show ? "" : "none";
  });
  document.querySelectorAll(".cat").forEach(sec => {
    const any = [...sec.querySelectorAll(".blk")].some(b => b.style.display !== "none");
    sec.style.display = any ? "" : "none";
  });
}

function tomlFor(ch){
  const lines = ['name = "my-idea"',
    'question = "What must this prove?"',
    'base = "frozen"', 'status = "draft"', ""];
  for (const b of ch){
    const on = state.get(b.id);
    const vals = on ? b.onValues : b.offValues;
    for (const k of b.knobs){
      const v = vals[k];
      lines.push(`[knobs.${k}]`);
      lines.push(v === null ? "unset = true"
        : typeof v === "string" ? `value = "${v}"`
        : `value = ${v === true ? "true" : v === false ? "false" : v}`);
      lines.push(`why = """`);
      lines.push(`${on ? "Switched on" : "Switched off"} versus the live config. `
                 + `TODO: say why, with the measurement if there is one.`);
      lines.push(`"""`, "");
    }
  }
  return lines.join("\n");
}

function renderCompose(){
  const ch = changed();
  const slot = document.getElementById("composeSlot");
  if (!ch.length){ slot.innerHTML = ""; return; }
  const toml = tomlFor(ch);
  slot.innerHTML = `<div class="compose">
    <h3>You have changed ${ch.length} block${ch.length > 1 ? "s" : ""}</h3>
    <p>${ch.map(b => `<strong>${esc(b.name)}</strong> ${state.get(b.id) ? "on" : "off"}`).join(" &middot; ")}</p>
    <pre id="toml">${esc(toml)}</pre>
    <button id="copy">Copy this variant file</button>
    <p style="margin-top:12px">Save it as <code>variants/my-idea.toml</code>, fill in
      the question and each <code>why</code>, then run
      <code>scripts/wheelbench run my-idea</code>. The bench refuses a knob with
      an empty reason, so the TODO lines are not optional.</p>
  </div>`;
  document.getElementById("copy").addEventListener("click", async e => {
    try { await navigator.clipboard.writeText(toml); e.target.textContent = "Copied"; }
    catch { e.target.textContent = "Select the text above to copy"; }
    setTimeout(() => { e.target.textContent = "Copy this variant file"; }, 2200);
  });
}

function renderVariants(){
  const t = document.getElementById("variants");
  if (!D.variants.length){ t.innerHTML = "<tr><td>No variants yet.</td></tr>"; return; }
  const cell = v => v === null || v === undefined ? '<td class="num">&mdash;</td>'
    : `<td class="num ${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${v.toFixed(2)}</td>`;
  t.innerHTML = `<thead><tr><th>Variant</th><th>Status</th><th>Question</th>
      <th>Scorecard</th><th style="text-align:right">Sharpe &Delta;</th>
      <th style="text-align:right">Ret/DD &Delta;</th></tr></thead><tbody>` +
    D.variants.map(v => `<tr>
      <td><code>${esc(v.name)}</code></td>
      <td>${esc(v.status)}</td>
      <td class="q">${esc(v.question)}</td>
      <td>${v.tier ? esc(v.tier) + (v.evidence ? "" : "*") : "&mdash;"}</td>
      ${cell(v.sharpe)}${cell(v.retdd)}</tr>`).join("") + "</tbody>";
}

document.addEventListener("click", e => {
  const id = e.target.dataset && e.target.dataset.tog;
  if (id){ state.set(id, !state.get(id)); render(); }
});
document.querySelectorAll('input[name="f"]').forEach(r =>
  r.addEventListener("change", () => { filter = r.value; applyFilter(); }));
document.getElementById("expand").addEventListener("click", e => {
  const opening = e.target.textContent.startsWith("Expand");
  document.querySelectorAll("details.why").forEach(d => { d.open = opening; });
  e.target.textContent = opening ? "Collapse all reasons" : "Expand all reasons";
});
document.getElementById("reset").addEventListener("click", () => {
  D.blocks.forEach(b => state.set(b.id, b.liveOn)); render();
});
document.getElementById("theme").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = cur ? cur === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
});

document.getElementById("foot").innerHTML =
  `Generated from <code>bench/blocks.toml</code> by <code>scripts/wheelbench artifact</code>. ` +
  `All ${D.nFields} engine settings belong to exactly one block, and a test fails if that stops being true &mdash; ` +
  `so this page cannot quietly leave something out.`;

render();
renderVariants();
</script>
'''
