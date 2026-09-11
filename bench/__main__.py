"""The bench CLI.

    scripts/wheelbench <command> [args]
    PYTHONPATH=. .venv/bin/python -m bench <command> [args]

The whole workflow, in the order you use it:

    bench new <name> --question "..."     scaffold a paragraph
    bench validate                        is every variant well-formed?
    bench show <name>                     the variant, resolved, with reasons
    bench diff <name>                     what it would change about the master
    bench run <name> --tier smoke         seconds: does it do anything at all?
    bench run <name>                      the full tier: the verdict
    bench replay <name> --since 2026-08-01  re-decide real live days
    bench promote <name> --dry-run        what would change, and what blocks it
    bench promote <name> --verdict "..."  merge the paragraph into the essay

    bench frozen                          the master config, with provenance
    bench registry                        regenerate variants/README.md
    bench tiers                            what the tiers are

Every command that can refuse does so with the reason and the fix, because a
bench that fails obscurely is a bench nobody uses, and a bench nobody uses is
how you end up with thirty untracked harnesses in scratchpad/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from bench import variant as bench_variant


def _die(msg: str, code: int = 2) -> int:
    print(f"\n{msg}\n", file=sys.stderr)
    return code


# ---------------------------------------------------------------------------

def cmd_new(args) -> int:
    name = args.name
    p = bench_variant.VARIANTS_DIR / f"{name}.toml"
    if p.exists():
        return _die(f"{p} already exists. Pick another name, or edit that one.")
    today = dt.date.today().isoformat()
    v = bench_variant.Variant(
        name=name, question=args.question, base=args.base, status="draft",
        created=today, updated=today, touches=list(args.touches or []),
        note=("Scaffolded by `bench new`. Fill in the knobs below, each with the "
              "reason it has that value -- `bench validate` refuses a knob whose "
              "`why` is empty, and so does `bench promote`."))
    if args.branch:
        v.code = {"branch": args.branch, "new_flags": list(args.new_flag or [])}
        for f in (args.new_flag or []):
            v.knobs[f] = bench_variant.Knob(
                name=f, value=True,
                why="TODO: why does this flag have this value? One paragraph, "
                    "with the measurement behind it if there is one.")
    else:
        # Seed each knob with the value it ALREADY has in the base, so a fresh
        # variant is a no-op and `bench diff` starts empty. You then change one
        # number and the diff shows exactly the one thing you are testing --
        # rather than a scaffolded 0.0 that silently becomes the experiment.
        base_now = bench_variant.resolve(args.base).knobs
        defaults = {n: f.default for n, f in bench_variant.wheel_fields().items()}
        for f in (args.knob or []):
            v.knobs[f] = bench_variant.Knob(
                name=f, value=base_now.get(f, defaults.get(f)),
                why=f"TODO: this is `{args.base}`'s current value, unchanged. "
                    f"Change it, then write why -- with the measurement behind "
                    f"it if there is one.")
    v.save(p)
    from bench import fingerprint
    print(f"wrote {fingerprint.rel(p)}")
    print("\nnext:")
    print(f"  $EDITOR {fingerprint.rel(p)}")
    if args.branch:
        print(f"  git checkout -b {args.branch}")
    print(f"  bench validate")
    print(f"  bench run {name} --tier smoke")
    return 0


def cmd_validate(args) -> int:
    names = [args.name] if args.name else sorted(bench_variant.all_variants())
    known = bench_variant.wheel_fields()
    bad = 0
    for n in names:
        v = bench_variant.load(n)
        problems = bench_variant.validate(v, known_fields=known)
        if problems:
            bad += 1
            print(f"\n{n}: {len(problems)} problem(s)")
            for p in problems:
                print(f"  - {p}")
        elif args.verbose:
            print(f"{n}: ok")
    if not bad:
        print(f"{len(names)} variant(s) valid.")
    return 1 if bad else 0


def cmd_show(args) -> int:
    from bench import fingerprint
    v = bench_variant.load(args.name)
    r = bench_variant.resolve(args.name)
    print(f"{v.name}   [{v.status}]   base: {v.base or '(root)'}")
    print(f"lineage: {' -> '.join(r.lineage)}")
    if v.question:
        print(f"\nquestion: {v.question}")
    if v.touches:
        print(f"touches: {', '.join(v.touches)}")
    if v.code:
        print(f"code: branch {v.code.get('branch')} "
              f"new_flags {v.code.get('new_flags')}")
    print(f"config hash: {fingerprint.config_hash(r.knobs)}")
    print(f"\n{'knob':<32}{'value':<18}from")
    print("-" * 78)
    for k in sorted(r.knobs):
        print(f"{k:<32}{str(r.knobs[k]):<18}{r.origin.get(k,'')}")
    if args.why:
        for k in sorted(r.knobs):
            kn = r.provenance.get(k)
            if kn and kn.why.strip():
                print(f"\n### {k} = {r.knobs[k]!r}"
                      + (f"   (since {kn.since})" if kn.since else ""))
                print(kn.why.strip())
    return 0


def cmd_diff(args) -> int:
    from bench import promote as promote_mod
    changed = promote_mod.diff_for(args.name)
    if not changed:
        print(f"`{args.name}` changes nothing about the master config.")
        return 0
    w = max(len(k) for k in changed)
    print(f"`{args.name}` would change {len(changed)} knob(s) in "
          f"variants/frozen.toml:\n")
    for k, (old, new) in sorted(changed.items()):
        print(f"  {k:<{w}}   {old!r}  ->  {new!r}")
    return 0


def cmd_frozen(args) -> int:
    from bench import config as bench_config, lock
    r = bench_variant.resolve(bench_variant.MASTER_VARIANT)
    knobs = bench_config.load_frozen()
    print("THE MASTER CONFIG -- what the live paper bot runs")
    print(f"source: variants/frozen.toml   lock: variants/frozen.lock.json\n")
    print(f"{'knob':<32}{'value':<16}{'since':<12}source")
    print("-" * 84)
    for k in sorted(knobs):
        kn = r.provenance.get(k)
        print(f"{k:<32}{str(knobs[k]):<16}{(kn.since or '') if kn else '':<12}"
              f"{(kn.source or '') if kn else ''}")
    try:
        locked = lock.current()
        if locked != knobs:
            print("\n*** frozen.toml DISAGREES WITH frozen.lock.json ***")
            print("    Someone hand-edited the master config. `bench promote` "
                  "is the only sanctioned way to change it.")
            return 1
    except FileNotFoundError as e:
        print(f"\nwarning: {e}")
    if args.why:
        for k in sorted(knobs):
            kn = r.provenance.get(k)
            if kn and kn.why.strip():
                print(f"\n### {k} = {knobs[k]!r}")
                print(kn.why.strip())
    return 0


def cmd_run(args) -> int:
    from bench import run as run_mod
    try:
        run_mod.run(args.name, tier_name=args.tier, base_override=args.base,
                    save=not args.no_save, notes=args.note or "",
                    check_fills=args.check_fills)
    except (run_mod.RunRefused, FileNotFoundError) as e:
        return _die(str(e))
    return 0


def cmd_replay(args) -> int:
    from bench import replay as replay_mod
    try:
        replay_mod.run(args.name, since=args.since, until=args.until,
                       save=not args.no_save)
    except (replay_mod.ReplayRefused, FileNotFoundError) as e:
        return _die(str(e))
    return 0


def cmd_promote(args) -> int:
    from bench import promote as promote_mod
    res = promote_mod.promote(args.name, verdict=args.verdict or "",
                              dry_run=args.dry_run, skip_tests=args.skip_tests,
                              force=args.force)
    print(res.render())
    return 0 if (res.applied or args.dry_run) else 1


def cmd_blocks(args) -> int:
    from bench import blocks
    probs = blocks.coverage_problems()
    if probs:
        print("CATALOG PROBLEMS -- the block view is not complete:\n")
        for x in probs:
            print(f"  - {x}")
        print()
    if args.against:
        print(blocks.render_diff(args.against, args.name))
    else:
        print(blocks.render(args.name, verbose=args.why))
    return 1 if probs else 0


def cmd_jobs(args) -> int:
    from bench import jobs
    if args.start:
        j = jobs.start(args.start, tier=args.tier or "", note=args.note or "")
        print(f"started {j.id}  (pid {j.pid})")
        print(f"  watch:  scripts/wheelbench jobs")
        print(f"  log:    {j.log_path}")
        return 0
    if args.cancel:
        j = jobs.cancel(args.cancel)
        print(f"{args.cancel}: {j.state if j else 'no such job'}")
        return 0
    if args.clear:
        print(f"cleared {jobs.clear_finished()} finished job(s)")
        return 0
    rows = jobs.all_jobs()
    if not rows:
        print("no jobs. Start one: scripts/wheelbench jobs --start <variant>")
        return 0
    print(f"{'id':<44}{'variant':<22}{'tier':<8}{'state':<11}elapsed")
    print("-" * 96)
    for j in rows:
        print(f"{j.id:<44}{j.variant:<22}{j.tier or '(default)':<8}"
              f"{j.state:<11}{j.elapsed_secs():>6.0f}s"
              + (f"   {j.error[:40]}" if j.error else ""))
    return 0


def cmd_doctor(args) -> int:
    from bench import doctor
    r = doctor.run(quick=args.quick)
    print(r.render())
    return 0 if r.healthy else 1


def cmd_artifact(args) -> int:
    from bench import artifact, fingerprint
    p = artifact.build(args.out)
    kb = p.stat().st_size / 1024
    print(f"wrote {fingerprint.rel(p)}  ({kb:.0f} KB, self-contained)")
    print("\nOpen it directly, or publish it as an Artifact to get a URL you "
          "can read on a phone.")
    return 0


def cmd_salvage(args) -> int:
    from bench import salvage
    for f in salvage.run():
        print(f"  {f}")
    print("\nSalvaged studies are stamped NOT EVIDENCE and carry no config "
          "hash, so `bench promote` refuses them. Re-run one to make it "
          "evidence: `bench run <variant>`.")
    return 0


def cmd_registry(args) -> int:
    from bench import registry
    for f in registry.regenerate():
        print(f"wrote {f}")
    return 0


def cmd_tiers(args) -> int:
    from bench import tiers
    print(f"{'tier':<8}{'window':<26}{'universe':<12}{'evidence':<10}blurb")
    print("-" * 96)
    for name, t in tiers.TIERS.items():
        n = len(t.explicit) if t.explicit else (t.n_tickers or "all")
        print(f"{name:<8}{str(t.start.date()) + ' -> ' + str(t.end.date()):<26}"
              f"{str(n):<12}{'yes' if t.is_evidence else 'NO':<10}{t.blurb}")
    return 0


def cmd_policy(args) -> int:
    from bench import policy
    if args.name:
        print(policy.check(args.name, verdict=args.verdict or "",
                           skip_tests=True).render())
        return 0
    pol = policy.load_policy()
    ev = pol.get("evidence", {})
    print("promotion policy (bench/policy.toml)\n")
    print(f"  required        {', '.join(ev.get('required', []))}")
    print(f"  accept_tiers    {', '.join(ev.get('accept_tiers', []))}")
    print(f"  allow_dirty     {ev.get('allow_dirty_tree')}")
    print(f"  require_fresh   {ev.get('require_fresh')}")
    for key, block in (pol.get("conditional") or {}).items():
        print(f"  conditional[{key}]  when touches "
              f"{block.get('when_touches')} -> require {block.get('require')}")
    th = pol.get("thresholds") or {}
    print(f"  thresholds      "
          f"{th if th else '(empty by design -- owner pinned the bar 2026-08-14)'}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bench", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="scaffold a variant")
    p.add_argument("name")
    p.add_argument("--question", required=True,
                   help="what this paragraph must PROVE")
    p.add_argument("--base", default=bench_variant.MASTER_VARIANT)
    p.add_argument("--touches", nargs="*",
                   help="tags the promotion policy keys conditional gates on "
                        "(e.g. liquidity fills size)")
    p.add_argument("--branch", help="the git branch carrying its engine code")
    p.add_argument("--new-flag", action="append",
                   help="a WheelConfig flag this variant's branch introduces")
    p.add_argument("--knob", action="append", help="an existing knob to set")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("validate", help="check every variant is well-formed")
    p.add_argument("name", nargs="?")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("show", help="a variant, resolved, with provenance")
    p.add_argument("name")
    p.add_argument("--why", action="store_true", help="print every knob's reason")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("diff", help="what a variant would change about master")
    p.add_argument("name")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("frozen", help="the master config, with provenance")
    p.add_argument("--why", action="store_true")
    p.set_defaults(fn=cmd_frozen)

    p = sub.add_parser("run", help="run a variant against its base")
    p.add_argument("name")
    p.add_argument("--tier", help="smoke | mid | live | full (see `bench tiers`)")
    p.add_argument("--base", help="compare against this instead of the "
                                  "variant's declared base")
    p.add_argument("--note", help="free text recorded in the scorecard")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--check-fills", action="store_true",
                   help="check every fill against the chain it filled from: "
                        "order size vs the day's volume and open interest, "
                        "zero-volume strikes, and the spread paid. Writes "
                        "results/bench/<variant>/<tier>/fills_<arm>.csv and "
                        "records the summary on the scorecard.")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("replay", help="re-decide real live days")
    p.add_argument("name")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("promote", help="merge a variant into the master config")
    p.add_argument("name")
    p.add_argument("--verdict", help="the sentence you want to read in six "
                                     "months, recorded beside the knob")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-tests", action="store_true",
                   help="skip the test gate (recorded in the changelog)")
    p.add_argument("--force", action="store_true",
                   help="promote despite failing gates (recorded)")
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("policy", help="show the promotion bar, or check a variant")
    p.add_argument("name", nargs="?")
    p.add_argument("--verdict")
    p.set_defaults(fn=cmd_policy)

    p = sub.add_parser("blocks",
                       help="what the bot actually does, as readable mechanics")
    p.add_argument("name", nargs="?", default=bench_variant.MASTER_VARIANT)
    p.add_argument("--why", action="store_true",
                   help="print each block's full story, not just what it does")
    p.add_argument("--against", metavar="VARIANT",
                   help="show which BLOCKS differ versus this variant")
    p.set_defaults(fn=cmd_blocks)

    p = sub.add_parser("jobs", help="background backtest runs")
    p.add_argument("--start", metavar="VARIANT")
    p.add_argument("--tier")
    p.add_argument("--note")
    p.add_argument("--cancel", metavar="JOB_ID")
    p.add_argument("--clear", action="store_true",
                   help="delete finished job records (scorecards are kept)")
    p.set_defaults(fn=cmd_jobs)

    p = sub.add_parser("doctor",
                       help="did anything go around the bench? (exit 1 if so)")
    p.add_argument("--quick", action="store_true",
                   help="only the two live-config checks; fast enough for a hook")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("artifact",
                       help="build the standalone Blocks page (one HTML file)")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_artifact)

    p = sub.add_parser("salvage",
                       help="rescue verdicts from the gitignored results/ tree")
    p.set_defaults(fn=cmd_salvage)

    p = sub.add_parser("registry", help="regenerate variants/README.md")
    p.set_defaults(fn=cmd_registry)

    p = sub.add_parser("tiers", help="what the tiers are")
    p.set_defaults(fn=cmd_tiers)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except FileNotFoundError as e:
        return _die(str(e))
    except ValueError as e:
        return _die(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
