"""Command line entry point: `motion-sport <stage> ...` (or `python -m motion_sport.cli`)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from motion_sport import pipeline


def _ingest(a) -> None:
    from motion_sport import loaders

    kw = dict(clip_seconds=a.clip_seconds, stride_seconds=a.stride_seconds)
    if a.source == "toy":
        clips = loaders.make_toy_clips(a.n_toy, seed=a.seed)
    elif a.source == "metrica":
        clips = loaders.load_metrica_game(a.input, target_fps=a.target_fps, **kw)
    elif a.source == "sportvu":
        clips = loaders.load_sportvu_game(a.input, target_fps=a.target_fps, **kw)
    elif a.source == "nfl":
        clips = loaders.load_nfl_tracking(a.input, target_fps=a.target_fps,
                                          max_plays=a.max_plays,
                                          trim_start_s=a.trim_start_seconds, **kw)
    elif a.source == "long-csv":
        if not (a.sport and a.fps):
            sys.exit("long-csv needs --sport and --fps")
        clips = loaders.load_long_csv(
            a.input, sport=a.sport, source=a.name or pathlib.Path(a.input).stem, fps=a.fps,
            target_fps=a.target_fps, columns=json.loads(a.columns) if a.columns else None,
            unit_scale=a.unit_scale, tags=a.tags.split(",") if a.tags else None,
            group_extra=a.group_extra.split(",") if a.group_extra else None, **kw)
    else:
        sys.exit(f"unknown source {a.source}")
    n = 0
    for c in clips:
        c.save(a.out)
        n += 1
    print(f"wrote {n} clips to {a.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="motion-sport")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("ingest", help="source files -> clips/*.npz (metres, no ball)")
    g.add_argument("--source", required=True, choices=["toy", "metrica", "sportvu", "nfl", "long-csv"])
    g.add_argument("--input", help="file or directory of the source")
    g.add_argument("--out", required=True)
    g.add_argument("--clip-seconds", type=float, default=4.0)
    g.add_argument("--stride-seconds", type=float, default=4.0)
    g.add_argument("--target-fps", type=float, default=5.0,
                   help="every source is decimated to this rate (fps must not be a cue)")
    g.add_argument("--sport", help="long-csv: sport id")
    g.add_argument("--fps", type=float, help="long-csv: native frame rate")
    g.add_argument("--name", help="long-csv: source name")
    g.add_argument("--columns", help='long-csv: JSON mapping, e.g. {"frame":"frame_id","track":"id"}')
    g.add_argument("--unit-scale", type=float, default=1.0, help="to metres (yards 0.9144, feet 0.3048)")
    g.add_argument("--tags", help="long-csv: comma-separated tags for every clip")
    g.add_argument("--group-extra", help="long-csv: extra columns that split a match into "
                   "continuous streams (e.g. segment, playId)")
    g.add_argument("--trim-start-seconds", type=float, default=0.0,
                   help="nfl: drop the first seconds of every play (mid-play clips)")
    g.add_argument("--max-plays", type=int)
    g.add_argument("--n-toy", type=int, default=20)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=_ingest)

    p = sub.add_parser("prepare", help="controls + renders + prompts -> items dir")
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--preset", default="strict", choices=sorted(pipeline.PRESETS))
    p.add_argument("--conditions", default="formation,motion,motion_shuffled,kinematics,kinematics_solo")
    p.add_argument("--reprs", default="sheet,trails,text")
    p.add_argument("--per-sport", type=int)
    p.add_argument("--candidates", help="extra distractor sports offered to the model")
    p.add_argument("--frames-per-view", type=int, default=8)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--n-players", type=int,
                   help="override the preset's fixed player count (<= smallest roster compared)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=lambda a: print(pipeline.prepare(
        a.clips, a.out, preset=a.preset, conditions=a.conditions.split(","),
        reprs=a.reprs.split(","), per_sport=a.per_sport,
        candidates=a.candidates.split(",") if a.candidates else None,
        frames_per_view=a.frames_per_view, image_size=a.image_size, seed=a.seed,
        n_players=a.n_players)))

    r = sub.add_parser("run", help="query a model: chat (backends/chat.py) or typed-decision "
                                   "(backends/decision.py: jev:, laya:, laya-http:)")
    r.add_argument("--items", required=True)
    r.add_argument("--model", required=True, help="e.g. azure-anthropic:claude-opus-5-5")
    r.add_argument("--condition", required=True)
    r.add_argument("--repr", required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--state-format", default="text", choices=["text", "json"],
                   help="typed-decision models only: send the coordinates as text or JSON")
    r.add_argument("--workers", type=int, default=1, help="concurrent requests")
    r.add_argument("--replicate", type=int, default=1,
                   help="replicate index K: K > 1 is an independent re-run of the same cell "
                        "(own file, suffix __rK; never reuses another replicate's answers)")
    r.set_defaults(func=lambda a: print(pipeline.run_model(
        a.items, a.model, condition=a.condition, rep=a.repr, limit=a.limit,
        max_tokens=a.max_tokens, temperature=a.temperature, state_format=a.state_format,
        workers=a.workers, replicate=a.replicate)))

    b = sub.add_parser("baseline", help="nuisance / tempo / kinematic feature classifiers")
    b.add_argument("--items", required=True)
    b.add_argument("--features", required=True, choices=["nuisance", "tempo", "kinematic"])
    b.set_defaults(func=lambda a: print(pipeline.run_baseline(a.items, a.features)))

    fl = sub.add_parser("fit", help="MiniRocket (needs .[series]) or DeepSets (needs .[probe]), "
                                    "grouped CV by match")
    fl.add_argument("--items", required=True)
    fl.add_argument("--learner", required=True, choices=["minirocket", "deepsets"])
    fl.add_argument("--condition", default="motion")
    fl.add_argument("--seed", type=int, default=0)
    fl.add_argument("--epochs", type=int, default=150, help="deepsets only")
    fl.set_defaults(func=lambda a: print(pipeline.run_learner(
        a.items, a.learner, a.condition, a.seed, a.epochs)))

    pr = sub.add_parser("probe", help="frozen video encoder + linear probe (needs .[probe])")
    pr.add_argument("--items", required=True)
    pr.add_argument("--encoder", default="facebook/vjepa2-vitl-fpc64-256")
    pr.add_argument("--condition", default="motion")
    pr.add_argument("--num-frames", type=int, default=16)
    pr.set_defaults(func=lambda a: print(pipeline.run_probe(
        a.items, a.encoder, a.condition, a.num_frames)))

    rp = sub.add_parser("report", help="accuracy / balanced acc / log-loss with match-clustered CIs")
    rp.add_argument("--items", required=True)
    rp.add_argument("--tag", help="restrict to clips with this tag (e.g. pre_snap, static)")
    rp.add_argument("--n-boot", type=int, default=2000)
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=_report)

    a = ap.parse_args(argv)
    a.func(a)


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _name(model: str, width: int = 34) -> str:
    """Fit a model id in `width` keeping both ends (route and variant suffix)."""
    return model if len(model) <= width else model[:width - 17] + ".." + model[-15:]


def _var(s: dict) -> str:
    """Prompt style + replicate of a run, blank for the default (neutral, replicate 1)."""
    parts = ([] if s.get("prompt_style", "neutral") == "neutral" else [s["prompt_style"][:3]]) + \
        ([] if s.get("replicate", 1) == 1 else [f"r{s['replicate']}"])
    return "/".join(parts)


def _short(sport: str) -> str:
    return sport.replace("american_football", "am_football")[:11]


def _report(a) -> None:
    rep = pipeline.report(a.items, a.n_boot, a.tag)
    if a.json:
        print(json.dumps(rep, indent=2))
        return
    print(f"candidates: {rep['candidates']}  (chance = {1 / len(rep['candidates']):.2f})"
          + (f"  tag={rep['tag']}" if rep["tag"] else ""))
    # acc = accuracy; bal = balanced accuracy; F1 = macro-F1; kappa = Cohen's kappa;
    # pc-acc = prior-corrected accuracy (response bias removed, leave-one-match-out)
    print(f"{'model':34} {'condition':15} {'repr':8} {'var':6} {'n':>4} {'acc':>5} {'95% CI':>12} "
          f"{'bal':>5} {'F1':>5} {'kappa':>6} {'pc-acc':>6} {'95% CI':>12} {'logloss':>7}")
    for s in rep["runs"].values():
        print(f"{_name(s['model']):34} {s['condition'][:15]:15} {s['repr'][:8]:8} {_var(s):6} {s['n']:4d} "
              f"{s['accuracy']:5.2f} {_fmt_ci(s['accuracy_ci']):>12} {s['balanced_accuracy']:5.2f} "
              f"{s['macro_f1']:5.2f} {s['kappa']:6.2f} {s['prior_corrected_accuracy']:6.2f} "
              f"{_fmt_ci(s['prior_corrected_accuracy_ci']):>12} {s['log_loss']:7.3f}")
    cls = rep["candidates"]
    print("\nper-class recall (95% CI in the JSON output):")
    print(f"{'model':34} {'condition':15} {'repr':8} {'var':6} " + " ".join(f"{_short(c):>11}" for c in cls))
    for s in rep["runs"].values():
        rec = s.get("per_class_recall", {})
        print(f"{_name(s['model']):34} {s['condition'][:15]:15} {s['repr'][:8]:8} {_var(s):6} "
              + " ".join(f"{rec[c]:11.2f}" if c in rec else f"{'-':>11}" for c in cls))
    if rep.get("replicates"):
        print("\nreplicates (clips answered in every replicate; contrasts use the per-item mean):")
        for g in rep["replicates"]:
            print(f"  {_name(g['model']):34} {g['cell']:24} K={g['k']} n={g['n_common']:<4d} "
                  f"acc {g['mean_accuracy']:.2f} {_fmt_ci(g['mean_accuracy_ci'])} "
                  f"SD {g['sd_between_replicates']:.3f}  agreement {g['agreement']:.2f}")
    if rep.get("ignored", {}).get("rows_not_in_items"):
        ig = rep["ignored"]
        print(f"\nignored {ig['rows_not_in_items']} prediction rows of clips not in the current "
              f"items ({len(ig['files'])} files)")
    if rep["contrasts"]:
        print("\npaired contrasts (accuracy difference, match-clustered CI):")
        for c in rep["contrasts"]:
            print(f"  {c['model'][:38]:38} {c['a']} - {c['b']}: {c['diff']:+.2f} "
                  f"{_fmt_ci(c['ci'])}  (n={c['n']})")

if __name__ == "__main__":
    main()
