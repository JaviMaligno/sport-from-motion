"""Stages: clips -> (controls) -> items (renders + prompts) -> predictions -> report.

On-disk layout of one prepared set (everything a run needs, nothing else):

    <items_dir>/
      config.json            preset, seed, candidates, conditions, reprs
      clips/*.npz            controlled clips (what the baselines/probe read)
      render/<clip>/...      PNG / GIF files
      items.jsonl            one line per (clip, condition, representation)
      predictions/*.jsonl    one file per (model, condition, representation)
"""
from __future__ import annotations

import io
import json
import pathlib
import zlib
from collections import Counter, defaultdict
from dataclasses import replace as dc_replace

import numpy as np

from motion_sport.conditions import build_view
from motion_sport.controls import PRESETS, apply_controls, median_speed
from motion_sport.prompts import build_prompt, parse_answer
from motion_sport.render import auto_extent, contact_sheet, render_points, save_gif
from motion_sport.schema import Clip, load_clips
from motion_sport.serialize import view_to_text

# which representations make sense for which condition
VALID = {
    "formation": {"sheet", "text"},
    "motion": {"sheet", "frames", "trails", "text", "gif"},
    "motion_shuffled": {"sheet", "frames", "text"},
    "kinematics": {"sheet", "trails", "text", "gif"},
    "kinematics_solo": {"sheet", "trails", "text", "gif"},
}
STATIC_SPEED_MS = 1.0  # median player speed (m/s) under which a clip is tagged "static"


def stable_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode())


def balanced_sample(clips: list[Clip], per_sport: int | None, seed: int) -> list[Clip]:
    """Up to `per_sport` clips per sport, spread round-robin across matches."""
    rng = np.random.default_rng(seed)
    by_sport: dict[str, dict[str, list[Clip]]] = defaultdict(lambda: defaultdict(list))
    for c in clips:
        by_sport[c.sport][c.match_id].append(c)
    out = []
    for sport, matches in sorted(by_sport.items()):
        pools = {m: list(rng.permutation(len(cs))) for m, cs in matches.items()}
        chosen: list[Clip] = []
        while pools and (per_sport is None or len(chosen) < per_sport):
            for m in list(pools):
                if not pools[m]:
                    del pools[m]
                    continue
                chosen.append(matches[m][pools[m].pop()])
                if per_sport is not None and len(chosen) >= per_sport:
                    break
        out += chosen
    return out


def _extent(xy: np.ndarray, space: str) -> float:
    return 0.5 if space == "field" else auto_extent(xy)


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def prepare(clips_dir: str, out_dir: str, *, preset: str = "strict",
            conditions: list[str], reprs: list[str], per_sport: int | None = None,
            candidates: list[str] | None = None, frames_per_view: int = 8,
            image_size: int = 320, seed: int = 0) -> pathlib.Path:
    cfg = PRESETS[preset]
    out = pathlib.Path(out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    raw = balanced_sample(load_clips(clips_dir), per_sport, seed)
    if cfg.tempo == "auto":
        cfg = dc_replace(cfg, tempo=resolve_auto_tempo(raw, cfg, seed))
    cands = sorted(set(candidates or []) | {c.sport for c in raw})
    items, kept, kept_by_sport = [], 0, Counter()
    for clip in raw:
        rng = np.random.default_rng(stable_seed(clip.clip_id, str(seed)))
        tags = list(clip.tags)
        if median_speed(clip) < STATIC_SPEED_MS and clip.source != "toy":
            tags.append("static")  # computed in metres, before any rescaling
        c = apply_controls(clip, cfg, rng)
        if c is None:
            continue
        c = c.replace(tags=sorted(set(tags)))
        c.save(out / "clips")
        kept += 1
        kept_by_sport[c.sport] += 1
        for cond in conditions:
            view = build_view(c, cond, rng, k=frames_per_view)
            ext = _extent(np.stack(view.frames), cfg.space)
            for rep in reprs:
                if rep not in VALID[cond]:
                    continue
                rdir = out / "render" / c.clip_id
                rdir.mkdir(parents=True, exist_ok=True)
                files: list[str] = []
                text = None
                if rep in ("sheet", "frames", "gif"):
                    imgs = [render_points(f, extent=ext, size=image_size) for f in view.frames]
                    if rep == "sheet":
                        img = imgs[0] if len(imgs) == 1 else contact_sheet(imgs, cols=4)
                        p = rdir / f"{cond}_sheet.png"
                        img.save(p)
                        files = [p]
                    elif rep == "frames":
                        files = []
                        for i, im in enumerate(imgs):
                            p = rdir / f"{cond}_f{i + 1:02d}.png"
                            im.save(p)
                            files.append(p)
                    else:
                        allf = [render_points(f, extent=ext, size=image_size) for f in c.xy] \
                            if cond == "motion" else imgs
                        files = [save_gif(allf, rdir / f"{cond}.gif", c.fps if cond == "motion" else 2)]
                elif rep == "trails":
                    img = render_points(view.frames[-1], extent=ext, size=image_size,
                                        history=view.frames[:-1])
                    p = rdir / f"{cond}_trails.png"
                    img.save(p)
                    files = [p]
                elif rep == "text":
                    text = view_to_text(view)
                prompt_repr = {"sheet": "image", "frames": "image", "gif": "image"}.get(rep, rep)
                prompt, order = build_prompt(view, prompt_repr, cands,
                                             seed=stable_seed(c.clip_id, cond, rep), text_payload=text)
                items.append({
                    "item_id": f"{c.clip_id}__{cond}__{rep}", "clip_id": c.clip_id,
                    "sport": c.sport, "source": c.source, "match_id": c.match_id,
                    "tags": c.tags, "condition": cond, "repr": rep,
                    "files": [str(pathlib.Path(f).relative_to(out)) for f in files],
                    "prompt": prompt, "options": order, "frame_order": view.order,
                })
    in_by_sport = Counter(c.sport for c in raw)
    for sport, n_in in in_by_sport.items():
        if kept_by_sport[sport] < 0.5 * n_in:
            # Differential attrition is itself a leak: the surviving clips of one
            # sport would be a biased subset (e.g. only its slowest plays).
            print(f"WARNING: {sport}: only {kept_by_sport[sport]}/{n_in} clips survived the "
                  f"controls — check clip length / n_players before trusting results")
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (out / "config.json").write_text(json.dumps({
        "preset": preset, "controls": cfg.as_dict(), "seed": seed, "candidates": cands,
        "conditions": conditions, "reprs": reprs, "per_sport": per_sport,
        "frames_per_view": frames_per_view, "image_size": image_size,
        "n_clips_in": len(raw), "n_clips_kept": kept, "n_items": len(items),
        "clips_in_by_sport": dict(in_by_sport), "clips_kept_by_sport": dict(kept_by_sport),
    }, indent=2))
    return out


def resolve_auto_tempo(clips: list[Clip], cfg, seed: int, percentile: float = 10.0) -> float:
    """A low percentile of the final windows' speeds after the spatial controls.

    Low on purpose: slowing a clip down is always possible, speeding it up needs
    extra source footage. With the median, every clip of an intrinsically faster
    sport had to be sped up and, on short clips, the whole sport was dropped —
    differential attrition, i.e. a new leak.
    """
    pre = dc_replace(cfg, tempo=None, rotate=False)
    speeds = []
    for clip in clips:
        c = apply_controls(clip, pre, np.random.default_rng(stable_seed(clip.clip_id, str(seed))))
        if c is not None:
            speeds.append(median_speed(c))
    if not speeds:
        raise ValueError("no clip survived the spatial controls")
    return float(np.percentile(speeds, percentile))


def load_items(items_dir: str | pathlib.Path) -> list[dict]:
    p = pathlib.Path(items_dir) / "items.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _pred_path(items_dir, name: str, condition: str, rep: str) -> pathlib.Path:
    safe = name.replace(":", "__").replace("/", "_")
    return pathlib.Path(items_dir) / "predictions" / f"{safe}__{condition}__{rep}.jsonl"


def run_model(items_dir: str, model_id: str, *, condition: str, rep: str,
              limit: int | None = None, max_tokens: int = 1024, temperature: float = 0.0,
              complete=None) -> pathlib.Path:
    """Query one chat backend on every matching item. Resumable: skips done items."""
    from motion_sport.backends.chat import Request
    from motion_sport.backends.chat import complete as default_complete

    complete = complete or default_complete
    root = pathlib.Path(items_dir)
    cands = json.loads((root / "config.json").read_text())["candidates"]
    items = [i for i in load_items(root) if i["condition"] == condition and i["repr"] == rep]
    if rep == "gif":
        raise ValueError("gif items are for the human study / video models, not chat backends")
    items = items[:limit] if limit else items
    path = _pred_path(root, model_id, condition, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error") or r.get("label"):
                done[r["item_id"]] = r
    with path.open("w") as fh:
        for r in done.values():
            fh.write(json.dumps(r) + "\n")
        for n, it in enumerate(items, 1):
            if it["item_id"] in done:
                continue
            images = [(root / f).read_bytes() for f in it["files"]]
            try:
                raw = complete(model_id, Request(it["prompt"], images, max_tokens, temperature))
                parsed = parse_answer(raw, cands)
            except Exception as e:  # noqa: BLE001 — record and move on; resumable
                raw, parsed = "", {"label": None, "probs": {}, "rationale": "",
                                   "error": f"{type(e).__name__}: {e}"[:300]}
            rec = {"item_id": it["item_id"], "clip_id": it["clip_id"], "sport": it["sport"],
                   "match_id": it["match_id"], "tags": it["tags"], "model": model_id,
                   "condition": condition, "repr": rep, "raw": raw[:2000], **parsed}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{n}/{len(items)}] {it['clip_id']} true={it['sport']} "
                  f"pred={parsed['label'] or parsed['error']}", flush=True)
    return path


def run_baseline(items_dir: str, features: str, seed: int = 0) -> pathlib.Path:
    from motion_sport.backends.probe import fit_probe_oof
    from motion_sport.baselines import featurize

    root = pathlib.Path(items_dir)
    clips = load_clips(root / "clips")
    X = featurize(clips, features)
    pred, probs = fit_probe_oof(X, [c.sport for c in clips], [c.match_id for c in clips], seed=seed)
    return _write_clip_preds(root, f"baseline:{features}", clips, pred, probs)


def run_probe(items_dir: str, encoder: str, condition: str = "motion", num_frames: int = 16,
              image_size: int = 256, seed: int = 0) -> pathlib.Path:
    """Frozen video encoder on re-rendered clips + grouped-CV logistic probe."""
    from motion_sport.backends.probe import HFVideoEncoder, fit_probe_oof

    root = pathlib.Path(items_dir)
    cfg = json.loads((root / "config.json").read_text())
    clips = load_clips(root / "clips")
    enc = HFVideoEncoder(encoder, num_frames=num_frames)
    feats = []
    for c in clips:
        rng = np.random.default_rng(stable_seed(c.clip_id, str(seed)))
        view = build_view(c, condition, rng, k=max(num_frames, cfg["frames_per_view"]))
        ext = _extent(np.stack(view.frames), cfg["controls"]["space"])
        frames = [np.asarray(render_points(f, extent=ext, size=image_size)) for f in view.frames]
        feats.append(enc.encode(frames))
    X = np.stack(feats)
    np.save(root / f"embeddings_{encoder.replace('/', '_')}_{condition}.npy", X)
    pred, probs = fit_probe_oof(X, [c.sport for c in clips], [c.match_id for c in clips], seed=seed)
    return _write_clip_preds(root, f"probe:{encoder}", clips, pred, probs, condition=condition)


def _write_clip_preds(root, name, clips, pred, probs, condition="all", rep="features"):
    path = _pred_path(root, name, condition, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c, p, pr in zip(clips, pred, probs):
            fh.write(json.dumps({"item_id": c.clip_id, "clip_id": c.clip_id, "sport": c.sport,
                                 "match_id": c.match_id, "tags": c.tags, "model": name,
                                 "condition": condition, "repr": rep, "label": p or None,
                                 "probs": pr, "error": None if p else "no fold"}) + "\n")
    return path


def report(items_dir: str, n_boot: int = 2000, tag: str | None = None) -> dict:
    """Summaries for every prediction file + the two paired contrasts per model."""
    from motion_sport.evaluate import paired_difference, summarize

    root = pathlib.Path(items_dir)
    cands = json.loads((root / "config.json").read_text())["candidates"]
    runs = {}
    for p in sorted((root / "predictions").glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if tag:
            rows = [r for r in rows if tag in r.get("tags", [])]
        if rows:
            runs[p.stem] = {"meta": {k: rows[0][k] for k in ("model", "condition", "repr")},
                            "rows": rows, "summary": summarize(rows, cands, n_boot)}
    contrasts = []
    by_key = {(v["meta"]["model"], v["meta"]["condition"], v["meta"]["repr"]): v["rows"]
              for v in runs.values()}
    for (model, cond, rep), rows in by_key.items():
        if cond != "motion":
            continue
        for other in ("motion_shuffled", "formation", "kinematics", "kinematics_solo"):
            for orep in (rep, "sheet"):
                if (model, other, orep) in by_key:
                    contrasts.append({"model": model, "a": f"motion/{rep}", "b": f"{other}/{orep}",
                                      **paired_difference(rows, by_key[(model, other, orep)], n_boot)})
                    break
    return {"candidates": cands, "tag": tag,
            "runs": {k: {**v["meta"], **v["summary"]} for k, v in runs.items()},
            "contrasts": contrasts}
