"""Stages: clips -> (controls) -> items (renders + prompts) -> predictions -> report.

On-disk layout of one prepared set (everything a run needs, nothing else):

    <items_dir>/
      config.json            preset, seed, candidates, conditions, reprs
      clips/*.npz            controlled clips (what the baselines/probe read)
      render/<clip>/...      PNG / GIF / MP4 files
      items.jsonl            one line per (clip, condition, representation)
      predictions/*.jsonl    one file per (model, condition, representation)
"""
from __future__ import annotations

import io
import json
import pathlib
import re
import threading
import zlib
from collections import Counter, defaultdict
from dataclasses import replace as dc_replace

import numpy as np

from motion_sport.conditions import build_view
from motion_sport.controls import PRESETS, apply_controls, median_speed
from motion_sport.prompts import PROMPT_STYLES, build_decision, build_prompt, parse_answer
from motion_sport.render import auto_extent, contact_sheet, render_points, save_gif, save_mp4
from motion_sport.schema import Clip, load_clips
from motion_sport.serialize import view_to_state, view_to_text

# which representations make sense for which condition
VALID = {
    "formation": {"sheet", "text"},
    "motion": {"sheet", "frames", "trails", "text", "gif", "video"},
    "motion_shuffled": {"sheet", "frames", "text", "video"},
    "kinematics": {"sheet", "trails", "text", "gif", "video"},
    "kinematics_solo": {"sheet", "trails", "text", "gif", "video"},
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
            image_size: int = 320, seed: int = 0,
            n_players: int | None = None, player_mode: str | None = None) -> pathlib.Path:
    cfg = PRESETS[preset]
    if n_players and cfg.n_players:
        # N must not exceed the smallest roster in the comparison (10 in basketball)
        cfg = dc_replace(cfg, n_players=n_players)
    if player_mode:
        # "central" keeps the N most central players; "random" a uniform subset, so a
        # sport whose whole roster fits in N is not the only one seen complete
        if player_mode not in ("central", "random"):
            raise ValueError(f"player_mode must be central or random, got {player_mode!r}")
        cfg = dc_replace(cfg, player_mode=player_mode)
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
            state = rng.bit_generator.state  # replayed below for the full-rate video view
            view = build_view(c, cond, rng, k=frames_per_view)
            ext = _extent(np.stack(view.frames), cfg.space)
            for rep in reprs:
                if rep not in VALID[cond]:
                    continue
                rdir = out / "render" / c.clip_id
                rdir.mkdir(parents=True, exist_ok=True)
                files: list[str] = []
                text = None
                shown = view
                if rep == "video":
                    # every frame of the clip at its own rate (5 fps), same random draws
                    # as the sheet (same kinematics layout); shuffled for motion_shuffled
                    vrng = np.random.default_rng()
                    vrng.bit_generator.state = state
                    shown = build_view(c, cond, vrng, k=c.n_frames)
                    vext = _extent(np.stack(shown.frames), cfg.space)
                    files = [save_mp4([render_points(f, extent=vext, size=image_size)
                                       for f in shown.frames], rdir / f"{cond}.mp4", c.fps)]
                elif rep in ("sheet", "frames", "gif"):
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
                pseed = stable_seed(c.clip_id, cond, rep)
                prompt, order = build_prompt(shown, prompt_repr, cands, seed=pseed,
                                             text_payload=text)
                prompt_inf, _ = build_prompt(shown, prompt_repr, cands, seed=pseed,
                                             text_payload=text, style="informed")
                item = {
                    "item_id": f"{c.clip_id}__{cond}__{rep}", "clip_id": c.clip_id,
                    "sport": c.sport, "source": c.source, "match_id": c.match_id,
                    "tags": c.tags, "condition": cond, "repr": rep,
                    "files": [str(pathlib.Path(f).relative_to(out)) for f in files],
                    # "prompt" is the neutral style (the key older items dirs have)
                    "prompt": prompt, "prompt_informed": prompt_inf,
                    "options": order, "frame_order": shown.order,
                }
                if rep == "video":
                    item["video_fps"] = c.fps
                if rep == "text":  # typed-decision models (Jev, Laya) read these fields
                    instr, criteria = build_decision(view, cands, seed=pseed)
                    _, criteria_inf = build_decision(view, cands, seed=pseed, style="informed")
                    item.update(decision_instructions=instr, decision_criteria=criteria,
                                decision_criteria_informed=criteria_inf,
                                state_text=text, state_json=view_to_state(view))
                items.append(item)
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


def _pred_path(items_dir, name: str, condition: str, rep: str, *, prompt_style: str = "neutral",
               replicate: int = 1) -> pathlib.Path:
    """predictions/<model>__<condition>__<repr>[__informed][__r<K>].jsonl (K > 1 only)."""
    safe = name.replace(":", "__").replace("/", "_")
    suffix = ("" if prompt_style == "neutral" else f"__{prompt_style}") + \
        ("" if replicate == 1 else f"__r{replicate}")
    return pathlib.Path(items_dir) / "predictions" / f"{safe}__{condition}__{rep}{suffix}.jsonl"


def interleave(items: list[dict]) -> list[dict]:
    """Round-robin over sports, and over matches within a sport, in a fixed order.

    The key depends on the clip only, never on the condition, so `--limit N` picks
    the *same* clips in every condition (paired contrasts need that) and any prefix
    of the run is balanced by sport instead of being all one sport.
    """
    by_sport: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for it in sorted(items, key=lambda i: stable_seed(i["clip_id"], "order")):
        by_sport[it["sport"]][it["match_id"]].append(it)

    def per_sport(matches: dict[str, list[dict]]) -> list[dict]:
        queues = [matches[m] for m in sorted(matches, key=lambda m: stable_seed(m, "order"))]
        out = []
        while any(queues):
            out += [q.pop(0) for q in queues if q]
        return out

    streams = [per_sport(by_sport[s]) for s in sorted(by_sport)]
    out = []
    while any(streams):
        out += [q.pop(0) for q in streams if q]
    return out


def run_model(items_dir: str, model_id: str, *, condition: str, rep: str,
              limit: int | None = None, max_tokens: int = 1024, temperature: float = 0.0,
              state_format: str = "text", complete=None, decide=None,
              workers: int = 1, replicate: int = 1,
              prompt_style: str = "neutral") -> pathlib.Path:
    """Query one model on every matching item. Resumable: skips done items.

    `replicate` K > 1 is an independent repetition of the same cell: its own file
    (suffix `__r<K>`), so it never reuses another replicate's answers. The items are
    the same in every replicate (`interleave` fixes the order, `limit` the prefix).

    `prompt_style="informed"` sends the prompt that also describes how players move
    in each option (decision models: the informed criteria); its file gets
    `__informed` and every record carries `prompt_style`.

    Chat models (backends/chat.py) get the full prompt and the images. Typed-decision
    models (backends/decision.py: Jev, Laya) get instructions + criteria + a state,
    text or JSON (`state_format`), and only exist for the `text` representation.
    """
    from motion_sport.backends import decision as dec
    from motion_sport.backends.chat import Request
    from motion_sport.backends.chat import complete as default_complete

    complete = complete or default_complete
    decide = decide or dec.decide
    is_decision = dec.is_decision_model(model_id)
    if is_decision and rep != "text":
        raise ValueError(f"{model_id} reads text only: use --repr text")
    if state_format not in ("text", "json"):
        raise ValueError(state_format)
    if replicate < 1:
        raise ValueError(f"replicate must be >= 1, got {replicate}")
    root = pathlib.Path(items_dir)
    cands = json.loads((root / "config.json").read_text())["candidates"]
    items = [i for i in load_items(root) if i["condition"] == condition and i["repr"] == rep]
    if rep == "gif":
        raise ValueError("gif items are for the human study / video models, not chat backends")
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {prompt_style!r}; known: {PROMPT_STYLES}")
    items = interleave(items)
    items = items[:limit] if limit else items
    prompt_key = "prompt" if prompt_style == "neutral" else f"prompt_{prompt_style}"
    criteria_key = "decision_criteria" + ("" if prompt_style == "neutral" else f"_{prompt_style}")
    need = criteria_key if is_decision else prompt_key
    if any(need not in it for it in items):
        raise ValueError(f"{root} has no {need!r} in its items (prepared before the "
                         f"{prompt_style} prompt existed): re-run `motion-sport prepare`")
    name = model_id + ("" if not is_decision or state_format == "text" else "+json")
    path = _pred_path(root, name, condition, rep, prompt_style=prompt_style, replicate=replicate)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("error") or r.get("label"):
                done[r["item_id"]] = r
    lock = threading.Lock()

    def one(n: int, it: dict, fh) -> None:
        try:
            if is_decision:
                state = it["state_text"] if state_format == "text" else it["state_json"]
                parsed = decide(model_id, dec.Decision(state, it["decision_instructions"],
                                                       it[criteria_key]))
                raw = parsed.pop("raw", "")
            else:
                media = [(root / f).read_bytes() for f in it["files"]]
                videos, images = (media, []) if rep == "video" else ([], media)
                from motion_sport.backends.chat import last_usage
                raw = complete(model_id, Request(it[prompt_key], images, max_tokens, temperature,
                                                 videos=videos, video_fps=it.get("video_fps")))
                parsed = parse_answer(raw, cands)
                parsed["usage"] = last_usage()
        except Exception as e:  # noqa: BLE001 — record and move on; resumable
            raw, parsed = "", {"label": None, "probs": {}, "rationale": "",
                               "error": f"{type(e).__name__}: {e}"[:300]}
        rec = {"item_id": it["item_id"], "clip_id": it["clip_id"], "sport": it["sport"],
               "match_id": it["match_id"], "tags": it["tags"], "model": name,
               "condition": condition, "repr": rep, "prompt_style": prompt_style,
               "replicate": replicate,
               "raw": raw[:2000], **parsed}
        with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{n}/{len(items)}] {it['clip_id']} true={it['sport']} "
                  f"pred={parsed['label'] or parsed['error']}", flush=True)

    todo = [(n, it) for n, it in enumerate(items, 1) if it["item_id"] not in done]
    with path.open("w") as fh:
        for r in done.values():
            fh.write(json.dumps(r) + "\n")
        if workers <= 1:
            for n, it in todo:
                one(n, it, fh)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(lambda t: one(t[0], t[1], fh), todo))
    return path


def run_baseline(items_dir: str, features: str, seed: int = 0) -> pathlib.Path:
    from motion_sport.backends.probe import fit_probe_oof
    from motion_sport.baselines import featurize

    root = pathlib.Path(items_dir)
    clips = load_clips(root / "clips")
    X = featurize(clips, features)
    pred, probs = fit_probe_oof(X, [c.sport for c in clips], [c.match_id for c in clips], seed=seed)
    return _write_clip_preds(root, f"baseline:{features}", clips, pred, probs)


def run_learner(items_dir: str, learner: str, condition: str = "motion", seed: int = 0,
                epochs: int = 150) -> pathlib.Path:
    """MiniRocket or DeepSets on the controlled clips, grouped-CV by match.

    Both see the same condition a VLM sees (built with the same seeded view), so
    `report` can compute the same paired contrasts (motion vs shuffled, etc.).
    """
    from motion_sport import learners

    root = pathlib.Path(items_dir)
    clips = load_clips(root / "clips")
    y, groups = [c.sport for c in clips], [c.match_id for c in clips]
    seeds = [stable_seed(c.clip_id, str(seed)) for c in clips]
    if learner == "minirocket":
        X = np.stack([learners.clip_series(c, condition, s) for c, s in zip(clips, seeds)])
        fp = learners.minirocket_fit_predict(X, y, seed=seed)
        rep = "series"
    elif learner == "deepsets":
        cfg = json.loads((root / "config.json").read_text())
        tok = [learners.player_tokens(c, condition, s, k=cfg["frames_per_view"])
               for c, s in zip(clips, seeds)]
        fp = learners.deepsets_fit_predict(tok, y, seed=seed, epochs=epochs)
        rep = "tracks"
    else:
        raise ValueError(f"unknown learner {learner!r}")
    pred, probs = learners.grouped_oof(y, groups, fp)
    return _write_clip_preds(root, learner, clips, pred, probs, condition=condition, rep=rep)


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


CellKey = tuple  # (model, condition, repr, prompt_style)

# Pre-registered primary contrasts (a - b), fixed before the final runs (gap A10).
# Every model's instance of these forms ONE family per report, Holm-corrected; every
# other contrast report computes is secondary: exploratory, raw p only.
# Cells are (condition, repr, prompt_style).
PRIMARY_CONTRASTS: dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]] = {
    "order": (("motion", "sheet", "neutral"), ("motion_shuffled", "sheet", "neutral")),
    "motion_over_shape": (("motion", "sheet", "neutral"), ("formation", "sheet", "neutral")),
    "text_vs_image": (("motion", "text", "neutral"), ("motion", "sheet", "neutral")),
    "prompt": (("motion", "sheet", "informed"), ("motion", "sheet", "neutral")),
}
ALPHA = 0.05


def _file_meta(path: pathlib.Path, rows: list[dict]) -> dict:
    """Cell + replicate of one prediction file; older files lack the newer fields."""
    first = rows[0]
    m = re.search(r"__r(\d+)$", path.stem)
    style = first.get("prompt_style") or ("informed" if "__informed" in path.stem else "neutral")
    return {"model": first.get("model", path.stem), "condition": first.get("condition", "?"),
            "repr": first.get("repr", "?"), "prompt_style": style,
            "replicate": int(first.get("replicate") or (m.group(1) if m else 1))}


def _cell_label(cond: str, rep: str, style: str) -> str:
    return f"{cond}/{rep}" + ("" if style == "neutral" else f"[{style}]")


def _contrast_pairs(cells: dict) -> list[tuple[CellKey, CellKey]]:
    """Every (a, b) pair of cells of one model worth a paired contrast."""
    pairs = []
    for (model, cond, rep, style) in cells:
        if cond == "motion":
            # condition contrasts only within one representation and prompt: text vs
            # another condition's image would mix RQ1/RQ2 with RQ4
            for other in ("motion_shuffled", "formation", "kinematics", "kinematics_solo"):
                pairs.append(((model, cond, rep, style), (model, other, rep, style)))
        if cond == "kinematics":  # the value of collective motion
            pairs.append(((model, cond, rep, style), (model, "kinematics_solo", rep, style)))
        if rep == "text":  # RQ4: same condition, text vs image
            pairs.append(((model, cond, rep, style), (model, cond, "sheet", style)))
        if style != "neutral":  # does telling the model what to look for help?
            pairs.append(((model, cond, rep, style), (model, cond, rep, "neutral")))
    return [(a, b) for a, b in pairs if a in cells and b in cells]


def report(items_dir: str, n_boot: int = 2000, tag: str | None = None) -> dict:
    """Summaries for every prediction file, replicate groups and paired contrasts.

    Robust by design: rows of clips that are not in the current items are ignored
    (e.g. predictions left over from an older `prepare`), and contrasts are only
    computed between cells that exist, so a model with missing cells never breaks it.
    """
    from motion_sport.evaluate import holm, paired_difference, replicate_summary, summarize

    root = pathlib.Path(items_dir)
    cands = json.loads((root / "config.json").read_text())["candidates"]
    known = {i["clip_id"] for i in load_items(root)} if (root / "items.jsonl").exists() else None
    runs, ignored = {}, {"rows_not_in_items": 0, "files": []}
    for p in sorted((root / "predictions").glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if known is not None:
            kept = [r for r in rows if r.get("clip_id") in known]
            if len(kept) < len(rows):
                ignored["rows_not_in_items"] += len(rows) - len(kept)
                ignored["files"].append(p.name)
            rows = kept
        if tag:
            rows = [r for r in rows if tag in r.get("tags", [])]
        if rows:
            runs[p.stem] = {"meta": _file_meta(p, rows), "rows": rows,
                            "summary": summarize(rows, cands, n_boot)}
    # cells: one or more replicates of the same (model, condition, repr, prompt style)
    reps: dict[CellKey, dict[int, list[dict]]] = defaultdict(dict)
    for v in runs.values():
        m = v["meta"]
        reps[(m["model"], m["condition"], m["repr"], m["prompt_style"])][m["replicate"]] = v["rows"]
    cells, replicates = {}, []
    for key, by_rep in reps.items():
        if len(by_rep) == 1:
            cells[key] = next(iter(by_rep.values()))
            continue
        summ = replicate_summary([by_rep[k] for k in sorted(by_rep)], n_boot)
        cells[key] = summ.pop("rows")  # per-item correctness averaged over replicates
        model, cond, rep, style = key
        replicates.append({"model": model, "cell": _cell_label(cond, rep, style),
                           "condition": cond, "repr": rep, "prompt_style": style,
                           "replicates": sorted(by_rep), **summ})
    primary_of = {spec: name for name, spec in PRIMARY_CONTRASTS.items()}
    contrasts = []
    for a, b in _contrast_pairs(cells):
        name = primary_of.get((a[1:], b[1:]))
        contrasts.append({"model": a[0], "a": _cell_label(*a[1:]), "b": _cell_label(*b[1:]),
                          "primary": name is not None, "contrast": name,
                          **paired_difference(cells[a], cells[b], n_boot)})
    primary = [c for c in contrasts if c["primary"]]
    for c, adj in zip(primary, holm([c["p"] for c in primary])):
        c["p_holm"] = adj
        c["significant"] = adj < ALPHA
    return {"candidates": cands, "tag": tag,
            "runs": {k: {**v["meta"], **v["summary"]} for k, v in runs.items()},
            "replicates": replicates, "contrasts": contrasts,
            "primary_contrasts": primary,
            "secondary_contrasts": [c for c in contrasts if not c["primary"]],
            "holm": {"family_size": sum(c["p"] == c["p"] for c in primary), "alpha": ALPHA},
            "ignored": ignored}
