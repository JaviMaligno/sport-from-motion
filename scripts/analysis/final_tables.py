"""Markdown tables of the final run (docs/preregistration.md sections 5-8 and 10) from the
JSON outputs of `motion-sport report`, `final_extras.py` and `measured_cost.py`.

Section 6 is applied mechanically, per chat model, on the primary report:
- "lee el orden temporal": `order` > 0 with p_holm < 0.05 AND the lower bound of the 95 %
  CI of the prior-corrected accuracy of motion/sheet above chance;
- "le sirven varios instantes, pero no su orden": `motion_over_shape` > 0 with
  p_holm < 0.05, prior-corrected accuracy of motion/sheet above chance (lower CI bound),
  and `order` not significant;
- "sin evidencia de que vea movimiento" otherwise (with the upper CI bound of `order`).
A significant negative contrast is reported as such and never counts.

    PYTHONPATH=src .venv/bin/python scripts/analysis/final_tables.py runs/analysis-final
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

SPECIALIST_PREFIXES = ("baseline:", "minirocket", "deepsets", "probe:")
CONTRAST_ORDER = ["order", "motion_over_shape", "text_vs_image", "prompt"]
SHORT = {"american_football": "AF", "basketball": "BB", "handball": "HB", "soccer": "SO",
         "__none__": "err"}
CELL_ORDER = ["motion/sheet", "motion_shuffled/sheet", "formation/sheet", "motion/text",
              "motion/sheet[informed]", "kinematics/sheet", "kinematics_solo/sheet",
              "motion/trails", "motion/video", "motion_shuffled/video",
              "motion_shuffled/text", "formation/text", "motion/text[informed]",
              "kinematics/text", "kinematics_solo/text"]


def f2(x) -> str:
    return "—" if x is None or x != x else f"{x:.2f}"


def fs(x) -> str:
    return "—" if x is None or x != x else f"{x:+.2f}"


def fp(x) -> str:
    if x is None or x != x:
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def ci(c) -> str:
    return "—" if not c or c[0] != c[0] else f"[{c[0]:.2f}; {c[1]:.2f}]"


def name(m: str) -> str:
    return m.split(":", 1)[-1].replace("~typesafe/", "")


def cell(r: dict) -> str:
    return f"{r['condition']}/{r['repr']}" + ("" if r["prompt_style"] == "neutral"
                                              else f"[{r['prompt_style']}]")


def is_chat(m: str) -> bool:
    return not m.startswith(SPECIALIST_PREFIXES) and not m.startswith("jev")


def runs_by(rep: dict) -> dict[tuple, dict]:
    return {(r["model"], cell(r)): r for r in rep["runs"].values()}


def primary_by(rep: dict) -> dict[tuple, dict]:
    return {(c["model"], c["contrast"]): c for c in rep["primary_contrasts"]}


def classify(rep: dict) -> dict[str, dict]:
    runs, prim = runs_by(rep), primary_by(rep)
    chance = 1 / len(rep["candidates"])
    out = {}
    for m in sorted({c["model"] for c in rep["primary_contrasts"]}):
        ms = runs.get((m, "motion/sheet"))
        pc_lo = ms["prior_corrected_accuracy_ci"][0] if ms else float("nan")
        above = pc_lo > chance
        o, s = prim.get((m, "order")), prim.get((m, "motion_over_shape"))
        o_sig = bool(o and o.get("significant"))
        s_sig = bool(s and s.get("significant"))
        if o_sig and o["diff"] > 0 and above:
            cls = "lee el orden temporal («ve el movimiento»)"
        elif s_sig and s["diff"] > 0 and above and not o_sig:
            cls = "le sirven varios instantes, pero no su orden"
        else:
            cls = "sin evidencia de que vea movimiento"
        neg = [c["contrast"] for (mm, _), c in prim.items()
               if mm == m and c.get("significant") and c["diff"] < 0]
        out[m] = {"class": cls, "order": o, "motion_over_shape": s,
                  "pc_acc": ms["prior_corrected_accuracy"] if ms else None,
                  "pc_ci": ms["prior_corrected_accuracy_ci"] if ms else None,
                  "pc_above_chance": above, "significant_negative": neg,
                  "order_upper": o["ci"][1] if o else None}
    return out


def table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def contrasts_table(rep: dict) -> str:
    rows = []
    for c in sorted(rep["primary_contrasts"],
                    key=lambda c: (c["model"], CONTRAST_ORDER.index(c["contrast"]))):
        flags = ", ".join(f"errores > 2 % en {x}" for x in c.get("flagged_cells", []))
        rows.append([name(c["model"]), c["contrast"], f"{c['a']} − {c['b']}", fs(c["diff"]),
                     ci(c["ci"]), fp(c["p"]), fp(c["p_holm"]), "sí" if c["significant"] else "no",
                     str(c["n"]), str(c["n_matches"]), flags or "—"])
    return table(["modelo", "contraste", "a − b", "d", "IC 95 %", "p", "p_holm", "sig.", "n",
                  "partidos", "marcas"], rows)


def cells_table(rep: dict, models=None) -> str:
    rows = []
    rb = runs_by(rep)
    for (m, c), r in sorted(rb.items(), key=lambda kv: (kv[0][0], CELL_ORDER.index(kv[0][1])
                                                         if kv[0][1] in CELL_ORDER else 99)):
        if models and not models(m):
            continue
        share = " ".join(f"{SHORT.get(k, k)} {v:.2f}" for k, v in r["predicted_share"].items() if v)
        note = f"{r['cell_errors']}/{r['cell_rows']} err" + (" (!)" if r["cell_status"] == "flagged"
                                                             else "") if r["cell_errors"] else ""
        rows.append([name(m), c, str(r["n"]), f2(r["accuracy"]), ci(r["accuracy_ci"]),
                     f2(r["balanced_accuracy"]), f2(r["macro_f1"]), f2(r["kappa"]),
                     f2(r["prior_corrected_accuracy"]), ci(r["prior_corrected_accuracy_ci"]),
                     f"{r['log_loss']:.2f}", share, note or "—"])
    return table(["modelo", "celda", "n", "acc", "IC", "bal", "F1", "κ", "pc-acc", "IC pc",
                  "log-loss", "predicted_share", "errores"], rows)


def recall_table(rep: dict, models=None) -> str:
    cls = rep["candidates"]
    rows = []
    rb = runs_by(rep)
    for (m, c), r in sorted(rb.items(), key=lambda kv: (kv[0][0], CELL_ORDER.index(kv[0][1])
                                                         if kv[0][1] in CELL_ORDER else 99)):
        if models and not models(m):
            continue
        rec, rci = r["per_class_recall"], r["per_class_recall_ci"]
        rows.append([name(m), c] + [f"{rec[k]:.2f} {ci(rci[k])}" if k in rec else "—" for k in cls])
    return table(["modelo", "celda"] + [SHORT[k] for k in cls], rows)


def confusion_table(rep: dict, target: str = "motion/sheet") -> str:
    cls = rep["candidates"]
    rows = []
    for (m, c), r in sorted(runs_by(rep).items()):
        if c != target:
            continue
        for t in cls:
            conf = r["confusion"][t]
            rows.append([name(m), SHORT[t]] + [str(conf.get(p, 0)) for p in cls + ["__none__"]])
    return table(["modelo", "real ↓ / predicho →"] + [SHORT[p] for p in cls + ["__none__"]], rows)


def secondary_table(rep: dict, pred) -> str:
    rows = [[name(c["model"]), f"{c['a']} − {c['b']}", fs(c["diff"]), ci(c["ci"]), fp(c["p"]),
             str(c["n"])] for c in rep["secondary_contrasts"] if pred(c)]
    return table(["modelo", "a − b", "d", "IC 95 %", "p (sin corregir)", "n"], rows)


ESTIMATE_USD = 575.95  # scripts/estimate_run.py, docs/preregistration.md annex B


def cost_totals(cost: dict) -> str:
    """Run total (every set but the sensitivity copies) against the estimate, then the
    sensitivity apart; old cost.json files without the split fall back to one total."""
    if "run_total_usd" not in cost:
        return (f"Total: {cost['total_usd']:.2f} USD (+ ≈ {cost['lost_usd_approx']:.2f} de "
                "reintentos que no quedan en los ficheros).")
    run, by_set = cost["run_total_usd"], cost["by_set_usd"]
    sens = sum(by_set[s] for s in cost["sensitivity_sets"])
    parts = " + ".join(f"{s} {u:.2f}" for s, u in sorted(by_set.items())
                       if s not in cost["sensitivity_sets"])
    return (f"Total de la corrida ({parts}): **{run:.2f} USD**, frente a {ESTIMATE_USD:.2f} "
            f"estimados ({run - ESTIMATE_USD:+.2f}; {100 * (run / ESTIMATE_USD - 1):+.1f} %). "
            f"Aparte, la sensibilidad D21: {sens:.2f} USD (con ella, "
            f"{cost['total_usd_incl_sensitivity']:.2f}). Más ≈ {cost['lost_usd_approx']:.2f} "
            "de reintentos que no quedan en los ficheros.")


def main() -> None:
    d = pathlib.Path(sys.argv[1])
    J = lambda f: json.loads((d / f).read_text())  # noqa: E731
    prim, full, a7b, sens, static = (J("report_primary.json"), J("report_full_replicates.json"),
                                     J("report_a7b.json"), J("report_sens.json"),
                                     J("report_static.json"))
    extras = J("extras.json")
    cost = J("cost.json") if (d / "cost.json").exists() else None
    md = ["# Corrida final: tablas del análisis (generado por scripts/analysis/final_tables.py)", ""]

    md += ["## 1. Contrastes primarios (§5; familia de Holm = "
           f"{prim['holm']['family_size']}, α = {prim['holm']['alpha']})", "",
           contrasts_table(prim), ""]
    md += ["Celdas excluidas (> 50 % errores): " + (", ".join(
        f"{name(c['model'])} {c['cell']}" for c in prim["excluded_cells"]) or "ninguna")
        + ". Celdas marcadas (2-50 %): " + (", ".join(
            f"{name(c['model'])} {c['cell']} {c['errors']}/{c['n_rows']}"
            for c in prim["flagged_cells"]) or "ninguna")
        + ". Modelos excluidos: " + (", ".join(e["model"] for e in prim["excluded_models"])
                                     or "ninguno") + ".", ""]

    cl = classify(prim)
    md += ["## 2. Clasificación §6", "",
           table(["modelo", "order d [IC], p_holm", "motion_over_shape d [IC], p_holm",
                  "pc-acc motion/sheet [IC]", "LI > 0,25", "clasificación", "negativos sig."],
                 [[name(m), f"{fs(v['order']['diff'])} {ci(v['order']['ci'])}, {fp(v['order']['p_holm'])}",
                   f"{fs(v['motion_over_shape']['diff'])} {ci(v['motion_over_shape']['ci'])}, "
                   f"{fp(v['motion_over_shape']['p_holm'])}",
                   f"{f2(v['pc_acc'])} {ci(v['pc_ci'])}", "sí" if v["pc_above_chance"] else "no",
                   v["class"] + ("" if "lee" in v["class"] else
                                 f" (el orden aporta como mucho {v['order_upper']:+.2f})"),
                   ", ".join(v["significant_negative"]) or "—"] for m, v in cl.items()]), ""]

    md += ["## 3. Todas las celdas × modelos (§7; exploratorio)", "", cells_table(prim), ""]
    md += ["## 4. Recall por clase [IC 95 %]", "", recall_table(prim), ""]
    md += ["## 5. Matriz de confusión, motion/sheet (400 clips, 100 por deporte)", "",
           confusion_table(prim), ""]
    md += ["## 6. Cinemática (RQ2), secundarios del informe primario", "",
           secondary_table(prim, lambda c: "kinematics" in c["a"] + c["b"]), ""]
    manual = extras["manual_contrasts"]
    vid = [c for c in prim["secondary_contrasts"] if "video" in c["a"]]
    md += ["## 7. Vídeo (Gemini) y trails", "",
           table(["modelo", "a − b", "d", "IC 95 %", "p (sin corregir)", "n"],
                 [[name(c["model"]), f"{c['a']} − {c['b']}", fs(c["diff"]), ci(c["ci"]),
                   fp(c["p"]), str(c["n"])] for c in vid + manual]), ""]

    reps = full["replicates"]
    fo = {c["model"]: c for c in full["primary_contrasts"] if c["contrast"] == "order"}
    po = {c["model"]: c for c in prim["primary_contrasts"] if c["contrast"] == "order"}
    md += ["## 8. Réplicas (informe sobre `runs/final` completo; 3 réplicas, 200 clips)", "",
           table(["modelo", "celda", "acc media [IC]", "acc por réplica", "DE entre réplicas",
                  "acuerdo"],
                 [[name(g["model"]), g["cell"], f"{f2(g['mean_accuracy'])} {ci(g['mean_accuracy_ci'])}",
                   " / ".join(f2(x) for x in g["accuracy_per_replicate"]),
                   f"{g['sd_between_replicates']:.3f}", f2(g["agreement"])] for g in reps]), "",
           "Comprobación de robustez de §6 (`order` sobre la corrección media de las 3 réplicas):",
           "",
           table(["modelo", "order réplica 1 (400)", "order media 3 réplicas (200) [IC], p",
                  "mismo signo"],
                 [[name(m), fs(po[m]["diff"]),
                   f"{fs(c['diff'])} {ci(c['ci'])}, {fp(c['p'])}",
                   "sí" if (c["diff"] > 0) == (po[m]["diff"] > 0) or c["diff"] == 0 else "NO"]
                  for m, c in sorted(fo.items())]), ""]

    jev = extras["jev_text_vs_json"]
    md += ["## 9. Jev", "", cells_table(prim, lambda m: m.startswith("jev")), "",
           secondary_table(prim, lambda c: c["model"].startswith("jev")), "",
           "Texto plano frente a JSON (descriptivo):", "",
           table(["celda", "acc texto", "acc JSON", "misma etiqueta", "JSON − texto [IC]",
                  "respuestas texto", "respuestas JSON"],
                 [[j["cell"], f2(j["acc_text"]), f2(j["acc_json"]), f2(j["same_label"]),
                   f"{fs(j['json_minus_text']['diff'])} {ci(j['json_minus_text']['ci'])}",
                   " ".join(f"{SHORT.get(k, k)} {v}" for k, v in sorted(j["label_share_text"].items(), key=str)),
                   " ".join(f"{SHORT.get(k, k)} {v}" for k, v in sorted(j["label_share_json"].items(), key=str))]
                  for j in jev]), "",
           "max_input_tokens por celda de Jev: " + ", ".join(
               f"{name(r['model'])} {cell(r)} {r.get('max_input_tokens')}"
               for r in prim["runs"].values() if r["model"].startswith("jev")), ""]

    for kind in ("sport", "source"):
        sl = [s for s in extras["slices"] if s["kind"] == kind]
        md += [f"## 10{'a' if kind == 'sport' else 'b'}. Corte por {'deporte' if kind == 'sport' else 'fuente'} "
               "(motion/sheet; contrastes dentro del corte, p sin corregir)", "",
               table(["modelo", "corte", "n", "partidos", "acc motion [IC]", "acc shuffled",
                      "order d [IC], p", "acc formation", "motion_over_shape d [IC], p"],
                     [[name(s["model"]), s["slice"], str(s["n"]), str(s["n_matches"]),
                       f"{f2(s['acc_motion_sheet'])} {ci(s['acc_ci'])}", f2(s.get("acc_shuffled_sheet")),
                       f"{fs(s['order']['diff'])} {ci(s['order']['ci'])}, {fp(s['order']['p'])}",
                       f2(s.get("acc_formation_sheet")),
                       f"{fs(s['motion_over_shape']['diff'])} {ci(s['motion_over_shape']['ci'])}, "
                       f"{fp(s['motion_over_shape']['p'])}"] for s in sl]), ""]

    st_counts = collections.Counter()
    for r in static["runs"].values():
        if r["model"] == "minirocket" and r["condition"] == "motion":
            st_counts = {k: sum(v.values()) for k, v in r["confusion"].items()}
    md += ["## 11. Corte `--tag static` (mediana de velocidad < 1 m/s)", "",
           f"Clips por deporte: {', '.join(f'{SHORT[k]} {v}' for k, v in st_counts.items())}.", "",
           cells_table(static, lambda m: is_chat(m)), "",
           contrasts_table(static), "",
           "(Holm dentro de este informe de corte: no es la familia pre-registrada.)", ""]

    md += ["## 12. Especialistas sobre los mismos 400 clips", "",
           cells_table(prim, lambda m: m.startswith(SPECIALIST_PREFIXES)), "",
           secondary_table(prim, lambda c: c["model"].startswith(SPECIALIST_PREFIXES)), ""]

    nfl = extras["nfl_phase"]
    md += ["## 13. Fase de la jugada NFL (recall de fútbol americano por desfase del snap)", "",
           f"Desfases de los 100 clips NFL (mín., cuartiles, máx.): "
           f"{', '.join(f'{x:.1f}' for x in extras['nfl_offsets']['quartiles'])} s", "",
           table(["modelo", "celda"] + [f"[{lo:g}; {hi:g}) s" for lo, hi in
                                         sorted({tuple(x['bin']) for x in nfl})],
                 [[name(m), c] + [next((f"{x['recall_af']:.2f} (n={x['n']})" for x in nfl
                                        if x["model"] == m and x["cell"] == c and tuple(x["bin"]) == b), "—")
                                  for b in sorted({tuple(x['bin']) for x in nfl})]
                  for m, c in sorted({(x["model"], x["cell"]) for x in nfl})]), ""]

    ex = {c["model"]: c for c in a7b["secondary_contrasts"] if is_chat(c["model"])}
    spec8 = [c for c in a7b["secondary_contrasts"] if not is_chat(c["model"])]
    rb8 = runs_by(a7b)
    md += ["## 14. A7b, clips de 8 s (exploratoria; 3 deportes, azar 0,33; p sin corregir)", "",
           table(["modelo", "acc motion [IC]", "pc-acc motion [IC]", "acc shuffled",
                  "recall BB / HB / SO (motion)", "order 8 s d [IC], p", "order 4 s d (primario)",
                  "8 s ≠ 0", "8 s > 4 s (descriptivo)"],
                 [[name(m), f"{f2(rb8[(m, 'motion/sheet')]['accuracy'])} {ci(rb8[(m, 'motion/sheet')]['accuracy_ci'])}",
                   f"{f2(rb8[(m, 'motion/sheet')]['prior_corrected_accuracy'])} "
                   f"{ci(rb8[(m, 'motion/sheet')]['prior_corrected_accuracy_ci'])}",
                   f2(rb8[(m, 'motion_shuffled/sheet')]["accuracy"]) if (m, 'motion_shuffled/sheet') in rb8 else "—",
                   " / ".join(f2(rb8[(m, 'motion/sheet')]["per_class_recall"].get(k))
                              for k in ("basketball", "handball", "soccer")),
                   f"{fs(c['diff'])} {ci(c['ci'])}, {fp(c['p'])}",
                   fs(po[m]["diff"]) if m in po else "—",
                   "sí" if c["p"] < 0.05 else "no",
                   ("sí" if c["diff"] > po[m]["diff"] else "no") if m in po else "—"]
                  for m, c in sorted(ex.items())]), "",
           "Especialistas sobre los mismos 300 clips: " + "; ".join(
               f"{name(r['model'])} {cell(r)} acc {f2(r['accuracy'])} {ci(r['accuracy_ci'])}"
               for r in a7b["runs"].values() if not is_chat(r["model"])) + ". "
           + "; ".join(f"{name(c['model'])} {c['a']} − {c['b']} {fs(c['diff'])} {ci(c['ci'])}"
                       for c in spec8) + ".", ""]

    rs, rp = runs_by(sens), runs_by(prim)
    changed = [(k, rp[k], rs[k]) for k in rp if k in rs and
               (abs(rp[k]["accuracy"] - rs[k]["accuracy"]) > 1e-9 or rp[k]["cell_errors"] != rs[k]["cell_errors"])]
    ps, pp = primary_by(sens), primary_by(prim)
    chc = [(k, pp[k], ps[k]) for k in pp if k in ps and
           (abs(pp[k]["diff"] - ps[k]["diff"]) > 1e-9 or pp[k]["significant"] != ps[k]["significant"])]
    cs = classify(sens)
    md += ["## 15. Sensibilidad D21 (`runs/final-sens`: solo los 40 ítems con error de Sonnet motion/text, re-corridos con margen de razonamiento)", "",
           "Celdas que cambian:", "",
           table(["modelo", "celda", "errores orig → sens", "acc orig → sens", "pc-acc orig → sens",
                  "bal orig → sens", "recall AF/BB/HB/SO sens"],
                 [[name(k[0]), k[1], f"{a['cell_errors']} → {b['cell_errors']}",
                   f"{f2(a['accuracy'])} → {f2(b['accuracy'])} {ci(b['accuracy_ci'])}",
                   f"{f2(a['prior_corrected_accuracy'])} → {f2(b['prior_corrected_accuracy'])}",
                   f"{f2(a['balanced_accuracy'])} → {f2(b['balanced_accuracy'])}",
                   " / ".join(f2(b["per_class_recall"].get(x)) for x in prim["candidates"])]
                  for k, a, b in changed]), "",
           "Contrastes primarios que cambian (p_holm recalculado sobre la familia de 20):", "",
           table(["modelo", "contraste", "d orig → sens", "IC sens", "p orig → sens",
                  "p_holm orig → sens", "sig. orig → sens"],
                 [[name(k[0]), k[1], f"{fs(a['diff'])} → {fs(b['diff'])}", ci(b["ci"]),
                   f"{fp(a['p'])} → {fp(b['p'])}", f"{fp(a['p_holm'])} → {fp(b['p_holm'])}",
                   f"{'sí' if a['significant'] else 'no'} → {'sí' if b['significant'] else 'no'}"]
                  for k, a, b in chc]), "",
           "Holm: decisiones que cambian = " + str(sum(a["significant"] != b["significant"]
                                                       for _, a, b in chc))
           + "; clasificación §6 que cambia = " + (", ".join(
               f"{name(m)}: {cl[m]['class']} → {cs[m]['class']}" for m in cl if cl[m]["class"] != cs[m]["class"])
               or "ninguna") + ".", ""]

    if cost:
        md += ["## 16. Coste medido (precios de lista de `scripts/estimate_run.py`) y tiempo", "",
               table(["conjunto", "modelo", "filas", "Mtok entrada", "Mtok salida",
                      "de ellos razonamiento", "USD", "intentos perdidos", "≈ USD perdidos"],
                     [[v["set"], name(v["model"]), str(v["rows"]), f"{v['input_tokens'] / 1e6:.2f}",
                       f"{v['output_tokens'] / 1e6:.2f}", f"{v['thinking_tokens'] / 1e6:.2f}",
                       f"{v['usd']:.2f}", str(v.get("lost_attempts", 0)),
                       f"{v.get('lost_usd_approx', 0):.2f}"] for v in cost["by_model"]]), "",
               cost_totals(cost), ""]
        if cost.get("wall_clock"):
            md += [table(["log", "inicio", "fin", "horas"],
                         [[k, v["start"], v["end"], f"{v['hours']:.2f}"]
                          for k, v in cost["wall_clock"].items()]), ""]
    (d / "tables.md").write_text("\n".join(md))
    json.dump({"classification": cl, "classification_sens": cs},
              (d / "classification.json").open("w"), indent=2)
    print(f"wrote {d / 'tables.md'} and {d / 'classification.json'}")


if __name__ == "__main__":
    main()
