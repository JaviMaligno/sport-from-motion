"""The one data type every stage passes around: a `Clip` of player trajectories.

A clip is deliberately *poorer* than the source data. It carries positions of
players over time and nothing else that could identify the sport by a shortcut:
no ball, no field lines, no surface, no jersey colours. Team labels and field
dimensions are kept as metadata (the leak controls need them to *remove* those
cues), but they never reach a model unless a condition explicitly asks for them.

Coordinates are in **metres** on the playing surface (top view) whenever the
source allows it, so that sports with different field sizes live in the same
units and scale can be removed on purpose rather than by accident.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

# Canonical sport ids. The display name is what a model sees in the prompt.
SPORTS: dict[str, str] = {
    "soccer": "association football (soccer)",
    "rugby_union": "rugby union",
    "rugby_sevens": "rugby sevens",
    "rugby_league": "rugby league",
    "american_football": "American football",
    "basketball": "basketball",
    "handball": "handball",
    "field_hockey": "field hockey",
    "ice_hockey": "ice hockey",
    "futsal": "futsal",
}

# Nominal field size (length, width) in metres. Only used by the controls and the
# nuisance baseline — it is exactly the kind of cue the experiment must not leak.
FIELD_SIZE_M: dict[str, tuple[float, float]] = {
    "soccer": (105.0, 68.0),
    "rugby_union": (100.0, 70.0),  # try line to try line, excluding in-goal
    "rugby_sevens": (100.0, 70.0),
    "rugby_league": (100.0, 68.0),
    "american_football": (109.7, 48.8),  # incl. end zones
    "basketball": (28.65, 15.24),  # NBA
    "handball": (40.0, 20.0),
    "field_hockey": (91.4, 55.0),
    "ice_hockey": (60.0, 26.0),
    "futsal": (40.0, 20.0),
}


@dataclass
class Clip:
    """Player trajectories for one short window of play.

    xy:    float32 [T, N, 2] positions in metres (NaN where a player is not observed).
    fps:   sampling rate of `xy` after any resampling.
    team:  optional int8 [N] (+1 / -1). Metadata only; never rendered by default.
    tags:  free-form labels used to define sub-cases, e.g. "set_piece", "static",
           "pre_snap", "lineout". The formation sub-study filters on these.
    """

    clip_id: str
    sport: str
    source: str
    match_id: str
    fps: float
    xy: np.ndarray
    team: np.ndarray | None = None
    field_size: tuple[float, float] | None = None
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sport not in SPORTS:
            raise ValueError(f"unknown sport {self.sport!r}; known: {sorted(SPORTS)}")
        self.xy = np.asarray(self.xy, dtype=np.float32)
        if self.xy.ndim != 3 or self.xy.shape[2] != 2:
            raise ValueError(f"xy must be [T, N, 2], got {self.xy.shape}")
        if self.team is not None:
            self.team = np.asarray(self.team, dtype=np.int8)
            if self.team.shape != (self.n_players,):
                raise ValueError("team must have one entry per player")

    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]

    @property
    def n_players(self) -> int:
        return self.xy.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def replace(self, **changes) -> "Clip":
        data = {**self.__dict__, **changes}
        data["tags"] = list(data["tags"])
        data["meta"] = dict(data["meta"])
        return Clip(**data)

    # ---- persistence: one .npz per clip, metadata as JSON inside it ----

    def save(self, directory: str | pathlib.Path) -> pathlib.Path:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.clip_id}.npz"
        meta = {
            "clip_id": self.clip_id, "sport": self.sport, "source": self.source,
            "match_id": self.match_id, "fps": self.fps,
            "field_size": list(self.field_size) if self.field_size else None,
            "tags": self.tags, "meta": self.meta,
        }
        arrays = {"xy": self.xy, "meta_json": np.array(json.dumps(meta))}
        if self.team is not None:
            arrays["team"] = self.team
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Clip":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            team = z["team"] if "team" in z.files else None
            xy = z["xy"]
        fs = meta.pop("field_size")
        return cls(xy=xy, team=team, field_size=tuple(fs) if fs else None, **meta)


def load_clips(directory: str | pathlib.Path) -> list[Clip]:
    return [Clip.load(p) for p in sorted(pathlib.Path(directory).glob("*.npz"))]
