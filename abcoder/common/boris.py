"""Read and write BORIS projects (.boris, format version 7.0).

ABC treats the BORIS project as its output format, not an export: the file the
client writes is meant to be opened in BORIS, scrubbed through, corrected by a
human, and saved again without ABC ever touching it. That drives two rules
here:

* every key BORIS writes is written back, in the shape BORIS expects, even the
  ones ABC does not use -- a missing key makes BORIS fall back to defaults and
  silently lose settings;
* media paths are stored **relative to the .boris file** whenever that is
  possible, so the project and its video folder can be moved or shared as a
  unit. Absolute server-side paths are never written into a client project.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Sequence
from typing import Any

from ..version import BORIS_FORMAT_VERSION
from .ethogram import (
    EPISODE_CATEGORY,
    STATE,
    Behavior,
    Ethogram,
    Subject,
    normalise_color,
)
from .events import NO_FOCAL_SUBJECT, Event, MediaInfo

#: BORIS supports up to eight simultaneous media players per observation.
N_PLAYERS = 8

TIME_FORMAT_HHMMSS = "hh:mm:ss"
TIME_FORMAT_SECONDS = "s"


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def relative_media_path(media: str | os.PathLike, project_file: str | os.PathLike) -> str:
    """Return *media* relative to the directory holding *project_file*.

    Falls back to an absolute path when no relative route exists (different
    Windows drive, or a UNC share). Always POSIX-separated: BORIS accepts
    forward slashes on every platform, and a backslash written on Windows makes
    the project unusable on Linux.
    """
    media_p = pathlib.Path(media).expanduser()
    project_dir = pathlib.Path(project_file).expanduser().parent
    try:
        media_abs = media_p.resolve()
        base_abs = project_dir.resolve()
    except OSError:  # pragma: no cover - unreadable mount
        media_abs, base_abs = media_p, project_dir
    try:
        rel = os.path.relpath(media_abs, base_abs)
    except ValueError:
        return media_abs.as_posix()
    # Refuse to climb out of the project tree with a long ../../.. chain; an
    # absolute path is more robust and more honest about where the media lives.
    if rel.startswith(".." + os.sep + ".." + os.sep):
        return media_abs.as_posix()
    return pathlib.PurePath(rel).as_posix()


def resolve_media_path(stored: str, project_file: str | os.PathLike) -> pathlib.Path:
    """Inverse of :func:`relative_media_path`: stored path -> a real path.

    Handles the doubled separator BORIS sometimes writes (``Merged//clip.mp4``)
    and normalises Windows backslashes so projects move between platforms.
    """
    cleaned = stored.replace("\\", "/")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    p = pathlib.Path(cleaned)
    if p.is_absolute():
        return p
    return (pathlib.Path(project_file).expanduser().parent / p).resolve()


# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------

def seconds_to_hhmmss(seconds: float) -> str:
    """Format seconds the way BORIS shows them (``HH:MM:SS.mmm``)."""
    if seconds < 0:
        return "-" + seconds_to_hhmmss(-seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


# --------------------------------------------------------------------------
# project
# --------------------------------------------------------------------------

class BorisProject:
    """A BORIS project held in memory, with ABC's helpers on top."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = data if data is not None else self._blank()

    # -- construction -------------------------------------------------------
    @staticmethod
    def _blank() -> dict[str, Any]:
        return {
            "time_format": TIME_FORMAT_HHMMSS,
            "project_date": _dt.datetime.now().isoformat(timespec="seconds"),
            "project_name": "",
            "project_description": "",
            "project_format_version": BORIS_FORMAT_VERSION,
            "subjects_conf": {},
            "behaviors_conf": {},
            "observations": {},
            "behavioral_categories": [],
            "behavioral_categories_config": {},
            "independent_variables": {},
            "coding_map": {},
            "behaviors_coding_map": [],
            "converters": {},
        }

    @classmethod
    def load(cls, path: str | os.PathLike) -> BorisProject:
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def save(self, path: str | os.PathLike) -> None:
        """Write the project. BORIS itself writes compact JSON; so do we."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, ensure_ascii=False)
        os.replace(tmp, path)

    # -- metadata -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.data.get("project_name", "")

    @name.setter
    def name(self, value: str) -> None:
        self.data["project_name"] = value

    @property
    def description(self) -> str:
        return self.data.get("project_description", "")

    @description.setter
    def description(self, value: str) -> None:
        self.data["project_description"] = value

    @property
    def time_format(self) -> str:
        return self.data.get("time_format", TIME_FORMAT_HHMMSS)

    @property
    def observation_ids(self) -> list[str]:
        return list(self.data.get("observations", {}).keys())

    # -- ethogram -----------------------------------------------------------
    @property
    def ethogram(self) -> Ethogram:
        """The project's coding scheme as an :class:`Ethogram`."""
        behaviors = [
            Behavior.from_boris(v)
            for _, v in sorted(
                self.data.get("behaviors_conf", {}).items(),
                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
            )
        ]
        subjects = [
            Subject.from_boris(v)
            for _, v in sorted(
                self.data.get("subjects_conf", {}).items(),
                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
            )
        ]
        categories = list(self.data.get("behavioral_categories", []))
        return Ethogram(behaviors=behaviors, subjects=subjects, categories=categories)

    @ethogram.setter
    def ethogram(self, etho: Ethogram) -> None:
        self.data["behaviors_conf"] = {
            str(i): b.to_boris(i) for i, b in enumerate(etho.behaviors)
        }
        self.data["subjects_conf"] = {
            str(i): s.to_boris() for i, s in enumerate(etho.subjects)
        }
        cats = list(etho.categories)
        for b in etho.behaviors:
            if b.category and b.category not in cats:
                cats.append(b.category)
        self.data["behavioral_categories"] = cats
        self.data.setdefault("behavioral_categories_config", {})
        for c in cats:
            self.data["behavioral_categories_config"].setdefault(c, "")

    # -- observations -------------------------------------------------------
    def add_observation(
        self,
        observation_id: str,
        media_paths: Sequence[str],
        events: Iterable[Event],
        media_info: Sequence[MediaInfo] = (),
        *,
        description: str = "",
        date: str | None = None,
        time_offset: float = 0.0,
        independent_variables: dict[str, Any] | None = None,
        player: int = 1,
    ) -> None:
        """Insert one observation.

        *media_paths* must already be in the form to store (call
        :func:`relative_media_path` first); *media_info* is keyed by the same
        strings. ABC writes every media file into a single player, which is
        what BORIS does for a one-camera observation.
        """
        files: dict[str, list[str]] = {str(i): [] for i in range(1, N_PLAYERS + 1)}
        files[str(player)] = list(media_paths)

        info = {m.path: m for m in media_info}
        length = {p: float(info[p].duration) if p in info else 0.0 for p in media_paths}
        fps = {p: float(info[p].fps) if p in info else 0.0 for p in media_paths}
        has_video = {p: bool(info[p].has_video) if p in info else True for p in media_paths}
        has_audio = {p: bool(info[p].has_audio) if p in info else False for p in media_paths}

        self.data.setdefault("observations", {})[observation_id] = {
            "file": files,
            "type": "MEDIA",
            "date": date or _dt.datetime.now().isoformat(timespec="milliseconds"),
            "description": description,
            "time offset": float(time_offset),
            "events": events_to_boris(events, self.ethogram, fps_by_path=fps,
                                      media_paths=list(media_paths)),
            "observation time interval": [0, 0],
            "independent_variables": independent_variables or {},
            "visualize_spectrogram": False,
            "visualize_waveform": False,
            "media_creation_date_as_offset": False,
            "media_scan_sampling_duration": 0,
            "image_display_duration": 1,
            "close_behaviors_between_videos": False,
            "media_info": {
                "length": length,
                "fps": fps,
                "hasVideo": has_video,
                "hasAudio": has_audio,
                "offset": {str(player): float(time_offset)},
                "display": dict.fromkeys(media_paths, "Nothing"),
            },
        }

    def observation_events(self, observation_id: str) -> list[Event]:
        """Read one observation's events back into the canonical model."""
        obs = self.data.get("observations", {}).get(observation_id)
        if obs is None:
            raise KeyError(observation_id)
        return events_from_boris(obs.get("events", []), self.ethogram)

    def observation_media(self, observation_id: str) -> list[str]:
        obs = self.data.get("observations", {}).get(observation_id, {})
        out: list[str] = []
        for i in range(1, N_PLAYERS + 1):
            out.extend(obs.get("file", {}).get(str(i), []))
        return out

    def remove_observation(self, observation_id: str) -> None:
        self.data.get("observations", {}).pop(observation_id, None)

    # -- path maintenance ---------------------------------------------------
    def rewrite_media_paths(
        self, resolver, project_file: str | os.PathLike
    ) -> list[str]:
        """Re-point every observation's media through *resolver*.

        *resolver* takes the stored path and returns a real filesystem path (or
        ``None`` to leave it alone). Used when a project produced elsewhere is
        opened against a local copy of the videos. Returns the paths it could
        not resolve.
        """
        unresolved: list[str] = []
        for obs_id, obs in self.data.get("observations", {}).items():
            mapping: dict[str, str] = {}
            for i in range(1, N_PLAYERS + 1):
                key = str(i)
                new_list = []
                for stored in obs.get("file", {}).get(key, []):
                    target = resolver(stored, obs_id)
                    if target is None:
                        new_list.append(stored)
                        unresolved.append(f"{obs_id}: {stored}")
                        continue
                    new_stored = relative_media_path(target, project_file)
                    mapping[stored] = new_stored
                    new_list.append(new_stored)
                if new_list:
                    obs.setdefault("file", {})[key] = new_list
            mi = obs.get("media_info", {})
            for field in ("length", "fps", "hasVideo", "hasAudio", "display"):
                if field in mi:
                    mi[field] = {mapping.get(k, k): v for k, v in mi[field].items()}
        return unresolved


# --------------------------------------------------------------------------
# event conversion
# --------------------------------------------------------------------------

def events_to_boris(
    events: Iterable[Event],
    ethogram: Ethogram,
    fps_by_path: dict[str, float] | None = None,
    media_paths: Sequence[str] | None = None,
) -> list[list[Any]]:
    """Canonical events -> BORIS ``events`` rows.

    A BORIS row is ``[time, subject, behaviour, modifiers, comment, frame]``.
    A state event becomes two rows -- BORIS stores onset and offset as separate
    occurrences of the same code and pairs them by ordinal position, so the
    rows must be emitted in time order and must always come in pairs.
    """
    fps = 0.0
    if fps_by_path and media_paths:
        for p in media_paths:
            if fps_by_path.get(p):
                fps = float(fps_by_path[p])
                break

    rows: list[list[Any]] = []
    for e in events:
        beh = ethogram.get(e.behavior)
        state = beh.is_state if beh is not None else (e.stop is not None)
        comment = e.comment or ""
        rows.append([round(e.start, 3), e.subject, e.behavior,
                     e.modifiers or "", comment, _frame(e.start, fps)])
        if state:
            stop = e.stop if e.stop is not None else e.start
            rows.append([round(stop, 3), e.subject, e.behavior,
                         e.modifiers or "", comment, _frame(stop, fps)])
    rows.sort(key=lambda r: (r[0], r[2], r[1]))
    return rows


def events_from_boris(rows: Sequence[Sequence[Any]], ethogram: Ethogram) -> list[Event]:
    """BORIS ``events`` rows -> canonical events, re-pairing state onsets/offsets."""
    open_states: dict[tuple[str, str, str], list[Any]] = {}
    out: list[Event] = []

    for row in sorted(rows, key=lambda r: float(r[0])):
        time = float(row[0])
        subject = row[1] if len(row) > 1 else NO_FOCAL_SUBJECT
        behavior = row[2] if len(row) > 2 else ""
        modifiers = row[3] if len(row) > 3 else ""
        comment = row[4] if len(row) > 4 else ""
        beh = ethogram.get(behavior)
        state = beh.is_state if beh is not None else False

        if not state:
            out.append(Event(behavior=behavior, start=time, subject=subject,
                             modifiers=str(modifiers or ""), comment=str(comment or "")))
            continue

        key = (behavior, subject, str(modifiers or ""))
        if key in open_states:
            start_row = open_states.pop(key)
            out.append(Event(behavior=behavior, start=float(start_row[0]), stop=time,
                             subject=subject, modifiers=str(modifiers or ""),
                             comment=str(start_row[4] if len(start_row) > 4 else "")))
        else:
            open_states[key] = row

    # An unpaired onset means the human stopped coding mid-state; keep it open
    # rather than inventing an offset, and let the caller decide.
    for key, row in open_states.items():
        behavior, subject, modifiers = key
        out.append(Event(behavior=behavior, start=float(row[0]), stop=None,
                         subject=subject, modifiers=modifiers,
                         comment=str(row[4] if len(row) > 4 else "")))
    out.sort(key=lambda e: (e.start, e.behavior, e.subject))
    return out


def _frame(seconds: float, fps: float) -> int:
    """BORIS's sixth column: the 0-based frame index of the event."""
    if not fps or fps <= 0:
        return 0
    return int(round(seconds * fps))


# --------------------------------------------------------------------------
# ethogram import / export
# --------------------------------------------------------------------------

def ethogram_from_boris_file(path: str | os.PathLike) -> Ethogram:
    """Load an ethogram from a .boris project (observations ignored)."""
    return BorisProject.load(path).ethogram


def ethogram_to_boris_file(etho: Ethogram, path: str | os.PathLike, name: str = "") -> None:
    """Write an ethogram-only .boris file, the way BORIS exports one."""
    proj = BorisProject()
    proj.ethogram = etho
    proj.name = name
    proj.save(path)


#: Column headers BORIS uses when it exports an ethogram to a spreadsheet.
_XLSX_BEHAVIOR_COLUMNS = {
    "behavior code": "code",
    "behavior type": "type",
    "description": "description",
    "key": "key",
    "color": "color",
    "behavioral category": "category",
    "excluded behaviors": "excluded",
    "modifiers": "modifiers",
}
_XLSX_SUBJECT_COLUMNS = {
    "subject name": "name",
    "description": "description",
    "key": "key",
}


def _read_xlsx_rows(path: str | os.PathLike) -> list[list[str]]:
    """Minimal xlsx reader: first worksheet, as a list of string rows.

    Deliberately dependency-free -- the client must be able to import a BORIS
    spreadsheet export without pulling in openpyxl, and this format is simple
    enough that the stdlib handles it.
    """
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{ns}t")) for si in root]

        sheets = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet")]
        if not sheets:
            return []
        root = ET.fromstring(z.read(sorted(sheets)[0]))

        rows: list[list[str]] = []
        for row in root.iter(f"{ns}row"):
            cells: list[str] = []
            for c in row.iter(f"{ns}c"):
                ctype = c.get("t")
                v = c.find(f"{ns}v")
                text = v.text if v is not None else ""
                if ctype == "s" and text:
                    idx = int(text)
                    text = shared[idx] if 0 <= idx < len(shared) else ""
                elif ctype == "inlineStr":
                    is_el = c.find(f"{ns}is")
                    text = ("".join(t.text or "" for t in is_el.iter(f"{ns}t"))
                            if is_el is not None else "")
                cells.append(text or "")
            rows.append(cells)
        return rows


def _read_delimited_rows(path: str | os.PathLike) -> list[list[str]]:
    import csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel_tab if "\t" in sample else csv.excel
        return [list(r) for r in csv.reader(fh, dialect)]


def ethogram_from_table(path: str | os.PathLike) -> Ethogram:
    """Import an ethogram from a BORIS spreadsheet export (.xlsx/.csv/.tsv).

    Recognises the header row BORIS writes and is tolerant about column order
    and capitalisation. A sheet whose headers look like a subject list is
    imported as subjects instead of behaviours, so the same function handles
    both files BORIS exports.
    """
    path = pathlib.Path(path)
    rows = (_read_xlsx_rows(path) if path.suffix.lower() in (".xlsx", ".xlsm")
            else _read_delimited_rows(path))
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return Ethogram()

    header = [(c or "").strip().lower() for c in rows[0]]
    etho = Ethogram()

    if "subject name" in header:
        idx = {v: header.index(k) for k, v in _XLSX_SUBJECT_COLUMNS.items() if k in header}
        for row in rows[1:]:
            name = _cell(row, idx, "name")
            if name:
                etho.add_subject(Subject(name=name,
                                         description=_cell(row, idx, "description"),
                                         key=_cell(row, idx, "key")))
        return etho

    idx = {v: header.index(k) for k, v in _XLSX_BEHAVIOR_COLUMNS.items() if k in header}
    if "code" not in idx:
        raise ValueError(
            f"{path.name} has no 'Behavior code' column; expected a BORIS ethogram export"
        )
    for i, row in enumerate(rows[1:]):
        code = _cell(row, idx, "code")
        if not code:
            continue
        etho.add(Behavior(
            code=code,
            type=_cell(row, idx, "type") or "Point event",
            description=_cell(row, idx, "description"),
            key=_cell(row, idx, "key"),
            color=normalise_color(_cell(row, idx, "color"), i),
            category=_cell(row, idx, "category"),
            excluded=_cell(row, idx, "excluded"),
            modifiers=_cell(row, idx, "modifiers"),
        ))
    return etho


def _cell(row: Sequence[str], columns: dict[str, int], field: str) -> str:
    """One field of a spreadsheet row, empty when the column is absent or short.

    Exported sheets are ragged: trailing empty cells are simply not written, so
    indexing has to tolerate a row shorter than the header.
    """
    position = columns.get(field)
    if position is None or position >= len(row):
        return ""
    return (row[position] or "").strip()


def ethogram_to_table(etho: Ethogram, path: str | os.PathLike) -> None:
    """Export the ethogram as a BORIS-compatible TSV."""
    headers = ["Behavior code", "Behavior type", "Description", "Key", "Color",
               "Behavioral category", "Excluded behaviors", "Modifiers"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(headers) + "\n")
        for i, b in enumerate(etho.behaviors):
            fh.write("\t".join([
                b.code, b.type, b.description.replace("\t", " "), b.key,
                normalise_color(b.color, i), b.category, b.excluded,
                str(b.modifiers or ""),
            ]) + "\n")


# --------------------------------------------------------------------------
# migration: subjects-as-phases -> proper BORIS semantics
# --------------------------------------------------------------------------

def migrate_subjects_to_episodes(
    project: BorisProject,
    real_subjects: Sequence[Subject],
    *,
    category: str = EPISODE_CATEGORY,
    keep_existing_episode_behaviors: bool = True,
) -> tuple[BorisProject, list[str]]:
    """Convert a project that used the subject field to mark trial phases.

    Every former subject becomes a state behaviour in *category*, the events
    that merely marked a phase boundary become that state's onset/offset, and
    all real behaviour events are re-attributed to *real_subjects* -- to the
    single subject when only one is given, otherwise left for a human to split.

    Returns a new project and a report of what changed. The input is not
    modified.
    """
    report: list[str] = []
    old = project.ethogram
    phase_names = [s.name for s in old.subjects]
    if not phase_names:
        report.append("Project declares no subjects; nothing to migrate.")
        return BorisProject(json.loads(json.dumps(project.data))), report

    new = BorisProject(json.loads(json.dumps(project.data)))
    etho = new.ethogram

    # Phases that already exist as behaviours keep their definition; the rest
    # are created as state events in the episode category.
    for name in phase_names:
        existing = etho.get(name)
        subj = next((s for s in old.subjects if s.name == name), None)
        desc = subj.description if subj else ""
        if existing is None:
            etho.add(Behavior(code=name, type=STATE, description=desc, category=category))
            report.append(f"created state behaviour {name!r} in category {category!r}")
        else:
            if existing.type != STATE:
                existing.type = STATE
                report.append(f"{name!r} was a point event; changed to a state event")
            if keep_existing_episode_behaviors and not existing.category:
                existing.category = category
                report.append(f"moved {name!r} into category {category!r}")
            if not existing.description and desc:
                existing.description = desc

    etho.subjects = list(real_subjects)
    if category not in etho.categories:
        etho.categories.append(category)
    new.ethogram = etho

    default_subject = real_subjects[0].name if len(real_subjects) == 1 else (
        real_subjects[0].name if real_subjects else NO_FOCAL_SUBJECT
    )
    if len(real_subjects) > 1:
        report.append(
            f"{len(real_subjects)} subjects declared; every event was attributed to "
            f"{default_subject!r} because the source project did not record which "
            f"actor performed it -- split them in BORIS."
        )

    phase_set = set(phase_names)
    for obs_id, obs in new.data.get("observations", {}).items():
        rows = obs.get("events", [])
        rebuilt: list[list[Any]] = []
        n_phase = 0
        for row in rows:
            row = list(row)
            subject = row[1] if len(row) > 1 else ""
            behavior = row[2] if len(row) > 2 else ""
            # A row where subject == behaviour was a phase marker, not a behaviour.
            if behavior in phase_set and subject == behavior:
                row[1] = NO_FOCAL_SUBJECT
                n_phase += 1
            else:
                row[1] = default_subject
            rebuilt.append(row)
        rebuilt.sort(key=lambda r: (float(r[0]), str(r[2]), str(r[1])))
        obs["events"] = rebuilt
        if n_phase:
            report.append(f"{obs_id}: converted {n_phase} phase marker row(s)")

    return new, report
