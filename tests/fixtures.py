"""Synthetic fixtures that stand in for the reference project.

The real study data in ``example/`` is optional -- it is a researcher's own
footage and coding, and a public checkout may not carry it. These builders
reproduce the parts of it the tests actually depend on, including the awkward
parts, so the suite is meaningful either way:

* a BORIS 7.0 project whose subjects are trial phases rather than actors;
* state events stored as two rows paired by ordinal position;
* media paths written with the doubled separator BORIS itself emits;
* an ethogram spreadsheet in BORIS's own export layout, written as real xlsx
  so the reader is exercised against the format rather than a stub.
"""

from __future__ import annotations

import json
import pathlib
import zipfile
from collections.abc import Sequence

#: Mirrors the shape of the reference ethogram: point events for behaviours,
#: state events for the ordered phases of a session.
BEHAVIOR_ROWS: list[tuple[str, str, str, str, str]] = [
    ("Approaching robot", "Point event", "Moving towards robot while orienting at it", "A", "#aa0000"),
    ("Approaching owner", "Point event",
     "Moving toward the owner while orienting at the owner or agent OR staying "
     "close to the owner the whole time", "S", "#00007f"),
    ("Orienting at owner", "Point event",
     "Dog orients at the owner's face immediately after orienting at the agent", "D", "#01750b"),
    ("Backing", "Point event",
     "Moving backwards but not toward the owner while orienting at the agent", "F", "#aa557f"),
    ("Whine", "Point event", "Short, cyclic, high pitched and tonal call", "G", "#ffff00"),
    ("Excited bark", "Point event", "Relatively but not extremely high-pitched bark", "H", "#00ffff"),
    ("Aggressive bark", "Point event", "Lower-pitched, noisier bark", "J", "#ff5500"),
    ("Growl", "Point event",
     "Low-frequency broadband, noisy vocalisation, built up of sequences of "
     "variable duration, divided by pauses", "K", "#aaaa00"),
    ("Puffing", "Point event",
     "Low-intensity vocalisation produced by air being forced through the "
     "slightly opened mouth", "L", "#ff557f"),
    ("Tucked tail", "Point event",
     "The tail of the dog is tucked under its rear, or is pushed downwards", "Y", "#5555ff"),
    ("Tail wagging", "Point event", "Horizontal tail movements", "X", "#ffbc14"),
    ("Play bow", "Point event", "The dog lowers itself on its forelegs and lifts its rear", "C", "#55aa00"),
    ("Jumping", "Point event", "The dog elevates both forelegs off the ground", "V", "#55aaff"),
    ("Shake off", "Point event", "The dog shakes its fur along its body axis", "B", "#ffffff"),
    ("Episode1_start", "State event", "from robot visible to first sequence", "w", ""),
    ("Episode2_firstseq", "State event", "from start of 1st sequence to end of 1st sequence", "e", ""),
    ("Episode3_pause", "State event", "from robot stopping until start of second sequence", "r", ""),
    ("Episode4_secondseq", "State event", "from start of 2nd sequence to end of 2nd sequence", "t", ""),
]

#: The subject-as-trial-phase idiom this project exists to convert away from.
PHASE_SUBJECTS = ["Episode1_start", "Episode2_firstseq", "Episode3_pause", "Episode4_secondseq"]


def example_dir() -> pathlib.Path | None:
    """The real reference data, when this checkout carries it."""
    candidate = pathlib.Path(__file__).resolve().parents[1] / "example"
    needed = ("ethogram.xlsx", "coded_project.boris", "subjects_aka_episodes.xlsx")
    return candidate if all((candidate / n).is_file() for n in needed) else None


# --------------------------------------------------------------------------
# xlsx
# --------------------------------------------------------------------------

def write_xlsx(path: pathlib.Path, rows: Sequence[Sequence[str]]) -> pathlib.Path:
    """Write a minimal single-sheet xlsx with inline strings.

    Inline strings keep this short: no shared-strings table to build, and the
    reader handles both forms, so the fixture still exercises the real parser.
    """
    def cell(col: int, row: int, text: str) -> str:
        ref = f"{_column_name(col)}{row}"
        escaped = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escaped}</t></is></c>'

    body = "".join(
        f'<row r="{r}">' + "".join(cell(c, r, str(v)) for c, v in enumerate(values)) + "</row>"
        for r, values in enumerate(rows, start=1)
    )
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{body}</sheetData></worksheet>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                   "</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   "</Relationships>")
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                   "</Relationships>")
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return path


def _column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def write_ethogram_xlsx(path: pathlib.Path) -> pathlib.Path:
    """A behaviours sheet in BORIS's own export layout."""
    header = ["Behavior code", "Behavior type", "Description", "Key", "Color",
              "Behavioral category", "Excluded behaviors", "Modifiers", "Modifiers (JSON)"]
    rows = [header] + [[code, kind, desc, key, color, "", "", "", ""]
                       for code, kind, desc, key, color in BEHAVIOR_ROWS]
    return write_xlsx(path, rows)


def write_subjects_xlsx(path: pathlib.Path) -> pathlib.Path:
    """A subjects sheet holding trial phases, as the reference project does."""
    rows = [["Key", "Subject name", "Description"]]
    rows += [["", name, f"phase {i + 1} of the session"] for i, name in enumerate(PHASE_SUBJECTS)]
    return write_xlsx(path, rows)


# --------------------------------------------------------------------------
# BORIS project
# --------------------------------------------------------------------------

def write_coded_project(
    path: pathlib.Path,
    *,
    observation_ids: Sequence[str] = ("03_30_152", "04_05_162"),
    media_dir: str = "Merged",
    fps: float = 25.0,
    duration: float = 500.16,
) -> pathlib.Path:
    """A coded BORIS 7.0 project in the reference project's awkward shape.

    Subjects are trial phases; a phase-marker row carries the phase as *both*
    subject and behaviour, while a real behaviour carries the phase as its
    subject. That is what :func:`migrate_subjects_to_episodes` has to unpick.
    """
    behaviors_conf = {
        str(i): {"type": kind, "key": key, "code": code, "description": desc,
                 "color": color, "category": "", "modifiers": "", "excluded": "",
                 "coding map": ""}
        for i, (code, kind, desc, key, color) in enumerate(BEHAVIOR_ROWS)
    }
    subjects_conf = {
        str(i): {"key": "", "name": name, "description": f"phase {i + 1} of the session"}
        for i, name in enumerate(PHASE_SUBJECTS)
    }

    observations = {}
    for n, obs_id in enumerate(observation_ids):
        # Doubled separator on purpose: BORIS writes these, and the reader has
        # to cope with them.
        media = f"{media_dir}//{obs_id}.MP4"
        base = 80.0 + n * 5
        events: list[list] = []

        # Four phases in order, each opened and closed, with a behaviour inside.
        cursor = base
        inner = ["Tucked tail", "Approaching robot", "Excited bark", "Tail wagging"]
        for phase, behavior in zip(PHASE_SUBJECTS, inner, strict=True):
            events.append(_row(cursor, phase, phase, fps))              # phase onset
            events.append(_row(cursor + 4.0, phase, behavior, fps))     # behaviour inside it
            events.append(_row(cursor + 8.0, phase, phase, fps))        # phase offset
            cursor += 10.0

        events.sort(key=lambda r: (r[0], r[2], r[1]))
        observations[obs_id] = {
            "file": {**{str(i): [] for i in range(1, 9)}, "1": [media]},
            "type": "MEDIA",
            "date": "2026-04-11T11:49:21.933",
            "description": "",
            "time offset": 0.0,
            "events": events,
            "observation time interval": [0, 0],
            "independent_variables": {},
            "visualize_spectrogram": False,
            "visualize_waveform": False,
            "media_creation_date_as_offset": False,
            "media_scan_sampling_duration": 0,
            "image_display_duration": 1,
            "close_behaviors_between_videos": False,
            "media_info": {
                "length": {media: duration}, "fps": {media: fps},
                "hasVideo": {media: True}, "hasAudio": {media: True},
                "offset": {"1": 0.0}, "display": {media: "Nothing"},
            },
        }

    project = {
        "time_format": "hh:mm:ss", "project_date": "", "project_name": "",
        "project_description": "", "project_format_version": "7.0",
        "subjects_conf": subjects_conf, "behaviors_conf": behaviors_conf,
        "observations": observations, "behavioral_categories": [],
        "behavioral_categories_config": {}, "independent_variables": {},
        "coding_map": {}, "behaviors_coding_map": [], "converters": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
    return path


def _row(time: float, subject: str, behavior: str, fps: float) -> list:
    """One BORIS events row: [time, subject, behaviour, modifiers, comment, frame]."""
    return [round(time, 2), subject, behavior, "", "", int(round(time * fps))]


def build_all(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    """Write the full fixture set into *directory*."""
    directory = pathlib.Path(directory)
    return {
        "ethogram_xlsx": write_ethogram_xlsx(directory / "ethogram.xlsx"),
        "subjects_xlsx": write_subjects_xlsx(directory / "subjects_aka_episodes.xlsx"),
        "coded_project": write_coded_project(directory / "coded_project.boris"),
    }


# --------------------------------------------------------------------------
# shared instance
# --------------------------------------------------------------------------

_SHARED: dict[str, pathlib.Path] | None = None


def shared() -> dict[str, pathlib.Path]:
    """The fixture set, built once per test session into a temp directory.

    Tests read these rather than ``example/`` so the suite is meaningful in a
    checkout that carries no study data. Tests that specifically want the real
    recordings ask for :func:`example_dir` and skip when it is absent.
    """
    global _SHARED
    if _SHARED is None:
        import atexit
        import shutil
        import tempfile

        directory = pathlib.Path(tempfile.mkdtemp(prefix="abc-fixtures-"))
        atexit.register(shutil.rmtree, directory, True)
        _SHARED = build_all(directory)
    return _SHARED
