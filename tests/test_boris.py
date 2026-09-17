"""BORIS reading, writing and path handling.

These run against the real example project in ``example/``, because the format
has corners that a synthetic fixture would not reproduce -- doubled path
separators, state events paired by ordinal position, and a subjects table that
holds trial phases.
"""

import json
import pathlib
import tempfile
import unittest

from abcoder.common.boris import (
    BorisProject,
    ethogram_from_table,
    ethogram_to_boris_file,
    events_from_boris,
    events_to_boris,
    migrate_subjects_to_episodes,
    relative_media_path,
    resolve_media_path,
    seconds_to_hhmmss,
)
from abcoder.common.ethogram import STATE, Behavior, Ethogram, Subject
from abcoder.common.events import Event
from tests import fixtures

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: Synthetic stand-ins for the reference study data, so the suite is meaningful
#: in a checkout that carries no recordings. See tests/fixtures.py.
FIXTURES = fixtures.shared()


class TestPaths(unittest.TestCase):
    def test_relative_media_path_is_posix(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            (base / "Merged").mkdir()
            video = base / "Merged" / "clip.mp4"
            video.touch()
            self.assertEqual(
                relative_media_path(video, base / "project.boris"), "Merged/clip.mp4")

    def test_resolve_handles_doubled_separators(self):
        resolved = resolve_media_path("Merged//03_30_1_152.MP4",
                                      FIXTURES["coded_project"])
        self.assertEqual(resolved.name, "03_30_1_152.MP4")
        self.assertEqual(resolved.parent.name, "Merged")

    def test_resolve_handles_windows_separators(self):
        resolved = resolve_media_path(r"Merged\clip.mp4", pathlib.Path("/tmp/p.boris"))
        self.assertEqual(resolved.parent.name, "Merged")

    def test_absolute_when_outside_the_tree(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            video = pathlib.Path(a) / "deep" / "deeper" / "clip.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            stored = relative_media_path(video, pathlib.Path(b) / "x" / "y" / "p.boris")
            self.assertTrue(pathlib.PurePosixPath(stored).is_absolute())

    def test_hhmmss(self):
        self.assertEqual(seconds_to_hhmmss(0), "00:00:00.000")
        self.assertEqual(seconds_to_hhmmss(3661.5), "01:01:01.500")


class TestEthogramImport(unittest.TestCase):
    def test_reads_the_boris_xlsx_export(self):
        etho = ethogram_from_table(FIXTURES["ethogram_xlsx"])
        self.assertEqual(len(etho), 18)
        self.assertEqual(len(etho.state_behaviors()), 4)
        self.assertTrue(etho.has("Approaching robot"))
        self.assertEqual(etho.get("Growl").type, "Point event")

    def test_reads_a_subjects_sheet_as_subjects(self):
        etho = ethogram_from_table(FIXTURES["subjects_xlsx"])
        self.assertEqual(len(etho.subjects), 4)
        self.assertEqual(etho.subjects[0].name, "Episode1_start")
        self.assertEqual(len(etho.behaviors), 0)

    def test_ethogram_only_boris_file(self):
        # An ethogram-only .boris file: behaviours and subjects, no observations.
        # This is what `abc import-ethogram` writes and what BORIS exports.
        source = ethogram_from_table(FIXTURES["ethogram_xlsx"])
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "ethogram.boris"
            ethogram_to_boris_file(source, path, name="fixture")
            reloaded = BorisProject.load(path)
        self.assertEqual(len(reloaded.ethogram), 18)
        self.assertEqual(reloaded.observation_ids, [])
        self.assertEqual(reloaded.name, "fixture")


class TestProjectRoundTrip(unittest.TestCase):
    def setUp(self):
        self.project = BorisProject.load(FIXTURES["coded_project"])

    def test_reads_the_example_project(self):
        self.assertEqual(len(self.project.observation_ids), 2)
        self.assertEqual(self.project.data["project_format_version"], "7.0")

    def test_state_events_are_paired_by_position(self):
        events = self.project.observation_events("03_30_152")
        episodes = [e for e in events if e.behavior.startswith("Episode")]
        self.assertTrue(episodes)
        for e in episodes:
            self.assertIsNotNone(e.stop, f"{e.behavior} lost its offset")
            self.assertGreater(e.stop, e.start)

    def test_write_preserves_every_top_level_key(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "out.boris"
            self.project.save(out)
            reloaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(set(reloaded), set(self.project.data))

    def test_events_round_trip(self):
        etho = Ethogram(behaviors=[Behavior("Bark"), Behavior("Phase", STATE)],
                        subjects=[Subject("Dog")])
        original = [Event("Bark", 10.0, subject="Dog"),
                    Event("Phase", 5.0, stop=20.0, subject="Dog")]
        rows = events_to_boris(original, etho, fps_by_path={"a": 25.0}, media_paths=["a"])
        # A point event is one row; a state event is two.
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][5], 125)  # 5.0s at 25 fps
        recovered = events_from_boris(rows, etho)
        self.assertEqual(len(recovered), 2)
        phase = next(e for e in recovered if e.behavior == "Phase")
        self.assertEqual((phase.start, phase.stop), (5.0, 20.0))

    def test_unpaired_state_onset_stays_open(self):
        etho = Ethogram(behaviors=[Behavior("Phase", STATE)])
        recovered = events_from_boris([[5.0, "", "Phase", "", "", 0]], etho)
        self.assertEqual(len(recovered), 1)
        self.assertIsNone(recovered[0].stop)


class TestMigration(unittest.TestCase):
    def test_phases_become_behaviours_and_events_get_a_real_subject(self):
        project = BorisProject.load(FIXTURES["coded_project"])
        migrated, report = migrate_subjects_to_episodes(project, [Subject("Dog")])

        self.assertEqual(migrated.ethogram.subject_names, ["Dog"])
        episodes = [b.code for b in migrated.ethogram.episode_behaviors()]
        self.assertIn("Episode1_start", episodes)
        self.assertTrue(report)

        events = migrated.observation_events("03_30_152")
        for e in events:
            if e.behavior.startswith("Episode"):
                self.assertEqual(e.subject, "", "a trial phase must have no focal subject")
            else:
                self.assertEqual(e.subject, "Dog")

    def test_the_input_is_not_modified(self):
        project = BorisProject.load(FIXTURES["coded_project"])
        before = json.dumps(project.data, sort_keys=True)
        migrate_subjects_to_episodes(project, [Subject("Dog")])
        self.assertEqual(before, json.dumps(project.data, sort_keys=True))


if __name__ == "__main__":
    unittest.main()


class TestRealExampleData(unittest.TestCase):
    """Run against the reference study data when this checkout carries it.

    The synthetic fixtures reproduce the format's awkward parts deliberately;
    these tests confirm the real thing still parses, and are skipped otherwise.
    """

    @classmethod
    def setUpClass(cls):
        cls.example = fixtures.example_dir()
        if cls.example is None:
            raise unittest.SkipTest("example/ study data is not present in this checkout")

    def test_reads_the_real_ethogram_export(self):
        etho = ethogram_from_table(self.example / "ethogram.xlsx")
        self.assertGreater(len(etho), 0)
        self.assertTrue(etho.state_behaviors())

    def test_reads_the_real_coded_project(self):
        project = BorisProject.load(self.example / "coded_project.boris")
        self.assertTrue(project.observation_ids)
        self.assertEqual(project.data["project_format_version"], "7.0")
        for obs_id in project.observation_ids:
            for event in project.observation_events(obs_id):
                behavior = project.ethogram.get(event.behavior)
                if behavior is not None and behavior.is_state and event.stop is not None:
                    self.assertGreaterEqual(event.stop, event.start)

    def test_migration_of_the_real_project(self):
        project = BorisProject.load(self.example / "coded_project.boris")
        migrated, report = migrate_subjects_to_episodes(project, [Subject("Dog")])
        self.assertEqual(migrated.ethogram.subject_names, ["Dog"])
        self.assertTrue(report)
