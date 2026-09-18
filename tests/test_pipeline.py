"""Job manifests, sharding, engine matching and the full offline round trip."""

import json
import pathlib
import tempfile
import unittest

from abcoder.common.boris import BorisProject, ethogram_from_table, resolve_media_path
from abcoder.common.config import ENGINE_PRESETS, engine_defaults, suggest_ownership
from abcoder.common.ethogram import EPISODE_CATEGORY, Ethogram, Subject
from abcoder.common.events import MediaInfo
from abcoder.common.jobspec import (
    STAGING_SHARED,
    EngineSpec,
    JobLayout,
    JobSpec,
    ObservationSpec,
    SlurmSpec,
)
from abcoder.common.media import duplicate_observation_ids, observation_id_for, sample_times
from abcoder.common.paths import ProbeToken, map_to_client, map_to_server
from abcoder.common.project_builder import write_project
from abcoder.server.engines import EngineContext, available_engines
from abcoder.server.runner import main as runner_main
from abcoder.server.slurm import plan_shards, render_sbatch
from tests import fixtures

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = fixtures.shared()


class TestObservationIds(unittest.TestCase):
    def test_id_is_the_stem(self):
        self.assertEqual(observation_id_for("/a/b/03_30_152.MP4"), "03_30_152")

    def test_case_differing_extensions_collide(self):
        paths = [pathlib.Path("/a/clip.mp4"), pathlib.Path("/a/clip.MP4")]
        self.assertIn("clip", duplicate_observation_ids(paths))


class TestSampling(unittest.TestCase):
    def test_respects_the_cap(self):
        self.assertEqual(len(sample_times(0, 600, 1.0, 96)), 96)

    def test_covers_the_whole_window_when_capped(self):
        times = sample_times(0, 600, 1.0, 96)
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 600.0, places=1)

    def test_short_window_is_not_over_sampled(self):
        self.assertEqual(len(sample_times(0, 4, 1.0, 96)), 5)


class TestSharding(unittest.TestCase):
    def test_one_task_per_video_by_default(self):
        self.assertEqual(len(plan_shards(52, 1)), 52)

    def test_packs_into_fewer_tasks_above_the_array_limit(self):
        shards = plan_shards(5000, 1, max_tasks=1000)
        self.assertLessEqual(len(shards), 1000)
        self.assertEqual(shards[-1][1], 5000)

    def test_shards_cover_every_index_exactly_once(self):
        covered = []
        for start, stop in plan_shards(37, 4):
            covered.extend(range(start, stop))
        self.assertEqual(covered, list(range(37)))


class TestJobSpec(unittest.TestCase):
    def _job(self):
        etho = ethogram_from_table(FIXTURES["ethogram_xlsx"])
        etho.subjects = [Subject("Dog")]
        job = JobSpec(project_name="t", ethogram=etho, staging=STAGING_SHARED)
        job.observations = [ObservationSpec("a", server_media=["/x/a.mp4"])]
        job.engines = [EngineSpec("mock")]
        return job

    def test_round_trips_through_json(self):
        job = self._job()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "job.json"
            job.save(path)
            reloaded = JobSpec.load(path)
        self.assertEqual(reloaded.project_name, "t")
        self.assertEqual(len(reloaded.ethogram), len(job.ethogram))
        self.assertEqual(reloaded.observations[0].observation_id, "a")

    def test_validation_catches_the_common_mistakes(self):
        job = self._job()
        job.observations = []
        self.assertIn("No videos selected.", job.validate())

        job = self._job()
        job.ownership = {"Growl": "audio"}
        self.assertTrue(any("not enabled" in p for p in job.validate()))

    def test_future_protocol_version_is_refused(self):
        raw = self._job().to_dict()
        raw["protocol_version"] = 99
        with self.assertRaises(ValueError):
            JobSpec.from_dict(raw)

    def test_layout_sanitises_observation_ids(self):
        layout = JobLayout("/tmp/x")
        self.assertEqual(layout.result_file("a/b c", "mock").name, "a_b_c.mock.json")


class TestSbatch(unittest.TestCase):
    def test_script_carries_the_resource_request(self):
        job = JobSpec(job_id="jid")
        job.observations = [ObservationSpec(f"o{i}", server_media=[f"/x/{i}.mp4"])
                            for i in range(10)]
        engine = EngineSpec("vlm_vllm", slurm=SlurmSpec(
            partition="extralarge", gres="gpu:a100:1", nodelist="nipg38",
            array_throttle=3))
        job.engines = [engine]
        with tempfile.TemporaryDirectory() as d:
            layout = JobLayout(d).ensure()
            text = render_sbatch(job, engine, layout, plan_shards(10, 1),
                                 python="/v/bin/python", abc_root="/abc")
        self.assertIn("#SBATCH --gres=gpu:a100:1", text)
        self.assertIn("#SBATCH --array=0-9%3", text)
        self.assertIn("#SBATCH --nodelist=nipg38", text)
        self.assertIn("abcoder.server.runner", text)
        self.assertIn("set -euo pipefail", text)


class TestLlamaCppRuntimeDiscovery(unittest.TestCase):
    """The binary needs CUDA libraries the compute nodes do not have.

    llama.cpp links libcudart/libcublas dynamically. A compute node has the
    driver but no CUDA toolkit, so a binary built on a login node fails to
    start unless the runtime travels with it.
    """

    def _engine(self, **options):
        from abcoder.server.engines.vlm_llamacpp import LlamaCppEngine

        return LlamaCppEngine(EngineContext(ethogram=Ethogram(), options=options))

    def test_finds_the_lib_directory_beside_the_binary(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "bin").mkdir()
            (root / "lib").mkdir()
            binary = root / "bin" / "llama-server"
            binary.touch()
            dirs = self._engine()._library_dirs(str(binary))
        self.assertEqual([p.name for p in dirs], ["lib"])

    def test_no_lib_directory_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            binary = pathlib.Path(d) / "bin" / "llama-server"
            binary.parent.mkdir()
            binary.touch()
            self.assertEqual(self._engine()._library_dirs(str(binary)), [])

    def test_an_explicit_directory_takes_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "bin").mkdir()
            (root / "lib").mkdir()
            (root / "custom").mkdir()
            binary = root / "bin" / "llama-server"
            binary.touch()
            dirs = self._engine(cuda_lib_dir=str(root / "custom"))._library_dirs(str(binary))
        self.assertEqual([p.name for p in dirs], ["custom", "lib"])


class TestNewJobCommand(unittest.TestCase):
    """`abc new-job` -- the step between scanning a folder and submitting."""

    def _run(self, *args) -> dict:
        import io
        import sys as _sys

        from abcoder.server.cli import build_parser

        parsed = build_parser().parse_args(["new-job", *args])
        stdout, _sys.stdout = _sys.stdout, io.StringIO()
        try:
            parsed.func(parsed)
            return json.loads(_sys.stdout.getvalue())
        finally:
            _sys.stdout = stdout

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.videos = pathlib.Path(self.tmp.name) / "videos"
        self.videos.mkdir()
        for name in ("03_30_152.mp4", "04_05_162.mp4"):
            (self.videos / name).write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_a_submittable_job(self):
        out = self._run(
            "--videos", str(self.videos),
            "--ethogram", str(FIXTURES["ethogram_xlsx"]),
            "--engine", "mock", "--subject", "Dog",
            "--job-dir", str(pathlib.Path(self.tmp.name) / "job"),
            "--no-probe")
        self.assertEqual(out["observations"], 2)
        self.assertEqual(out["problems"], [])
        job = JobSpec.load(pathlib.Path(out["job_dir"]) / "job.json")
        self.assertEqual(job.validate(), [])
        self.assertEqual(sorted(o.observation_id for o in job.observations),
                         ["03_30_152", "04_05_162"])

    def test_episode_prefix_moves_phases_out_of_the_subject_field(self):
        out = self._run(
            "--videos", str(self.videos),
            "--ethogram", str(FIXTURES["ethogram_xlsx"]),
            "--engine", "mock", "--subject", "Dog",
            "--episode-prefix", "Episode",
            "--job-dir", str(pathlib.Path(self.tmp.name) / "job2"),
            "--no-probe")
        self.assertEqual(len(out["episode_behaviors"]), 4)
        job = JobSpec.load(pathlib.Path(out["job_dir"]) / "job.json")
        self.assertEqual(job.ethogram.subject_names, ["Dog"])
        for code in out["episode_behaviors"]:
            behavior = job.ethogram.get(code)
            self.assertEqual(behavior.category, EPISODE_CATEGORY)
            self.assertTrue(behavior.is_state)

    def test_colliding_observation_ids_are_refused(self):
        (self.videos / "03_30_152.MOV").write_bytes(b"x")
        out = self._run(
            "--videos", str(self.videos),
            "--ethogram", str(FIXTURES["ethogram_xlsx"]),
            "--engine", "mock",
            "--job-dir", str(pathlib.Path(self.tmp.name) / "job3"),
            "--no-probe")
        self.assertIn("error", out)
        self.assertIn("03_30_152", out["duplicate_observation_ids"])

    def test_ownership_can_be_prefilled(self):
        out = self._run(
            "--videos", str(self.videos),
            "--ethogram", str(FIXTURES["ethogram_xlsx"]),
            "--engine", "audio", "--engine", "pose", "--own",
            "--job-dir", str(pathlib.Path(self.tmp.name) / "job4"),
            "--no-probe")
        self.assertEqual(out["ownership"]["Growl"], "audio")
        self.assertEqual(out["ownership"]["Approaching robot"], "pose")


class TestSubmissionPreflight(unittest.TestCase):
    """Catch the deployment mistakes that make every array task fail alike."""

    def test_node_local_roots_are_recognised(self):
        from abcoder.server.slurm import node_local

        self.assertTrue(node_local("/tmp/checkout"))
        self.assertTrue(node_local("/dev/shm/x"))
        self.assertFalse(node_local(str(pathlib.Path.home())))

    def test_a_checkout_on_node_local_storage_is_refused(self):
        from abcoder.server.slurm import preflight

        problems = preflight(pathlib.Path("/tmp/job"), "/tmp/abc", "/usr/bin/python3")
        self.assertTrue(problems)
        self.assertTrue(any("node-local" in p for p in problems))

    def test_a_missing_interpreter_is_reported(self):
        from abcoder.server.slurm import preflight

        with tempfile.TemporaryDirectory(dir=pathlib.Path.home()) as d:
            problems = preflight(pathlib.Path(d), d, str(pathlib.Path(d) / "nope"))
        self.assertTrue(any("does not exist" in p for p in problems))

    def test_submit_refuses_rather_than_queueing_a_doomed_job(self):
        from abcoder.server.slurm import SlurmError, submit

        job = JobSpec(job_id="jid")
        job.observations = [ObservationSpec("o", server_media=["/x/o.mp4"])]
        job.engines = [EngineSpec("mock")]
        with tempfile.TemporaryDirectory() as d:
            layout = JobLayout(pathlib.Path(d) / "job").ensure()
            # A dry run reports the problem but still writes the scripts:
            # inspecting them is the whole point of the mode.
            _, warnings = submit(job, layout, python="/usr/bin/python3",
                                 abc_root="/tmp/abc", dry_run=True)
            self.assertTrue(any("node-local" in w for w in warnings))

            # A real submission refuses rather than occupying the queue.
            with self.assertRaises(SlurmError) as ctx:
                submit(job, layout, python="/usr/bin/python3", abc_root="/tmp/abc")
        self.assertIn("node-local", str(ctx.exception))

    def test_the_script_pins_its_working_directory(self):
        job = JobSpec(job_id="jid")
        job.observations = [ObservationSpec("o", server_media=["/x/o.mp4"])]
        engine = EngineSpec("mock")
        job.engines = [engine]
        with tempfile.TemporaryDirectory() as d:
            layout = JobLayout(d).ensure()
            text = render_sbatch(job, engine, layout, plan_shards(1, 1),
                                 python="/v/bin/python", abc_root="/abc")
        # Otherwise the task inherits a submitting directory the node may not have.
        self.assertIn(f"#SBATCH --chdir={layout.root}", text)


class TestEngineMatching(unittest.TestCase):
    def setUp(self):
        self.etho = ethogram_from_table(FIXTURES["ethogram_xlsx"])

    def test_audio_claims_exactly_the_vocalisations(self):
        from abcoder.server.engines.audio import AudioEngine

        engine = AudioEngine(EngineContext(ethogram=self.etho))
        self.assertEqual(
            [b.code for b in engine._vocal_behaviors()],
            ["Whine", "Excited bark", "Aggressive bark", "Growl", "Puffing"])

    def test_audio_distinguishes_the_two_barks(self):
        from abcoder.server.engines.audio import AudioEngine

        engine = AudioEngine(EngineContext(ethogram=self.etho))
        classes = {b.code: engine._canonical_class(b) for b in engine._vocal_behaviors()}
        self.assertEqual(classes["Excited bark"], "bark_high")
        self.assertEqual(classes["Aggressive bark"], "bark_low")

    def test_pose_rules_ignore_subordinate_clauses(self):
        from abcoder.server.engines.pose import PoseEngine

        engine = PoseEngine(EngineContext(ethogram=self.etho))
        rules = {b.code: r for b, r in engine._assign_rules()}
        # "Backing" is described as "...while orienting at the agent"; matching
        # the description first would code a retreat as an orientation.
        self.assertEqual(rules["Backing"], "withdraw_robot")
        self.assertEqual(rules["Orienting at owner"], "orient_person")
        self.assertEqual(rules["Approaching owner"], "approach_person")
        self.assertEqual(rules["Approaching robot"], "approach_robot")

    def test_ownership_suggestion_splits_the_ethogram_sensibly(self):
        owners = suggest_ownership(self.etho.codes, ["vlm_vllm", "audio", "pose"])
        self.assertEqual(owners["Growl"], "audio")
        self.assertEqual(owners["Approaching robot"], "pose")
        self.assertNotIn("Tail wagging", owners)  # left to the VLM

    def test_every_preset_has_a_loadable_class(self):
        for name in ENGINE_PRESETS:
            self.assertIn(name, available_engines())


class TestVlmPrompting(unittest.TestCase):
    def setUp(self):
        self.etho = ethogram_from_table(FIXTURES["ethogram_xlsx"])
        self.etho.subjects = [Subject("Dog")]

    def test_schema_constrains_behaviours_to_the_ethogram(self):
        from abcoder.server.engines.vlm_common import events_schema

        schema = events_schema(self.etho)
        enum = schema["properties"]["events"]["items"]["properties"]["behavior"]["enum"]
        self.assertEqual(sorted(enum), sorted(self.etho.codes))

    def test_windows_overlap_and_cover_the_video(self):
        from abcoder.server.engines.vlm_common import plan_windows

        windows = plan_windows(500.0, 120.0, 5.0)
        self.assertEqual(windows[0][0], 0.0)
        self.assertEqual(windows[-1][1], 500.0)
        for (_, prev_stop), (next_start, _) in zip(windows, windows[1:], strict=False):
            self.assertLess(next_start, prev_stop)

    def test_response_parsing_survives_fences_and_prose(self):
        from abcoder.server.engines.vlm_common import Window, parse_response

        window = Window(0, 0.0, 120.0, [])
        text = ('Here is the coding:\n```json\n{"events":[{"behavior":"Growl",'
                '"subject":"Dog","start":12.5,"stop":null,"confidence":0.8}]}\n```')
        events, _ = parse_response(text, self.etho, window)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior, "Growl")

    def test_window_relative_times_are_shifted_not_discarded(self):
        from abcoder.server.engines.vlm_common import Window, parse_response

        window = Window(2, 240.0, 360.0, [])
        text = '{"events":[{"behavior":"Growl","subject":"Dog","start":10.0,"confidence":0.5}]}'
        events, warnings = parse_response(text, self.etho, window)
        self.assertEqual(events[0].start, 250.0)
        self.assertTrue(any("window-relative" in w for w in warnings))

    def test_a_point_event_cannot_carry_a_duration(self):
        from abcoder.server.engines.vlm_common import Window, parse_response

        window = Window(0, 0.0, 120.0, [])
        text = '{"events":[{"behavior":"Growl","subject":"Dog","start":5,"stop":9,"confidence":1}]}'
        events, _ = parse_response(text, self.etho, window)
        self.assertIsNone(events[0].stop)

    def test_voting_needs_agreement(self):
        from abcoder.common.events import Event
        from abcoder.server.engines.vlm_common import vote

        samples = [[Event("Growl", 10.0, confidence=0.9)],
                   [Event("Growl", 10.5, confidence=0.7)],
                   [Event("Bark", 40.0, confidence=0.9)]]
        kept = vote(samples, min_votes=2)
        self.assertEqual([e.behavior for e in kept], ["Growl"])


class TestPathProbe(unittest.TestCase):
    def test_probe_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            token = ProbeToken.write(d)
            self.assertTrue(token.verify())
            token.cleanup()
            self.assertFalse(token.verify())

    def test_mount_point_translation(self):
        client = map_to_client("/nas/home/u/v/a.mp4", "/nas/home/u", "/Volumes/nas")
        self.assertEqual(client, "/Volumes/nas/v/a.mp4")
        self.assertEqual(map_to_server(client, "/Volumes/nas", "/nas/home/u"),
                         "/nas/home/u/v/a.mp4")

    def test_unrelated_paths_pass_through(self):
        self.assertEqual(map_to_client("/other/a.mp4", "/nas", "/Volumes"), "/other/a.mp4")


class TestOfflineRoundTrip(unittest.TestCase):
    """Job -> mock engine -> results -> fused BORIS project, with real paths."""

    def test_full_round_trip(self):
        etho = ethogram_from_table(FIXTURES["ethogram_xlsx"])
        etho.subjects = [Subject("Dog"), Subject("Owner")]
        for b in etho.behaviors:
            if b.code.lower().startswith("episode"):
                b.category = EPISODE_CATEGORY
        etho.categories.append(EPISODE_CATEGORY)

        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            videos = base / "client" / "Merged"
            videos.mkdir(parents=True)
            names = ["03_30_152.mp4", "04_05_162.mp4"]
            for name in names:
                (videos / name).write_bytes(b"not really a video")

            job_dir = base / "job"
            layout = JobLayout(job_dir).ensure()
            job = JobSpec(project_name="round trip", ethogram=etho,
                          staging=STAGING_SHARED, job_dir=str(job_dir))
            defaults = engine_defaults("mock")
            job.engines = [EngineSpec("mock", options=defaults["options"],
                                      slurm=SlurmSpec.from_dict(defaults["slurm"]))]
            for name in names:
                path = videos / name
                info = MediaInfo(path=str(path), duration=300.0, fps=25.0,
                                 has_video=True, has_audio=True, width=1280, height=720)
                job.observations.append(ObservationSpec(
                    observation_id=path.stem, server_media=[str(path)],
                    client_media=[str(path)], info=[info]))
            job.save(layout.manifest)
            self.assertEqual(job.validate(), [])

            for index in range(len(names)):
                code = runner_main(["--job-dir", str(job_dir), "--engine", "mock",
                                    "--shard-index", str(index), "--per-task", "1"])
                self.assertEqual(code, 0)

            project_path = base / "client" / "coded.boris"
            report = write_project(job, layout, project_path)
            self.assertEqual(report.observations_written, 2)
            self.assertGreater(report.total_events, 0)
            self.assertEqual(report.missing_media, [])

            project = BorisProject.load(project_path)
            self.assertEqual(sorted(project.observation_ids), ["03_30_152", "04_05_162"])
            self.assertEqual(project.ethogram.subject_names, ["Dog", "Owner"])

            stored = project.observation_media("03_30_152")[0]
            self.assertEqual(stored, "Merged/03_30_152.mp4")
            self.assertTrue(resolve_media_path(stored, project_path).exists())

            events = project.observation_events("03_30_152")
            self.assertTrue(events)
            for e in events:
                self.assertTrue(etho.has(e.behavior))
                if e.behavior.lower().startswith("episode"):
                    self.assertEqual(e.subject, "")
                else:
                    self.assertIn(e.subject, ["Dog", "Owner"])
                self.assertIn("[mock", e.comment)

            # The frame column must match the media's frame rate.
            row = project.data["observations"]["03_30_152"]["events"][0]
            self.assertEqual(row[5], int(round(row[0] * 25.0)))


if __name__ == "__main__":
    unittest.main()
