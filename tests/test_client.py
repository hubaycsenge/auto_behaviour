"""Client logic: transport selection, staging, and the controller's job build.

No Qt here -- everything the GUI does lives in the controller so it can be
tested headless.
"""

import io
import json
import pathlib
import sys
import tempfile
import unittest

from abcoder.client.controller import Controller
from abcoder.client.transport import filesystem_is_shared
from abcoder.client.transport.base import CommandResult, TransportError
from abcoder.client.transport.shared_fs import SharedFilesystemTransport
from abcoder.common.config import DEFAULT_CONFIG, deep_merge
from abcoder.common.ethogram import Subject
from abcoder.common.jobspec import STAGING_SHARED, STAGING_UPLOAD
from tests import fixtures

ROOT = pathlib.Path(__file__).resolve().parents[1]
ABC = str(ROOT / "bin" / "abc")
FIXTURES = fixtures.shared()


class TestCommandResult(unittest.TestCase):
    def test_json_ignores_a_login_banner(self):
        # A cluster .bashrc prints motd, module output and quota warnings before
        # anything we asked for.
        result = CommandResult(0, 'Welcome to nipg1\nQuota: 43% used\n{"ok": true}\n', "")
        self.assertEqual(result.json(), {"ok": True})

    def test_json_finds_a_bare_list(self):
        self.assertEqual(CommandResult(0, 'noise\n[1, 2]', "").json(), [1, 2])

    def test_empty_output_is_an_error_with_the_stderr(self):
        with self.assertRaises(TransportError) as ctx:
            CommandResult(1, "", "command not found: abc").json()
        self.assertIn("command not found", str(ctx.exception))


class TestCheckIsCheap(unittest.TestCase):
    """`abc check` runs on every client connect, so it must never block.

    It used to import torch to read the GPU's compute capability. On a warm NAS
    that costs seconds; on a half-installed torch it stalls indefinitely and
    takes the GUI's connect with it.
    """

    def test_availability_is_answered_without_importing(self):
        import sys as _sys

        from abcoder.server.cli import _module_present

        for name in ("vllm", "torch", "ultralytics"):
            _module_present(name)
            self.assertNotIn(name, _sys.modules,
                             f"checking for {name} must not import it")

    def test_check_is_fast(self):
        import time

        from abcoder.server.cli import cmd_check

        class Args:
            pass

        stdout, sys.stdout = sys.stdout, io.StringIO()
        try:
            started = time.time()
            cmd_check(Args())
            elapsed = time.time() - started
            payload = json.loads(sys.stdout.getvalue())
        finally:
            sys.stdout = stdout

        self.assertLess(elapsed, 10.0, "check must stay well under the client's timeout")
        self.assertIn("engines", payload)
        self.assertTrue(payload["engines"]["mock"]["available"])

    def test_gpu_capability_comes_from_nvidia_smi(self):
        import shutil as _shutil

        from abcoder.server.cli import _gpu_info

        info = _gpu_info()
        if not _shutil.which("nvidia-smi"):
            self.assertIsNone(info)
            return
        if info is None:
            self.skipTest("nvidia-smi present but reported no GPU")
        self.assertIn("compute_capability", info)
        # bf16 support is the line that decides vLLM versus llama.cpp.
        major = int(float(info["compute_capability"]))
        self.assertEqual(info["bf16"], major >= 8)


class TestSharedTransport(unittest.TestCase):
    def setUp(self):
        self.transport = SharedFilesystemTransport(abc_command=ABC, jobs_root="~/abc_jobs")

    def test_it_does_not_upload(self):
        self.assertFalse(self.transport.uploads)
        self.assertEqual(self.transport.kind, "shared")

    def test_remote_join_is_posix(self):
        self.assertEqual(self.transport.remote_join("/a/", "b/", "/c"), "/a/b/c")

    def test_put_file_to_itself_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "a.mp4"
            path.write_bytes(b"x")
            self.transport.put_file(path, str(path))
            self.assertEqual(path.read_bytes(), b"x")

    def test_abc_command_may_be_several_words(self):
        transport = SharedFilesystemTransport(
            abc_command=f"{ROOT}/bin/abc", jobs_root="~/abc_jobs")
        self.assertIn("abc_version", transport.check())

    def test_probe_confirms_a_shared_filesystem(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(filesystem_is_shared(self.transport, d))


class TestController(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        config = deep_merge(DEFAULT_CONFIG, {
            "client": {"server_abc": ABC},
            "server": {"jobs_root": str(pathlib.Path(self.tmp.name) / "jobs")},
        })
        self.controller = Controller(config)
        self.controller.connect()
        self.controller.set_engine_defaults(self.controller.available_engines())

    def tearDown(self):
        self.controller.close()
        self.tmp.cleanup()

    def _prepare(self):
        videos = pathlib.Path(self.tmp.name) / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        for name in ("03_30_152.mp4", "04_05_162.mp4"):
            (videos / name).write_bytes(b"x" * 1024)
        self.controller.load_ethogram(FIXTURES["ethogram_xlsx"])
        self.controller.state.ethogram.subjects = [Subject("Dog")]
        self.controller.scan_source(videos)
        self.controller.state.engines["mock"].enabled = True
        self.controller.state.project_name = "test"
        return videos

    def test_connects_to_the_local_cluster(self):
        self.assertEqual(self.controller.transport.kind, "shared")
        self.assertFalse(self.controller.uploads_required)
        self.assertIn("mock", self.controller.server_info["engines"])

    def test_scan_derives_observation_ids_from_file_names(self):
        self._prepare()
        self.assertEqual([v.observation_id for v in self.controller.state.videos],
                         ["03_30_152", "04_05_162"])

    def test_shared_filesystem_job_reads_videos_in_place(self):
        videos = self._prepare()
        job = self.controller.build_job()
        self.assertEqual(job.staging, STAGING_SHARED)
        self.assertEqual(job.observations[0].server_media[0],
                         str((videos / "03_30_152.mp4").resolve()))
        self.assertEqual(self.controller.upload(), 0)

    def test_manifest_round_trips_through_the_server(self):
        self._prepare()
        self.controller.build_job()
        job_dir = self.controller.write_manifest()
        manifest = json.loads((pathlib.Path(job_dir) / "job.json").read_text())
        self.assertEqual(manifest["project_name"], "test")
        self.assertEqual(len(manifest["observations"]), 2)

    def test_purge_refuses_on_a_shared_filesystem(self):
        self._prepare()
        self.controller.build_job()
        self.controller.write_manifest()
        result = self.controller.purge_videos()
        self.assertEqual(result["deleted"], [])
        self.assertIn("already are", result["note"])

    def test_upload_staging_copies_and_then_purges(self):
        videos = self._prepare()
        # Force the upload path even though the filesystem is shared, to
        # exercise the branch a laptop client takes.
        self.controller.uploads_required = True
        job = self.controller.build_job()
        self.assertEqual(job.staging, STAGING_UPLOAD)
        self.controller.write_manifest()

        sent = self.controller.upload()
        self.assertEqual(sent, 2048)
        staged = pathlib.Path(job.job_dir) / "videos"
        self.assertEqual(sorted(p.name for p in staged.iterdir()),
                         ["03_30_152.mp4", "04_05_162.mp4"])

        result = self.controller.purge_videos()
        self.assertEqual(sorted(result["deleted"]), ["03_30_152.mp4", "04_05_162.mp4"])
        self.assertEqual(result["bytes_freed"], 2048)
        self.assertEqual(list(staged.iterdir()), [])
        # The originals must survive.
        self.assertTrue((videos / "03_30_152.mp4").exists())

    def test_submit_and_collect_end_to_end(self):
        videos = self._prepare()
        self.controller.build_job()
        self.controller.write_manifest()
        self.controller.submit(dry_run=True)

        from abcoder.server.runner import main as runner_main
        for index in range(2):
            runner_main(["--job-dir", self.controller.job_dir, "--engine", "mock",
                         "--shard-index", str(index), "--per-task", "1"])

        status = self.controller.status()
        self.assertEqual(status["done"], 2)

        target = videos / "out.boris"
        report = self.controller.collect(target)
        self.assertEqual(report.observations_written, 2)

        from abcoder.common.boris import BorisProject
        project = BorisProject.load(target)
        # Videos sit beside the project, so the stored path is a bare filename.
        self.assertEqual(project.observation_media("03_30_152"), ["03_30_152.mp4"])

    def test_marking_trial_phases_makes_them_state_events(self):
        self._prepare()
        self.controller.mark_episode_behaviors(["Episode1_start"])
        behavior = self.controller.state.ethogram.get("Episode1_start")
        self.assertEqual(behavior.category, "Episode")
        self.assertTrue(behavior.is_state)

    def test_attach_reopens_a_submitted_job(self):
        self._prepare()
        self.controller.build_job()
        job_dir = self.controller.write_manifest()

        fresh = Controller(self.controller.config)
        fresh.connect()
        job = fresh.attach(job_dir)
        self.assertEqual(job.project_name, "test")
        self.assertEqual(len(fresh.state.ethogram), 18)
        fresh.close()


if __name__ == "__main__":
    unittest.main()
