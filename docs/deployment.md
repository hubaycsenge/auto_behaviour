# Deploying on NIPG

## The hardware, and what it means

Compute capability decides which engine can run where. There are two cliff
edges and it is worth knowing exactly where they are.

| nodes | GPUs | VRAM | cc | bf16 / FA2 | vLLM | use for |
|---|---|---|---|---|---|---|
| nipg38 | 8 × A100 | 40 GB | 8.0 | yes | yes | `vlm_vllm` — the workhorse, 320 GB aggregate |
| nipg10 | 2 × RTX 3090 | 24 GB | 8.6 | yes | yes | `vlm_vllm` secondary, `pose` |
| nipg32 | 4 × A4000 | 16 GB | 8.6 | yes | yes | 7B/8B VLMs, `pose`, parallel workers |
| nipg7, 34–36 | 2 × Titan RTX | 24 GB | 7.5 | **no** | effectively no | `vlm_llamacpp`, `pose`, `audio` |
| nipg6, 30, 31, 33 | 3–4 × 2080 Ti | 11 GB | 7.5 | **no** | effectively no | `audio`, `pose`, small GGUF VLMs |
| nipg3–5 | 2 × GTX 1080 | 8 GB | 6.1 | no | **cannot** | `audio`, light CV, llama.cpp |

Partitions: `small` (default, every node), `medium`, `large`, `extralarge`
(nipg38 only). `MaxArraySize` is 1001.

**Turing (cc 7.5).** No bfloat16 and no FlashAttention-2. Qwen2.5-VL runs under
vLLM only with `--dtype=half`, and can crash from FP16 overflow because the
weights were trained in bf16. Qwen3-VL's vision path does not work there at all
— upstream closed the support request as *not planned*. ABC's `vlm_vllm` engine
refuses to start on these nodes with a message saying so; that is deliberate,
not a bug.

**Pascal (cc 6.1).** Below vLLM's floor entirely, and dropped by CUDA 13.x.
llama.cpp still supports it.

## Installing

```bash
git clone <repo> ~/abc && cd ~/abc

# core: mock engine, BORIS handling, SLURM submission, media probing
setup/install_server.sh

# heavy extras — run each from a node with the GPU you intend to use, so pip
# picks a matching CUDA wheel
srun -p extralarge --nodelist=nipg38 --gres=gpu:a100:1 --pty setup/install_server.sh --vllm
srun -p medium --gres=gpu:1 --pty setup/install_server.sh --audio --pose

source ~/abc_env/abc-env.sh
abc check
```

`setup/install_server.sh` bootstraps pip by hand: these nodes have no
`python3-venv`, so `python3 -m venv` fails on `ensurepip` and the script falls
back to `get-pip.py`. There is also **no system ffmpeg anywhere on the
cluster** — ABC depends on PyAV, which carries its own, and nothing in the
codebase may shell out to an ffmpeg binary.

For the llama.cpp engine — **on the login node, not under `srun`**:

```bash
setup/install_llamacpp.sh     # builds for sm_61;75;80;86
setup/fetch_gguf.sh           # model + its mmproj projector
```

The CUDA toolkit is installed on nipg1 only: a compute node has `gcc` and
`make` but neither `nvcc` nor `git`, so a CUDA build there is impossible. The
binary this produces targets every GPU architecture in the cluster, so it runs
on the compute nodes afterwards.

The script also works around two things this cluster does not provide. There
is no system `cmake`, so it falls back to one installed from PyPI into the ABC
virtualenv. And CUDA 12.0 refuses any host compiler newer than gcc 12 while
the default here is gcc 13, so it reads the limit from the toolkit's own
`host_config.h` and points `CMAKE_CUDA_HOST_COMPILER` at the installed
`g++-12`.

## Client

```bash
setup/install_client.sh          # PySide6-Essentials, PyAV, Pillow, numpy
setup/install_client.sh --ssh    # plus paramiko, for a machine with no /nas mount
bin/abc-gui
```

`PySide6-Essentials` rather than full `PySide6`: ABC uses QtWidgets only, and
the full package adds several hundred megabytes of WebEngine and 3D modules it
never touches.

**The client and server environments are separate.** The server needs vLLM,
torch and codecs but no Qt; the client needs Qt but none of the rest. `abc`
runs in the server environment, `abc-gui` in the client one, and each finds its
own — `abc-gui` does not defer to `abc`, or sourcing `abc-env.sh` would launch
the GUI with an interpreter that has no PySide6.

If the GUI reports `No module named 'PySide6'` after a successful-looking
install, the usual cause is pip having installed into a *different* Python — a
conda base environment, typically, since `pip` there is not the virtualenv's
pip. `setup/install_client.sh` now verifies the import before reporting success
and tells you the explicit command if it did not work:

```bash
.venv-client/bin/python -m pip install -r requirements-client.txt
```

Without paramiko the SSH transport falls back to the `ssh` and `scp` commands.
That works, but gives no upload progress and needs key-based auth —
`ssh-copy-id` first, because ABC never prompts for a password inside a command.

## Configuration

`~/.config/abc/client.json` and `~/.config/abc/server.json`, deep-merged over
the defaults in `abcoder/common/config.py`. `$ABC_CONFIG` overrides the path.

```json
{
  "client": {
    "ssh_host": "nipg1.inf.elte.hu",
    "ssh_user": "yourname",
    "server_abc": "/nas/home/yourname/abc/bin/abc",
    "shared_roots": ["~/", "/nas/home", "/Volumes/nipgnas1"],
    "poll_seconds": 15
  },
  "server": {
    "jobs_root": "~/abc_jobs",
    "venv": "~/abc_env",
    "hf_home": "~/.cache/huggingface"
  },
  "engines": {
    "vlm_vllm": {
      "options": {"model": "Qwen/Qwen3-VL-8B-Instruct", "fps": 1.0},
      "slurm": {"partition": "extralarge", "nodelist": "nipg38", "array_throttle": 4}
    }
  }
}
```

### Everything the tasks touch must be on shared storage

Three things have to be readable from a compute node, not just from nipg1:

- **`jobs_root`** — the tasks read the manifest and write results there;
- **the ABC checkout itself** — the sbatch script puts it on `PYTHONPATH`, so a
  clone in `/tmp` gives every task `ModuleNotFoundError: No module named
  'abcoder'`;
- **the Python interpreter** — same reason.

`/nas/home` satisfies all three. `/tmp`, `/var/tmp`, `/run` and `/dev/shm` do
not: each node has its own, so a file written there on the login node simply
does not exist on nipg38.

`abc submit` checks this before queueing anything and refuses with an
explanation rather than letting the array run and fail identically forty times.

## Model weights

Set `hf_home` to a shared path so weights are downloaded once, not once per
array task. An 8B model is ~17 GB in FP16; forty tasks fetching it concurrently
is what actually saturates the NAS. Pre-warm the cache with a single-task run
before submitting a large array.

## Troubleshooting

**`abc check` says an engine is unavailable.** It reports what is installed on
the node you ran it from. Run it under `srun` on the target node — the login
node has a GTX 1080 Ti (compute capability 6.1) and is not where your jobs land.

`check` deliberately never *imports* the frameworks it reports on: it locates
packages with `importlib.util.find_spec` and reads the GPU from `nvidia-smi`.
Importing torch costs seconds on a warm NAS and stalls indefinitely on a
half-installed one, and the client runs `check` on every connect. Keep it that
way if you extend it.

**`No module named 'abcoder'` in a task log.** The checkout is somewhere the
compute node cannot see — almost always `/tmp`. Move it under your home
directory. `abc submit` refuses this case up front now, so an older job
directory is the usual way to still hit it.

**Tasks finish but write no results.** Look in `<job-dir>/logs/`. `abc status`
also consults `sacct` and tells you when tasks ended in a terminal failure
state, which is the usual sign of an out-of-memory kill or a wall-clock timeout.

**`vlm_vllm` says the GPU is too old.** It is, on that node. Either target
nipg38/10/32 with `nodelist`, or switch to `vlm_llamacpp`.

**`cmake is required`, or `nvcc` not found, when building llama.cpp.** You ran
it on a compute node. Only nipg1 has the CUDA toolkit and git; run
`setup/install_llamacpp.sh` directly on the login node.

**`llama-server did not become healthy`.** A 7B GGUF loading from cold NAS
storage can take minutes; the default `startup_timeout` is 600 s. If it exits
instead, `<job-dir>/logs/llama-server-*.log` has the reason — usually a
mismatched `mmproj`.

**Upload is slow.** Check whether it needs to happen at all. If the client can
see `/nas/home`, add its mount point to `shared_roots`; ABC will probe it and
skip the copy entirely.

**No media backend.** `pip install av` in whichever environment reported it.
ABC degrades gracefully — you can build an ethogram and pick videos without
codecs — but durations, frame rates and frame extraction all need PyAV.
