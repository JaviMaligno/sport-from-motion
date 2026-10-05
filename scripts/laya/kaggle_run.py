"""Kaggle runner for the Laya arm of sport-from-motion (see docs/preregistration-laya.md).

Spreads the jobs over the visible GPUs, one process per GPU, and writes one
prediction file + one .info.json per (model, condition, fold) to /kaggle/working/out.
"""
import glob
import json
import os
import subprocess
import sys
import time

FT_JOBS = os.environ.get("FT_JOBS", "")
ZS_JOBS = os.environ.get("ZS_JOBS", "")
FT_EXTRA = os.environ.get("FT_EXTRA", "")  # e.g. "--epochs 16 --schedule constant --tag e16" (DL1)
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

# Offline: the kernel has no DNS, so wheels and checkpoint come from a private dataset
# (laya 0.3.22, transformers 4.48.3, tokenizers 0.21.4, huggingface_hub 0.36.2 --
# the image ships huggingface_hub 1.x, which transformers 4.48 refuses; checkpoint at revision REVISION).
wheels = glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", *wheels], check=True)
import shutil  # noqa: E402
import torch  # noqa: E402

src = os.path.dirname(glob.glob("/kaggle/input/**/multilingual/rl_agent_config.json", recursive=True)[0])
model_dir = "/kaggle/working/laya-multilingual"
if not os.path.exists(model_dir):
    shutil.copytree(src, model_dir)
items = glob.glob("/kaggle/input/**/items.jsonl", recursive=True)[0]
print("wheels", [os.path.basename(w) for w in wheels], flush=True)
script = os.path.join(os.path.dirname(items), "train.py")
out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
n_gpu = torch.cuda.device_count()
print(f"GPUs: {n_gpu} | torch {torch.__version__} | items {items}", flush=True)
if n_gpu == 0:
    sys.exit("no GPU in this session: Kaggle only grants GPUs to phone-verified accounts")

ft = [j for j in FT_JOBS.split(",") if j]
zs = [j for j in ZS_JOBS.split(",") if j]
procs, t0 = [], time.time()
for g in range(n_gpu):
    mine_ft, mine_zs = ft[g::n_gpu], zs[g::n_gpu]
    base = f"{sys.executable} {script} --items {items} --model-dir {model_dir} --out {out} --device cuda:0"
    cmds = []
    if mine_ft:
        cmds.append(f"{base} {FT_EXTRA} --jobs {','.join(mine_ft)}")
    if mine_zs:
        cmds.append(f"{base} --zero-shot --jobs {','.join(mine_zs)}")
    if cmds:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(f"{out}/gpu{g}.log", "w")
        procs.append(subprocess.Popen(" && ".join(cmds), shell=True, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
shutil.rmtree(model_dir, ignore_errors=True)  # 647 MB that would otherwise ship with the output
print(f"exit codes {codes} | {time.time() - t0:.0f}s", flush=True)
for f in sorted(glob.glob(f"{out}/*.info.json")):
    print(open(f).read().replace("\n", " "), flush=True)
for g in range(n_gpu):
    if os.path.exists(f"{out}/gpu{g}.log"):
        print(f"--- gpu{g}.log tail ---")
        print("".join(open(f"{out}/gpu{g}.log").readlines()[-15:]))
if any(codes):
    sys.exit(f"jobs failed: exit codes {codes}")
