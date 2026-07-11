import os
import sys
import uuid

MODELS = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models"


def check_torchtitan_import():
    try:
        import torchtitan  # noqa: F401
        return ("torchtitan import", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return ("torchtitan import", False, repr(exc))


def check_torchtitan_train():
    try:
        import torchtitan.train  # noqa: F401
        return ("torchtitan.train import", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return ("torchtitan.train import", False, repr(exc)[:160])


def check_tokenizer(path=f"{MODELS}/Llama-3.1-8B-Instruct"):
    ok = os.path.isdir(path) and os.access(path, os.R_OK)
    return (f"tokenizer:{path}", ok, "ok" if ok else "unreadable")


def check_dir_writable(path):
    try:
        probe = os.path.join(path, f".pf_{uuid.uuid4().hex}")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return (f"writable:{path}", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return (f"writable:{path}", False, repr(exc))


def check_gpu_visible():
    try:
        import torch
        n = torch.cuda.device_count()
        return ("gpu visible", n > 0, f"device_count={n}")
    except Exception as exc:  # noqa: BLE001
        return ("gpu visible", False, repr(exc))


def run_checks():
    return [
        check_torchtitan_import(),
        check_torchtitan_train(),
        check_tokenizer(),
        check_dir_writable("outputs"),
        check_gpu_visible(),
    ]


def main():
    results = run_checks()
    failed = 0
    train_import_failed = False
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        if name == "torchtitan.train import" and not ok:
            train_import_failed = True
        print(f"[{status}] {name}: {detail}")
    if train_import_failed:
        print("\n'torchtitan.train import' failed — this is the torch 2.10 gate. "
              "Rebuild the container (torch>=2.11) per container/dtitan.def.")
    if failed:
        print(f"\n{failed} check(s) failed. (A 'gpu visible' failure is expected on a "
              f"login node — run preflight inside a GPU allocation.)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
