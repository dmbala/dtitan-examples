import os
import sys
import uuid


def check_torch_import():
    try:
        import torch
        return ("torch import", True, torch.__version__)
    except Exception as exc:  # noqa: BLE001
        return ("torch import", False, repr(exc))


def check_dtensor_import():
    try:
        from torch.distributed.tensor import Shard  # noqa: F401
        from torch.distributed.device_mesh import init_device_mesh  # noqa: F401
        return ("dtensor import", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return ("dtensor import", False, repr(exc))


def check_dir_writable(path):
    try:
        probe = os.path.join(path, f".preflight_{uuid.uuid4().hex}")
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


def run_cpu_checks(dirs):
    rows = [check_torch_import(), check_dtensor_import()]
    rows += [check_dir_writable(d) for d in dirs]
    return rows


def main():
    dirs = ["/data", "/checkpoints", "/artifacts"]
    rows = run_cpu_checks(dirs)
    rows.append(check_gpu_visible())
    failed = 0
    for name, ok, detail in rows:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
