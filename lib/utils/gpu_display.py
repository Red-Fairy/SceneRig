"""An owned NVIDIA X display for a pipeline phase on one pinned physical GPU."""

import json
import os
import secrets
import select
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def _gpu_bus_id(gpu, expected_uuid):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("GPU display requires exactly one physical GPU")
    uuid, pci = (part.strip() for part in rows[0].split(","))
    if uuid != expected_uuid:
        raise RuntimeError(
            f"GPU display UUID mismatch: expected {expected_uuid}, observed {uuid}"
        )
    domain, bus, slot = pci.split(":")
    device, function = slot.split(".")
    bus = str(int(bus, 16))
    if int(domain, 16):
        bus += f"@{int(domain, 16)}"
    return f"PCI:{bus}:{int(device, 16)}:{int(function, 16)}"


def _module_paths():
    roots = [Path("/usr/lib64/xorg/modules"), Path("/usr/lib/xorg/modules")]
    nvidia = next(
        (
            p
            for p in roots
            if (p / "drivers/nvidia_drv.so").is_file()
            and (p / "extensions/libglxserver_nvidia.so").is_file()
        ),
        None,
    )
    if nvidia is None:
        raise RuntimeError(
            "NVIDIA Xorg modules missing from the supported system module paths"
        )
    return [nvidia] + [p for p in roots if p != nvidia and p.is_dir()]


def _config(bus_id, module_paths):
    modules = "\n".join(f'    ModulePath "{p}"' for p in module_paths)
    return f'''Section "ServerFlags"
    Option "AutoAddDevices" "false"
    Option "AutoAddGPU" "false"
EndSection
Section "Files"
{modules}
EndSection
Section "Device"
    Identifier "GraseGPU"
    Driver "nvidia"
    BusID "{bus_id}"
    Option "AllowEmptyInitialConfiguration" "true"
    Option "ProbeAllGpus" "false"
EndSection
Section "Screen"
    Identifier "GraseScreen"
    Device "GraseGPU"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual 1024 1024
    EndSubSection
EndSection
Section "ServerLayout"
    Identifier "GraseLayout"
    Screen 0 "GraseScreen"
EndSection
'''


@contextmanager
def gpu_display(
    gpu, expected_uuid, output_dir, env, *, startup_timeout=20, shutdown_timeout=10
):
    """Yield a copied phase environment; close/reap only its Xorg on every exit.

    Xorg selects a free display atomically with ``-displayfd``. Its authorization
    file contains a private cookie before startup; after allocation, the same
    cookie gets a client record for the chosen display. No global env is changed.
    """
    bus_id = _gpu_bus_id(gpu, expected_uuid)
    xorg = shutil.which("Xorg")
    if xorg is None:
        raise RuntimeError("Xorg missing; run scripts/install_system_libs.sh")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "xorg.conf").write_text(_config(bus_id, _module_paths()))
    metadata = {"gpu": str(gpu), "uuid": expected_uuid, "bus_id": bus_id}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="auth-", dir=out) as private:
        private = Path(private)
        authority = private / "Xauthority"
        authority.touch(mode=0o600)
        cookie = secrets.token_hex(16)

        def authorize(display):
            subprocess.run(
                ["xauth", "-f", str(authority), "source", "-"],
                input=f"add {display} MIT-MAGIC-COOKIE-1 {cookie}\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
            )

        authorize(":0")
        child_env = dict(env, XAUTHORITY=str(authority))
        read_fd, write_fd = os.pipe()
        proc = None
        try:
            with (out / "xorg.stderr.log").open("w") as log:
                proc = subprocess.Popen(
                    [
                        xorg,
                        "-displayfd",
                        str(write_fd),
                        "-config",
                        str(out / "xorg.conf"),
                        "-configdir",
                        str(private),
                        "-logfile",
                        str(out / "Xorg.log"),
                        "-auth",
                        str(authority),
                        "-nolisten",
                        "tcp",
                        "-noreset",
                        "-novtswitch",
                        "-sharevts",
                        "-isolateDevice",
                        bus_id,
                    ],
                    stdout=log,
                    stderr=log,
                    env=child_env,
                    pass_fds=(write_fd,),
                )
            metadata["pid"] = proc.pid
            (out / "display.json").write_text(json.dumps(metadata, indent=2) + "\n")
            os.close(write_fd)
            write_fd = None
            if not select.select([read_fd], [], [], startup_timeout)[0]:
                raise TimeoutError(
                    f"Xorg did not become ready within {startup_timeout}s; see {out}"
                )
            number = os.read(read_fd, 64).decode().strip()
            if not number.isdecimal() or proc.poll() is not None:
                raise RuntimeError(f"Xorg exited before selecting a display; see {out}")
            display = f":{number}"
            authorize(display)
            child_env["DISPLAY"] = display
            metadata.update(display=display, ready_seconds=time.monotonic() - started)
            (out / "display.json").write_text(json.dumps(metadata, indent=2) + "\n")
            yield child_env
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=shutdown_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                metadata["exit_code"] = proc.returncode
            metadata["total_seconds"] = time.monotonic() - started
            (out / "display.json").write_text(json.dumps(metadata, indent=2) + "\n")
