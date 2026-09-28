"""Physics-settle: drop each placed object onto its support to find a stable resting pose.

SAM3D reconstructs an object's shape and MoGE places it metrically, but neither adjusts how
the object *rests* — an articulated/flat object (e.g. an open laptop) can be left in an
unstable or wrong orientation. This module drops each placed object GLB onto its support and
reads the orientation it **settles** into, then ``incremental_settle`` bakes the cumulative
settle into a sibling ``<glb>_pm.glb`` placed mesh.

Settling runs on PhysX through the persistent Isaac Sim settle server
(``isaac/isaac_settle_server.py``) colliding CoACD convex parts
(``lib/tools/geometry/collision.py``). This is the engine + collider representation the
scenes are ultimately exported to (``isaac/blend_to_isaac.py``), so settled poses are at
rest **in the deployment engine** — critical for reset-heavy robotics use. The entry point
is ``incremental_settle``; the pre-2026-07-31 batch chain (``settle_hierarchical`` +
``apply_pose_matching`` + ``joint_settle``, over a one-shot ``isaac_settle_worker.py``,
plus a Bullet/convex-hull A/B backend) is gone — it had been default-unreachable and
frozen since the incremental path landed on 07-13.

Settling is **hierarchical and support-ordered** (a parent settles before its children):

  * root ↔ object (base case): the support is a large PASSIVE plane at z=0 (the table proxy;
    the initializer builds the real table top at z=0).
  * parent ↔ child (stacked): the support is the already-settled parent object.

Each object is released from hull-based first contact (+2 mm) and allowed to settle with its
full PhysX translation and rotation. A tilt above 45° (or >50 mm xy drift for a rollable)
opens the instability ladder: non-rollables try pristine and CoM/flatten stabilization before
accepting the fallen resting pose when nothing stands; rollables try a rollback/re-seat before
accepting their fallen pose. Per-object ICP then adjusts yaw/xy/scale, the object is re-dropped
so z is physically derived again, and the completed scene receives a free-sim certification
pass. See ``incremental_settle`` for the flip-accept and pristine-confirm short circuits.

The node-slug and oriented surface-slab helpers are shared with
``scripts/make_physics_scene.py`` (the press-Play viewer), which imports them.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lib.tools.geometry.agentic_mask import _binarize, slugify

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Shared scene-construction helpers (also used by make_physics_scene.py)       #
# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _node_slug(node: dict) -> str:
    cat, _, inst = node["id"].partition("#")  # node id is "category#k"
    return f"{slugify(cat)}_{inst or '0'}"


def _resize_mask(b: np.ndarray, h: int, w: int) -> np.ndarray:
    if b.shape == (h, w):
        return b
    from PIL import Image

    return np.asarray(Image.fromarray(b.astype("uint8") * 255).resize((w, h))) > 127


def surface_slab(
    world: np.ndarray, mask_path: Path, normal, thickness: float, margin: float
) -> Optional[list[list[float]]]:
    """8 world-space corners of an oriented thin slab fit to a surface's MoGE points.

    The slab's thickness axis is the RANSAC plane normal; the two in-plane axes + sizes
    come from PCA of the masked points projected into the plane. The slab is offset so its
    front/top face sits on the detected plane (objects rest on it, not inside it)."""
    h, w = world.shape[:2]
    m = _resize_mask(_binarize(np.load(mask_path)), h, w)
    pts = world[m & np.isfinite(world).all(axis=2)]
    if len(pts) < 30:
        return None
    n = _unit(np.asarray(normal, dtype=np.float64))
    c0 = np.median(pts, axis=0)
    d = pts - c0
    d_in = d - np.outer(d @ n, n)  # drop the normal component
    _, evecs = np.linalg.eigh(d_in.T @ d_in)  # ascending; [:,0]~normal, [:,1:]=in-plane
    e1, e2 = evecs[:, 2], evecs[:, 1]
    p1, p2 = d @ e1, d @ e2
    ext1, ext2 = (p1.max() - p1.min()) * margin, (p2.max() - p2.min()) * margin
    cen = c0 + e1 * (p1.max() + p1.min()) / 2 + e2 * (p2.max() + p2.min()) / 2
    cen = cen - n * (thickness / 2.0)  # front/top face lands on the plane
    hx, hy, hz = ext1 / 2.0, ext2 / 2.0, thickness / 2.0
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(
                    (cen + sx * hx * e1 + sy * hy * e2 + sz * hz * n).tolist()
                )
    return corners


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without Blender)                                   #
# --------------------------------------------------------------------------- #
def _ground_dz(lowest: float, support_top: float, gap: float) -> float:
    """z-shift that rests an object's ``lowest`` vertex ``gap`` above its ``support_top``.
    Positive lifts (out of penetration); negative drops it down onto the support."""
    return support_top + gap - lowest


def ancestor_chain(nodes_by_id: dict, node: Optional[dict]) -> list[str]:
    """Mesh names of every OBJECT above ``node`` in the support graph, nearest first;
    the chain stops at the first root surface (that's the z=0 slab in the sim)."""
    out: list[str] = []
    cur = nodes_by_id.get(node.get("parent")) if node else None
    while cur and cur.get("kind") != "root_surface":
        out.append(f"obj_{_node_slug(cur)}")
        cur = nodes_by_id.get(cur.get("parent"))
    return out


def support_mesh_names(nodes: list[dict]) -> set[str]:
    """Mesh names of every object that supports another object (scene-graph
    ``support``/``parent`` edges, root surfaces excluded). These get the finer
    SUPPORT_* CoACD budget in ``collision`` — a support's cavity is real contact
    geometry for its cargo, while a non-support only ever rests ON things."""
    by_id = {n["id"]: n for n in nodes}
    out: set[str] = set()
    for n in nodes:
        if n.get("kind") == "root_surface":
            continue
        for key in ("support", "parent"):
            p = by_id.get(n.get(key))
            if p is not None and p.get("kind") != "root_surface":
                out.add(f"obj_{_node_slug(p)}")
                break
    return out


def _semantic_container(category: Any) -> bool:
    """Whether an object needs cavity-preserving collision even before it has cargo."""
    words = set(str(category or "").lower().replace("_", " ").split())
    return bool(
        words
        & {
            "basket",
            "bin",
            "bowl",
            "bucket",
            "container",
            "cup",
            "dish",
            "mug",
            "pan",
            "pot",
            "tray",
        }
    )


def topo_by_support(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order objects so each one's support object precedes it (parents before children).

    Each obj has ``name`` and ``support`` (the parent object's name, or None for objects
    that rest on a root surface / the ground). Objects whose support is missing from the set
    are treated as ground-supported. A support cycle (shouldn't happen) breaks gracefully —
    the unresolved remainder is appended in input order."""
    names = {o["name"] for o in objs}
    ordered: list[dict[str, Any]] = []
    done: set[str] = set()
    remaining = list(objs)
    while remaining:
        progressed = False
        for o in list(remaining):
            sup = o.get("support")
            if sup is None or sup not in names or sup in done:
                ordered.append(o)
                done.add(o["name"])
                remaining.remove(o)
                progressed = True
        if not progressed:  # cycle / dangling support
            ordered.extend(remaining)
            break
    return ordered


DEFAULT_ISAAC_PYTHON = os.environ.get(
    "SCENERIG_ISAAC_PYTHON",
    os.environ.get("GRASE_ISAAC_PYTHON", "lib/utils/third_party/isaac/venv/bin/python"),
)
# Module attr (not inlined) so settle_client_test.py can swap in a fake server.
SETTLE_SERVER_SCRIPT = REPO_ROOT / "isaac/isaac_settle_server.py"
# Kit boot deadline. Measured 2026-07-28: a COLD boot is ~225 s on a fully idle box and
# ~67 s once warm; 8 concurrent lanes roughly double it. The cost is Lustre per-file
# metadata latency over the 25 GB / 63k-file Isaac install (stat 1.76 ms cold vs 0.069 ms
# warm), NOT bandwidth (bulk reads run 320-450 MB/s), so the "warm" case is just the
# kernel page cache. The old 300 s left ~25 % margin over an idle cold boot and blew twice
# in 0728_idle_9096 — 300 s wasted each time, then a retry that only succeeded because
# attempt 1 had populated the page cache. This bounds the HANG path only: a boot that dies
# is still caught in seconds by the proc.poll() check below. scripts/prewarm_isaac.sh
# removes the underlying cold-read cost; this is the safety margin for when it hasn't run.
BOOT_DEADLINE_S = float(os.environ.get("GRASE_ISAAC_BOOT_TIMEOUT", 900.0))
# Whose death should reap a shared settle server. NOT the spawning process: under a
# standalone main.py the first Isaac user is composition, which runs inside an exec.py
# MCP child that is torn down at the END of its stage (root.py) while certify still
# needs the warm server — keying on the parent would cost a ~100 s re-boot every rerun.
# The run entry points (runners/static_scene.py, main.py) export GRASE_RUN_OWNER_PID;
# static_scene.py wins when both are in play because main.py only setdefault()s it.
RUN_OWNER_PID_ENV = "GRASE_RUN_OWNER_PID"


def _open_boot_log(work: Path):
    """APPEND to <work>/settle_server.log, with a timestamped banner per spawn.

    Truncating ("w") meant a boot RETRY erased the failed attempt it was retrying —
    and a Kit boot that dies at the 300 s deadline is exactly when you need that log.
    On 2026-07-27 both the composition and certify logs held only the last attempt,
    so the hang had to be reproduced from scratch to find it (Lustre RPC wait). The
    path stays `settle_server.log` because downstream tooling expects that name."""
    f = open(work / "settle_server.log", "a")
    f.write(f"\n===== settle server spawn {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
    f.flush()  # banner lands before the child starts writing to the same fd
    return f


STABILIZE_THETA_DEG = 12.0  # target tip threshold for the CoM correction ladder step


def _stabilize_params(npz_path: str) -> Optional[dict[str, Any]]:
    """Stage-A stabilization candidate: keep natural CoM xy when edge margin is at
    least 5 mm, otherwise move minimally toward the footprint's Chebyshev center to
    reach that margin; floor height at 0.3x natural, plus flatten/friction/damping.
    Single source of truth is isaac/isaac_auto_stabilize.py (same math the
    export-side loop uses); imported by path since isaac/ is a script dir, not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "isaac_auto_stabilize", REPO_ROOT / "isaac/isaac_auto_stabilize.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        com, r, h, h_uni, inertia = mod.solve_com(npz_path, STABILIZE_THETA_DEG)
    except Exception:  # noqa: BLE001 - degenerate footprint (needle-thin) -> no candidate
        return None
    # Contact-footprint guard: when the bottom-slice footprint is a SLIVER relative to
    # the object (0713_obbfix_gpt1 glasses: r_cheb 4.3mm under a 132mm frame), a CoM
    # over it doesn't restore a natural rest — it manufactures a corner-balancing act
    # (the visual mesh cantilevers 10-40mm in the air). Refuse the rung; the ladder
    # then accepts the FALLEN pose, which for this class is the physically and
    # visually correct outcome (glasses tip over and lie flat, like real ones).
    d = np.load(npz_path)
    allv = np.vstack([d[f"v{i}"] for i in range(int(d["n"]))])
    ext = float((allv[:, :2].max(0) - allv[:, :2].min(0)).max())
    if r < 0.05 * ext:
        print(f"[stabilize] refused for {Path(npz_path).stem}: contact r_cheb {r * 1000:.1f}mm is a "
              f"sliver of the {ext * 1000:.0f}mm footprint — a fallen rest beats a corner balance")  # fmt: skip
        return None
    return {
        "com_world": com,
        # natural (uniform-density) tip margin of the UNmodified body: the gate for
        # arming this bundle on a standing object — atan(footprint radius / CoM
        # height). Below STABILIZE_THETA_DEG the body is metastable under its own
        # mass model (stands a short isolated drop, topples in a long joint sim).
        "tip_theta_uniform_deg": float(np.degrees(np.arctan2(r, max(h_uni, 1e-6)))),
        "flatten_base_mm": mod.SLICE_MM,
        "friction": 0.9,
        "angular_damping": 1.5,
        # MUST accompany com_world: a centerOfMass override with no matching
        # inertia tensor is a self-inconsistent rigid body — under a plain drop
        # (this ladder's own stability check) the inconsistency is invisible, but
        # real contact torque near a support edge can turn it into a launch
        # (0720_compfix_real8219, root-caused via ablation to exactly this gap).
        "diagonal_inertia": inertia["diagonal_inertia"],
        "principal_axes": inertia["principal_axes"],
        # solve_com's own (density-250) mass — the reference consumers need to
        # rescale diagonal_inertia (linear in mass) when they author a VLM mass
        # instead, keeping the (mass, CoM, inertia) triple self-consistent.
        "solver_mass_kg": float(inertia["mass"]),
        "_note": f"theta={STABILIZE_THETA_DEG}deg r_cheb={r * 1000:.1f}mm "
                 f"h={h * 1000:.1f}mm (uniform {h_uni * 1000:.1f}mm)",
    }  # fmt: skip


def _decompose_all(ordered: list[dict[str, Any]], work: Path) -> None:
    """CoACD-decompose each object's current GLB (+ pristine candidate) into
    ``collision_<name>.npz``. Fresh every time — the GLBs mutate in place across
    the pipeline, so caching would go stale. Parallel: CoACD is the slow half.

    SUPPORT objects with a pristine sibling are PRISTINE-FIRST (2026-08-22):
    CoACD only ever sees the axis-aligned pristine — whose voxel remesh is
    grid-aligned, so the cavity floor stays flat — and the placed collider is the
    exact pristine->placed similarity transform of that decomposition
    (``collision.placed_collider_from_pristine``; abc_5's ~4 deg tray cooked
    placed carried a +3.1 mm median floor margin with 5.9% phantom-wall cells vs
    a 2-6x flatter pristine cook). One decomposition instead of two per support;
    a broken correspondence falls back to today's placed cook."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from lib.tools.geometry.collision import (
        decompose_glb,
        placed_collider_from_pristine,
    )

    # An object is a support iff it appears in someone's ancestor chain (equal to
    # the set of direct scene-graph supports, since every chain entry is the direct
    # support of the object below it).
    supports = {a for o in ordered for a in (o.get("ancestors") or [])}
    supports.update(
        o["name"]
        for o in ordered
        if _semantic_container((o.get("placement") or {}).get("category"))
    )
    tasks = []
    derived = []  # supports whose placed npz is transformed from the pristine cook
    for o in ordered:
        o["parts_npz"] = str(work / f"collision_{o['name']}.npz")
        is_support = o["name"] in supports
        if o.get("pristine_glb"):
            o["pristine_parts_npz"] = str(work / f"collision_{o['name']}_pristine.npz")
            # The pristine task needs no clamp frame (it IS axis-aligned in world,
            # so its world z-clamp already trims uniformly across the flat faces).
            tasks.append(
                (o["pristine_glb"], o["pristine_parts_npz"], is_support, None)
            )
        if is_support and o.get("pristine_glb"):
            derived.append(o)
        else:
            # frame_glb: the pristine candidate is SAM3D's axis-aligned canonical
            # save placed UPRIGHT (owner insight 2026-08-05) — same topology as the
            # raw, so corner correspondence recovers the raw's tilt exactly and the
            # overshoot clamp trims uniformly (see collision._pristine_frame).
            tasks.append(
                (o["glb"], o["parts_npz"], is_support, o.get("pristine_glb"))
            )
    # spawn, not fork: CoACD's internal threads make forked workers deadlock if the
    # parent process already ran a decomposition (e.g. tests, repeated settles).
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(4, len(tasks)), mp_context=ctx) as ex:
        list(ex.map(decompose_glb, *zip(*tasks)))
    # Derive placed support colliders in the parent (a cheap exact transform — no
    # CoACD runs in this process, so no torch-after-CoACD hazard).
    fallback = []
    for o in derived:
        if not placed_collider_from_pristine(
            o["pristine_parts_npz"], o["pristine_glb"], o["glb"], o["parts_npz"]
        ):
            print(
                f"[collision] {o['name']}: pristine->placed similarity recovery "
                "failed; falling back to a placed CoACD decomposition"
            )
            fallback.append((o["glb"], o["parts_npz"], True, o["pristine_glb"]))
    if fallback:
        with ProcessPoolExecutor(
            max_workers=min(4, len(fallback)), mp_context=ctx
        ) as ex:
            list(ex.map(decompose_glb, *zip(*fallback)))


# --------------------------------------------------------------------------- #
# Incremental settle: persistent Isaac server, DFS build-up, interleaved ICP   #
# --------------------------------------------------------------------------- #
class SettleClient:
    """JSON-lines client for isaac_settle_server.py (skips Isaac log noise).

    With ``shared_dir`` (and GRASE_ISAAC_SHARED != "0"): connect to a warm server
    advertised by ``<shared_dir>/isaac.port``, else spawn one with ``--port-file``.
    The server survives ``disconnect()`` so later stages — which run in DIFFERENT
    OS processes (preprocess in static_scene.py, composition in the exec.py MCP
    server, certify in main.py) — reuse ONE SimulationApp boot per run instead of
    three. Reconnecting auto-``reset``s the registry, so every stage still sees a
    fresh empty server. ``close()`` (or shutdown_shared_settle_server at run end)
    shuts the server down. Without ``shared_dir``: the legacy private pipe-owned
    server, shut down by ``close()``/``disconnect()``.

    Those shutdowns are ``finally`` blocks, so they do NOT cover SIGKILL / OOM-kill /
    pod eviction. A spawned server therefore also self-reaps: it gets ``--owner-pid``
    (see RUN_OWNER_PID_ENV) and exits when that run dies, with an idle timeout as the
    catch-all backstop. See isaac_settle_server.main_tcp.
    """

    def __init__(
        self,
        work: Path,
        isaac_python: str = DEFAULT_ISAAC_PYTHON,
        shared_dir: Optional[Path] = None,
    ):
        self._log = None
        self.proc = None
        self.sock = None
        self._fin = None
        self.shared = (
            shared_dir is not None and os.environ.get("GRASE_ISAAC_SHARED", "1") != "0"
        )
        if not self.shared:
            self._spawn_pipe(work, isaac_python)
            return
        self._port_file = Path(shared_dir) / "isaac.port"
        if self._connect_existing():
            return
        for attempt in (1, 2):  # one respawn before giving up
            try:
                self._spawn_tcp(work, isaac_python)
                return
            except RuntimeError as exc:
                self._reap()
                if attempt == 2:
                    raise
                # stderr, NEVER stdout: inside the exec.py MCP stdio server,
                # stdout IS the JSON-RPC channel — a bare print corrupts it.
                print(f"[isaac] settle server boot failed ({exc}); retrying once",
                      file=sys.stderr, flush=True)  # fmt: skip

    # -- connection setup --------------------------------------------------- #

    def _connect_existing(self) -> bool:
        """Adopt the warm server behind the port file; False if absent/dead."""
        import socket

        try:
            info = json.loads(self._port_file.read_text())
            sock = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=2)
        except (OSError, ValueError, KeyError):
            return False
        self.sock, self._fin = sock, sock.makefile("r")
        try:
            sock.settimeout(30.0)  # a healthy idle server answers instantly
            pid = self.rpc({"cmd": "ping"}).get("pid")
            self.rpc({"cmd": "reset"})  # fresh-server invariant for this stage
        except (RuntimeError, OSError):
            self._close_sock()
            return False
        sock.settimeout(None)  # settles legitimately block for minutes
        # stderr, NEVER stdout (see boot-retry note): composition's client runs
        # inside the MCP stdio server where stdout is the protocol stream.
        print(f"[isaac] reused warm settle server (pid {pid})",
              file=sys.stderr, flush=True)  # fmt: skip
        return True

    def _spawn_tcp(self, work: Path, isaac_python: str):
        """Boot a shared server. Its stdout+stderr go to the log (an undrained
        stdout pipe would eventually block Kit); readiness = the port file
        appearing, with proc.poll() each tick so a crashed boot fails in
        seconds rather than sitting out the full BOOT_DEADLINE_S readline
        deadline (that deadline only bounds a genuine HANG)."""
        import socket
        import time

        self._port_file.unlink(missing_ok=True)
        self._log = _open_boot_log(work)
        # Self-reaping owner for the daemon: the run entry point if it exported one,
        # else us (direct callers — tests, one-off scripts — ARE the whole run).
        owner = os.environ.get(RUN_OWNER_PID_ENV) or str(os.getpid())
        self.proc = subprocess.Popen(
            [isaac_python, str(SETTLE_SERVER_SCRIPT),
             "--port-file", str(self._port_file), "--owner-pid", owner],
            stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "OMNI_KIT_ALLOW_ROOT": "1",
             # numpy threaded OpenBLAS livelocks (exec_blas_async_wait ->
             # sched_yield spin; 0724_roomfix_room1 wedged 20min on a tiny
             # 'add' matmul) with its default nproc-sized pool; the settle
             # path only does 3x3 / per-part-vertex products: single-thread it.
             "OPENBLAS_NUM_THREADS": "1"},
        )  # fmt: skip
        deadline = time.time() + BOOT_DEADLINE_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("settle server exited before ready")
            if self._port_file.exists():
                try:
                    info = json.loads(self._port_file.read_text())
                    self.sock = socket.create_connection(
                        ("127.0.0.1", int(info["port"])), timeout=5
                    )
                except (OSError, ValueError, KeyError) as exc:
                    raise RuntimeError(f"port file up but connect failed: {exc}")
                # create_connection's timeout PERSISTS on the socket; clear it
                # or every rpc dies at 5 s (killed the 0721_perffix settles —
                # any drop longer than 5 s aborted the whole incremental stage).
                self.sock.settimeout(None)
                self._fin = self.sock.makefile("r")
                return
            time.sleep(0.5)
        raise RuntimeError("settle server not ready in time")

    def _spawn_pipe(self, work: Path, isaac_python: str):
        """Legacy private server over stdin/stdout (GRASE_ISAAC_SHARED=0)."""
        import time

        self._log = _open_boot_log(work)
        self.proc = subprocess.Popen(
            [isaac_python, str(SETTLE_SERVER_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1, cwd=str(REPO_ROOT),
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "OMNI_KIT_ALLOW_ROOT": "1",
             # numpy threaded OpenBLAS livelocks (exec_blas_async_wait ->
             # sched_yield spin; 0724_roomfix_room1 wedged 20min on a tiny
             # 'add' matmul) with its default nproc-sized pool; the settle
             # path only does 3x3 / per-part-vertex products: single-thread it.
             "OPENBLAS_NUM_THREADS": "1"},
        )  # fmt: skip
        deadline = time.time() + BOOT_DEADLINE_S
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("settle server exited before ready")
            try:
                if json.loads(line).get("ready"):
                    return
            except json.JSONDecodeError:
                continue
        raise RuntimeError("settle server not ready in time")

    # -- protocol ------------------------------------------------------------ #

    def rpc(self, req: dict) -> dict:
        payload = json.dumps(req) + "\n"
        if self.sock is not None:
            self.sock.sendall(payload.encode())
            reader = self._fin
        else:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
            reader = self.proc.stdout
        while True:
            line = reader.readline()
            if not line:
                raise RuntimeError(f"settle server died during {req.get('cmd')}")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue  # Isaac log noise on stdout (pipe mode)
            if not resp.get("ok"):
                raise RuntimeError(
                    f"settle server {req.get('cmd')}: {resp.get('error')}"
                )
            return resp

    # -- lifetime ------------------------------------------------------------ #

    def disconnect(self):
        """Release this stage's connection; a SHARED server stays warm for the
        next stage. A private (pipe) server has no other clients — shut it down."""
        if self.sock is None:
            self.close()
            return
        self._close_sock()
        if self._log:
            self._log.close()  # our copy of the fd; the daemon keeps its dup

    def close(self):
        """Shut the server down (run-owner / legacy semantics)."""
        try:
            self.rpc({"cmd": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        self._close_sock()
        self._reap()

    def _close_sock(self):
        for s in (self._fin, self.sock):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        self.sock = self._fin = None

    def _reap(self):
        if self.proc is not None:
            try:
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            self.proc = None
        if self._log:
            self._log.close()
            self._log = None


def shutdown_shared_settle_server(shared_dir: Path) -> bool:
    """Best-effort run-end shutdown of a warm shared settle server. True if a
    server was told (or forced) to exit."""
    import signal
    import socket

    pf = Path(shared_dir) / "isaac.port"
    if not pf.exists():
        return False
    try:
        info = json.loads(pf.read_text())
    except (OSError, json.JSONDecodeError):
        info = {}
    ok = False
    try:
        with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=2) as s:
            s.settimeout(15.0)
            s.sendall(b'{"cmd": "shutdown"}\n')
            s.makefile("r").readline()
        ok = True
    except (OSError, ValueError, KeyError):
        try:
            os.kill(int(info["pid"]), signal.SIGTERM)
            ok = True
        except (OSError, ValueError, KeyError, TypeError):
            pass
    pf.unlink(missing_ok=True)  # the server removes it too; cover crash paths
    return ok


# CAPSIZE: the ONE definition of "this body did not stay the way it was put down",
# shared by the settle ladder, both certify passes and the demo (2026-07-31). It was
# three constants in two modules (physics.TILT_CAP, composition_physics.TILT_CAP_DEG /
# TOPPLE_DEG) plus two hand-rolled rollable branches, and they had already drifted:
# certify_composed_scene flagged on a bare tilt with NO rollable branch, so an in-place
# roll (0731_rls_workdesk marker: 165 deg at 43 mm) read as a capsize while the server's
# own pin path — same event, same numbers — did not.
CAPSIZE_DEG = 45.0
# A ROLLABLE (a lying marker/bottle, a round fruit) has no meaningful tilt: rolling about
# its own axis is a 90-180 deg "capsize" that is physically nothing. It is judged on how
# far it TRAVELLED instead — rolling in place passes, rolling away does not.
CAPSIZE_DISP_MM = 50.0
TILT_CAP = CAPSIZE_DEG  # legacy alias: the ladder's rung gate reads better as a "cap"


# STANDING VETO (2026-08-27). ``rollable`` is a VLM judgement, and it is wrong in one
# specific, expensive direction: an UPRIGHT slender object (marker, pen, bottle, can,
# yogurt drink) read as "lying". That flag then disables the ONLY guard against it
# toppling, because capsized() below does not evaluate tilt at all for a rollable — so
# settle may lay a correctly-standing object flat and nothing can reject the result.
# Measured on 0825/0826 eibin: 9 objects placed at 11-22 deg from vertical shipped at
# 87-90 deg, every one of them flagged rollable=True.
#
# The veto is one-directional (True -> False only): it can never GRANT rollable, so it
# cannot strip the tilt guard from anything. It is also NOT a pin — clearing the flag
# routes the body through the non-rollable ladder, which tries the pristine and
# CoM/flatten rungs and still accepts the fallen pose when nothing stands, so a wrong
# upright placement degrades instead of freezing.
#
# Judged on the PLACED collision geometry (the settle INPUT, decomposed by
# _decompose_all before the add), which is the one moment the object is still standing.
UPRIGHT_VETO_DEG = 25.0
# Elongation (sqrt of the ratio of the two largest PCA eigenvalues) below this means the
# long axis is numerical noise — a ball or a lemon. Those must keep their exemption:
# rolling is all they do. Two toy balls in benchmark_final sit at 1.00 and 1.21.
UPRIGHT_VETO_MIN_ELONGATION = 1.5


def placed_long_axis(vertices) -> tuple[float, float]:
    """(tilt of the long axis from world +Z in degrees, elongation) of a placed body.

    Elongation is sqrt(lambda_max / lambda_mid) of the vertex covariance: 1.0 for a
    sphere, large for a pen. The axis sign is arbitrary, hence ``abs`` on the Z term.
    """
    v = np.asarray(vertices, dtype=float)
    v = v - v.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(v.T))
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    ordered = sorted(eigenvalues)
    elongation = math.sqrt(ordered[-1] / max(ordered[-2], 1e-12))
    tilt_deg = math.degrees(math.acos(min(1.0, abs(float(axis[2])))))
    return tilt_deg, elongation


def placed_upright(
    vertices,
    max_tilt_deg: float = UPRIGHT_VETO_DEG,
    min_elongation: float = UPRIGHT_VETO_MIN_ELONGATION,
) -> bool:
    """Is this placed body a slender object standing on end? See UPRIGHT_VETO_DEG."""
    tilt_deg, elongation = placed_long_axis(vertices)
    return tilt_deg <= max_tilt_deg and elongation >= min_elongation


def load_parts_vertices(npz_path: str):
    """Stack the CoACD part vertices of a ``collision_<name>.npz`` (``v*`` keys only;
    the ``f*`` keys are face indices and stacking them silently corrupts the extents)."""
    with np.load(npz_path) as data:
        parts = [data[k] for k in data.files if k.startswith("v")]
    return np.vstack(parts) if parts else np.zeros((0, 3))


def capsized(tilt_deg: float, disp_mm: float, rollable: bool) -> bool:
    """Did this body fail to hold the pose it was placed/settled in?

    Mirrors isaac_settle_server.cmd_certify's own pin test (``tilt_bad = ... and not
    rollable``) so the flag a run REPORTS and the pose the server would have PINNED can
    never disagree."""
    return (
        disp_mm > CAPSIZE_DISP_MM if rollable else tilt_deg > CAPSIZE_DEG
    )  # fmt: skip


def drop_unstable(result: dict, rollable: bool) -> bool:
    """Can this drop be selected as a demonstrated rest?

    Older servers did not return ``converged``; preserve their capsize-only behavior.
    An explicit False means the pose is a mid-motion snapshot even when its current
    tilt/displacement is still below the class-aware cap.
    """
    return result.get("converged") is False or capsized(
        result["tilt_deg"], result.get("disp_xy_mm", 0.0), rollable
    )


def icp_redrop_rejection_reason(
    result: dict, rollable: bool, already_fell: bool
) -> Optional[str]:
    """Why an ICP-aligned re-drop must be rolled back, or ``None`` to retain it.

    An already-fallen object's cumulative tilt permanently includes that accepted fall,
    so applying the ordinary capsize predicate would reject every later ICP re-drop even
    when it only makes a small, stable correction.  Such an object is instead judged on
    this re-drop's convergence and centroid travel.  This preserves the accepted fallen
    pose while preventing ICP from turning it into a large scene-layout escape.
    """
    if result.get("converged") is False:
        return "non_converged"
    if already_fell:
        if float(result.get("disp_xy_mm", 0.0)) > CAPSIZE_DISP_MM:
            return "escaped_xy"
        return None
    if capsized(result["cum_tilt_deg"], result.get("disp_xy_mm", 0.0), rollable):
        return "capsized"
    return None


def _drop_attempt(result: dict, stage: str, role: str, attempt_id: int) -> dict:
    """Compact, JSON-safe evidence for one logical cmd_drop/clear_along call."""
    attempt = {
        "id": attempt_id,
        "stage": stage,
        "role": role,
        "continued": bool(result.get("drop_continued", False)),
        "converged": bool(result.get("converged", True)),
    }
    for key in ("steps_used", "tilt_deg", "cum_tilt_deg", "disp_xy_mm"):
        if result.get(key) is not None:
            attempt[key] = result[key]
    return attempt


def _finalize_drop_bookkeeping(
    rec: dict, attempts: list[dict], retained: Optional[int]
) -> None:
    """Project attempt history onto the final retained preprocessing pose.

    A failed trial *after* ``retained`` (notably a reverted ICP correction) must not
    pollute final convergence or recovery. Probe-only drops are evidence but are not
    candidates and therefore do not create ``settle_recovered``.
    """
    rec["drop_attempts"] = attempts
    rec["retained_drop_attempt"] = retained
    final = next((a for a in attempts if a["id"] == retained), None)
    if final is None:
        rec["converged"] = False
        rec["steps_used"] = None
    else:
        rec["converged"] = final["converged"]
        rec["steps_used"] = final.get("steps_used")
    rec["drop_continued"] = any(a["continued"] for a in attempts)
    rec["settle_failed"] = rec["converged"] is False
    rec["settle_recovered"] = bool(
        rec["converged"]
        and retained is not None
        and any(
            not a["converged"] and a["role"] == "candidate" and a["id"] < retained
            for a in attempts
        )
    )


# Owner rules 2026-07-31 (0730-batch replay in CHANGELOG):
# FLIP_ACCEPT_DEG — a non-rollable whose raw drop lands beyond this is FLIPPING to
# its stable face; every rescue rung lands ~the same flip (0730 moma tape 169.6 ->
# stab attempt 170.4; mugs4v2 clip 153.7 -> 153.6), so rescue burns two drops to
# change nothing — accept the flip and skip the rungs.
FLIP_ACCEPT_DEG = 145.0
# PRISTINE_CONFIRM_DEG — the pristine (complete) mesh landing within this of the raw
# tilt means the rotation is NOT a raw-mesh artifact: the raw rest is legitimate
# (0730_fix_breakfast spoon: raw 45.2 vs pristine 33.9 — a conform-in-place that the
# old adopt-if-under-cap rule turned into a mesh swap). Blocks PRISTINE adoption
# only; the stabilized rung still gets its shot (sunglasses/fan/plush class, where
# BOTH meshes are unstable at the authored pose and the CoM hold is correct).
PRISTINE_CONFIRM_DEG = 30.0


def flip_accepted(tilt_raw: float, rollable: bool) -> bool:
    """Rule 1: a NON-rollable body past ``FLIP_ACCEPT_DEG`` is flipping to its stable
    face — every rescue rung lands the same flip, so accept the raw rest and skip
    them. Rollables are exempt (their gate is xy drift; rolling in place reads as a
    huge tilt)."""
    return (not rollable) and tilt_raw > FLIP_ACCEPT_DEG


def pristine_confirms_raw(tilt_raw: float, tilt_pristine: float) -> bool:
    """Rule 2: the pristine (complete) mesh landing within ``PRISTINE_CONFIRM_DEG``
    of the raw tilt CONFIRMS the raw rest — the rotation is real geometry, not a
    raw-mesh artifact, so pristine must not be adopted (no mesh swap). The
    stabilization rung still runs: when BOTH meshes are unstable at the authored
    pose, a CoM hold is the correct answer (sunglasses/fan/plush)."""
    return abs(tilt_pristine - tilt_raw) < PRISTINE_CONFIRM_DEG


# Scene-entry topple retry: an object that STOOD in the isolation ladder
# (ladder_tilt <= HCLEAR_LADDER_OK) but FELL on the full-scene drop because a
# neighbor forced a big vertical lift (lift > HCLEAR_LIFT_MM) is retried by
# SLIDING it to the nearest clear xy within a HCLEAR_MAX-radius DISK (server
# cmd_clear_along), then settling from contact — no free-fall, no topple. A disk
# rather than the camera ray: the clear opening is usually off any single ray
# (0715 wendy1 marker cleared 25mm away in -x while its ray pointed -y).
# NOTE: the server's down-preference (DOWN_PREF_MIN_LIFT_MM, armed by the
# ``ancestors`` field on scene-entry drops) now resolves big non-ancestor lifts
# DOWNWARD first; these gates remain the reactive backstop for overlaps with no
# clear pose below.
HCLEAR_MAX = 0.10  # max horizontal slide radius (m)
HCLEAR_LADDER_OK = 15.0  # "stood in isolation" ceiling (deg)
HCLEAR_LIFT_MM = 30.0  # "big neighbor-forced lift" floor (mm)
# A dropped-from-atop object can land nearly UPRIGHT (low cum_tilt) yet far from
# where it was placed — it tumbled/slid off the thing it overlapped. A big
# horizontal drift of the settled origin vs the placed pose is that signal, and
# it fires the same clear-then-settle retry even when the tilt gate misses
# (abc3 microphone_1: 417mm lift-to-clear, landed at 11deg -> no topple, but
# drifted off its stand).
HCLEAR_XY_MM = 50.0  # "slid/tumbled far from placed xy" floor (mm)
# ATOP-A-NON-PARENT retry: scene-entry lift-to-clear resolves an XY overlap with a
# same-level neighbor VERTICALLY (lifts straight up), so an object can settle nearly
# upright and un-drifted yet STACKED on a body that is not its support (wendy1 pliers
# on the box: lift 74mm, cum_tilt 0deg -> the tilt/drift gates both miss it). When the
# body directly beneath is a non-ancestor, non-static object supporting most of the
# footprint, slide it off to a clear cell on its real support instead.
ATOP_FRAC = 0.5  # min fraction of the object's base a non-parent must support to fire
HCLEAR_MAX_ATOP = 0.15  # larger slide radius: must clear a (possibly wide) neighbor (m)
# RIM-TOPPLE NUDGE: an object that FELL only because its placed xy sits on the parent's
# upturned rim (topples on the parent/scene but rests flat on a bare slab) is slid toward
# the parent centroid in RIM_NUDGE_STEP_MM steps up to RIM_NUDGE_CAP_MM; the FIRST offset
# whose full-scene re-drop tilt <= TILT_CAP is accepted (0722 abc2 knife on the tray rim,
# 5.7deg on a slab -> 173deg on the rim -> ~9deg once nudged 20mm onto the floor).
RIM_NUDGE_STEP_MM = 5
RIM_NUDGE_CAP_MM = 25


def _first_stable_shift(
    drop_tilt, step_mm=RIM_NUDGE_STEP_MM, cap_mm=RIM_NUDGE_CAP_MM, tilt_cap=TILT_CAP
):
    """Smallest inward shift (mm, a multiple of ``step_mm``) whose ``drop_tilt(mm)``
    settle tilt is <= ``tilt_cap``, or None if the cap is reached without one.
    ``drop_tilt`` re-drops the object at that inward offset and returns the tilt (deg).
    Tilt is NON-monotonic in the shift (the long object slides across the rim slope),
    so this returns the FIRST acceptable step, not an interpolation or a minimum."""
    for mm in range(step_mm, cap_mm + 1, step_mm):
        if drop_tilt(mm) <= tilt_cap:
            return mm
    return None


def _bake_baseline_glb(o: dict, rec: dict) -> str:
    """The GLB the settle bake must compose the server's cumulative matrix onto.

    The server RESETS its cumulative matrix at every geometry swap ("the new npz IS
    the new baseline the client bakes onto" — isaac_settle_server 'swap' handler), so
    a PRISTINE-rung object's total is relative to the canonical-UPRIGHT mesh. Baking
    it onto the RAW glb leaves the raw-vs-canonical rotation uncorrected: in
    0725_arr2_room1 the merged vase stood at 2.2° in physics (and through both
    certifies) yet RENDERED lying at ~75° — exactly its raw SAM3D pose — because the
    bake ignored the swap. Hidden until then: pristine-rung objects historically had
    near-canonical raw poses (chair_1 same run: cum 0.0°). The STABILIZED rung swaps
    the RAW parts npz (flatten/CoM bundle only), so the raw glb stays its baseline;
    raw / rolled_back never swap."""
    if rec.get("chosen") == "pristine" and o.get("pristine_glb"):
        return o["pristine_glb"]
    return o["glb"]


def certify_rescue_candidates(
    drift: dict, ordered: list, records: dict, theta_deg: float = STABILIZE_THETA_DEG
) -> list[str]:
    """Objects to re-certify WITH their CoM bundle after a joint certify (2026-09-16).

    The CoM rung is decided on the isolated ladder drop, so a body that stands that short
    drop unaided but topples in the long joint certify shipped fallen with ``fell=False``
    (IMG_8219 plush: stood 2.5 deg in isolation for the first time in 14 runs, then 68 deg
    at certify and baked). Candidates are non-rollable RAW standers with no bundle yet,
    no accepted flip, whose certify tilt capsized and whose intrinsic tip margin is below
    ``theta_deg`` — the same metastability test the pristine branch already applies. A
    stable-once-upright object (margin >= theta) is never touched; nor is one that simply
    held certify, so the 57 marginal-but-standing objects of the 09-14/09-15/09-16 runs
    are unaffected (1 of 1010 raw standers qualified).
    """
    out = []
    for o in ordered:
        name = o["name"]
        rec = records.get(name) or {}
        stab = o.get("stabilize")
        if (
            not stab
            or rec.get("chosen") != "raw"
            or rec.get("fell")
            or rec.get("rollable")
            or rec.get("flip_accepted")
            or rec.get("physics_overrides")
        ):
            continue
        d = drift.get(name) or {}
        if float(d.get("tilt_deg", 0.0)) <= CAPSIZE_DEG:
            continue
        if float(stab.get("tip_theta_uniform_deg", float("inf"))) >= theta_deg:
            continue
        out.append(name)
    return out


def _chosen_swap(o: dict, rec: dict) -> dict:
    """The swap args that reproduce the collider RUNG the ladder chose (so a rim-nudge
    re-drop resets to the SAME baseline the object was settled with)."""
    if rec.get("chosen") == "pristine" and o.get("pristine_parts_npz"):
        args = {"npz": o["pristine_parts_npz"]}
        po = rec.get("physics_overrides")
        if po:  # metastable pristine carries the stabilization bundle (see ladder)
            args.update(
                {
                    "com": po["com_world"],
                    "friction": po["friction"],
                    "damping": po["angular_damping"],
                    "flatten_mm": po["flatten_base_mm"],
                }
            )
        return args
    if rec.get("chosen") == "stabilized" and o.get("stabilize"):
        s = o["stabilize"]
        return {"npz": o["parts_npz"], "com": s["com_world"],
                "friction": s["friction"], "damping": s["angular_damping"],
                "flatten_mm": s["flatten_base_mm"]}  # fmt: skip
    return {"npz": o["parts_npz"]}


# --------------------------------------------------------------------------- #
# physics/pose_changes.json — the run's settle record                          #
# --------------------------------------------------------------------------- #
# One file per scene, written by TWO stages in different processes: preprocess owns
# `objects` (whole-file write), composition certify merges its own block in at the end
# of the run. Both go through the helpers below so the schema and the run stamping stay
# in one place.
#
# The record keys are an explicit whitelist — a rec key absent here is silently dropped,
# which is how the first 0731_lad batch shipped records with the ladder rungs skipped
# but no reason recorded. pose_changes_test.py pins the set.
_POSE_CHANGE_KEYS = (
    "chosen", "tilt_raw", "tilt_pristine", "tilt_stabilized", "ladder_tilt_deg",
    "rollable", "disp_raw_mm", "ladder_disp_mm", "rollback_xy_mm", "tilt_rollback",
    "disp_rollback_mm",
    # owner ladder rules (2026-07-31): WHY the rescue rungs were skipped
    "flip_accepted", "raw_confirmed_by_pristine", "physics_overrides", "lift_mm",
    # signed releases (negative = snap-down of a floating pose) + the non-ancestor
    # down-preference flag
    "release_dz_mm", "ladder_release_dz_mm", "down_cleared",
    # F0b: convergence of the RETAINED drop, complete drop history, and whether the
    # ladder recovered from an earlier candidate that stayed in motion. A still-false
    # terminal pose is report-only: production continues with settle_failed=True.
    "converged", "drop_continued", "steps_used", "drop_attempts",
    "retained_drop_attempt", "settle_recovered", "settle_failed",
    "icp", "icp_skipped", "icp_skipped_overlap_mm", "icp_reverted_tilt_deg",
    "icp_reverted_disp_mm", "icp_reverted_reason",
    "ray_cleared", "ray_push_mm",
    # per-object drift of the PREPROCESS certify pass ({dxy, dz, tilt_deg}), and the
    # cumulative placed -> post-preprocess world matrix that the delivered-vs-placed
    # measure composes the composition delta onto (see certify_composed_scene)
    "certify", "settle_total",
    # certify-topple rescue (2026-09-16): an isolation-stander that toppled in the joint
    # certify was re-certified with its CoM bundle ({tilt_before_deg, tilt_after_deg, held})
    "certify_rescue",
    # standing veto: present only when a VLM rollable=True was cleared because the body
    # was PLACED standing on end ({placed_tilt_deg, elongation}); absent means no veto
    "rollable_veto",
)  # fmt: skip


def run_id_for(out_dir) -> str:
    """Identity of the run that owns a scene's artifacts: ``<run>/<task>``, from the
    output layout ``output/static_scene/<run>/<task>/``. Used to tell a run's OWN
    certify record apart from one that rode in on a --skip-preprocess staging copy."""
    p = Path(out_dir).resolve()
    return f"{p.parent.name}/{p.name}"


def write_pose_changes(work: Path, run_id: str, records: dict) -> dict:
    """Write the preprocess-owned half of pose_changes.json (whole-file). Returns the
    payload for tests/callers."""
    pc = {
        "run_id": run_id,
        "mode": "incremental",
        "objects": {
            n: {
                # `fell` == the ladder accepted a fallen/flipped rest. RENAMED from
                # `toppled` on 2026-07-31: it never meant "tilt > 45", and an accepted
                # flip-to-stable-face (flip_accepted) lands here too.
                "fell": r.get("fell"),
                "tilt_deg": r.get("cum_tilt_deg"),
                "rest_dz": r.get("redrop_dz_mm", 0.0) / 1000.0,
                **{k: r.get(k) for k in _POSE_CHANGE_KEYS},
            }
            for n, r in records.items()
        },
    }
    work.mkdir(parents=True, exist_ok=True)
    (work / "pose_changes.json").write_text(json.dumps(pc, indent=2))
    return pc


def merge_pose_changes(pc_path: Path, run_id: str, block: dict) -> dict:
    """Merge a late block (composition certify) into pose_changes.json, dropping any
    certify record that belongs to a DIFFERENT run.

    A ``--skip-preprocess`` staging copies the source run's whole file — ladder records
    (wanted: they carry the collider choice + stabilization bundles) and its certify
    blocks (never wanted: they describe another scene's delivered poses). Nothing in the
    file used to say which run wrote what, so stale flags survived silently."""
    pc = {}
    if pc_path.exists():
        try:
            pc = json.loads(pc_path.read_text())
        except Exception:  # noqa: BLE001 - a corrupt file must not lose this run's record
            pc = {}
    if not isinstance(pc, dict):
        pc = {}
    # certify stamp when present, else the objects stamp: a staged file predating the
    # stamping still carries the SOURCE run's id there.
    owner = pc.get("certify_run_id") or pc.get("run_id")
    if owner is not None and owner != run_id:
        # preprocess owns exactly these three; EVERY other block was computed by the
        # certify of whatever run wrote the file (composition_certify, delivered, …)
        stale = [k for k in pc if k not in ("objects", "mode", "run_id")]
        for k in stale:
            pc.pop(k)
        if stale:
            print(f"[pose-changes] dropped {len(stale)} certify block(s) from run "
                  f"{owner} (this run is {run_id}) — staged copy, not ours",
                  flush=True)  # fmt: skip
    pc.update(block)
    pc["certify_run_id"] = run_id
    pc_path.parent.mkdir(parents=True, exist_ok=True)
    pc_path.write_text(json.dumps(pc, indent=2))
    return pc


def incremental_settle(
    placement_objs: list[dict[str, Any]],
    graph: dict,
    blender_cmd: str,
    out_dir: str,
    R=None,
    T=None,
    isaac_python: str = DEFAULT_ISAAC_PYTHON,
    disable_icp: bool = False,
    icp_freeze: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Incremental physics build-up on the persistent Isaac server (DFS in support
    order). Per object: release at the depth-guided pose from ~FIRST CONTACT (+2 mm;
    lifted out of penetration, or snapped DOWN when the MoGE pose floats — a float
    free-falls and topples what a contact release keeps upright, 0725_arr_room1
    chair_1; big lifts forced by a NON-ancestor prefer a clear pose below,
    server down-preference) -> ladder on instability (tilt>45, or for
    a ROLLABLE object xy-displacement>50mm — rolling in place is benign). Ladder
    rungs: non-rollable raw -> pristine (canonical upright) -> CoM/flatten
    stabilize -> accept the FALLEN pose; rollable raw -> ROLL-BACK (keep the
    settled rotation, translate the drift back, re-seat) -> accept the fallen
    re-seat (rollables never take pristine or the CoM/flatten freeze). Two owner
    rules (2026-07-31) short-circuit the non-rollable rungs: tilt_raw >
    FLIP_ACCEPT_DEG accepts the flip outright (rescues land the same flip), and a
    pristine drop within PRISTINE_CONFIRM_DEG of raw blocks the PRISTINE adoption
    (the rotation is real, not a mesh artifact — no mesh swap; the stabilize rung
    still runs, and its failure restores the raw rest) ->
    every drop that hits its cap continues the same live rigid body for one full
    extra budget; a still-nonconverged candidate cannot win a rung, but if every
    fallback fails the terminal pose is retained and reported instead of aborting.
    ``drop_attempts`` preserves every candidate/probe while ``converged`` describes
    only the pose actually retained ->
    per-object ICP (yaw/x/y/scale, bottom-
    center pivot) -> re-drop so z is re-derived after photo alignment -> commit
    (frozen static collider for all later releases). Ends with a certify pass (all
    dynamic, short free sim — micro-drift baked + recorded) and ONE texture-preserving
    Blender bake of each object's cumulative matrix into ``<glb>_pm.glb``.

    Writes physics/pose_changes.json (ladder/ICP/re-drop/certify per object) and
    pose_match.json (ICP log, demo-compatible). Returns the per-object records.

    ``disable_icp`` skips the per-object ICP (and its re-drop): objects keep their
    MoGE-placed yaw/xy/scale and are only settled + certified, never photo-aligned.
    ``icp_freeze`` locks individual DOFs instead (any of ``xy``/``yaw``/``scale``):
    the ICP still runs but only fits the remaining DOFs."""
    from lib.tools.geometry.pose_match import (
        bake_world_matrices,
        bottom_center_pivot,
        correction_matrix,
        fit_similarity,
        icp_freeze_kwargs,
        load_placed_mesh,
        object_cloud_for,
    )

    freeze_kw = icp_freeze_kwargs(icp_freeze)

    # A retained object that failed placement or mesh reconstruction must stop at
    # the artifact boundary.  The old filter below silently skipped it, allowing
    # physics and every later agent stage to operate on a smaller scene inventory.
    from lib.tools.geometry.inventory_contract import validate_object_materialization

    validate_object_materialization(graph, placement_objs, artifact_root=out_dir)

    nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    objs = []
    for o in placement_objs:
        glb, name = o.get("mesh_glb"), o.get("mesh_name")
        # The boundary validator above has already proved these fields and the mesh
        # artifact. Do not retain a second best-effort filter that can shrink the
        # inventory (or reinterpret a repository-relative path from another cwd).
        if not (glb and name):
            raise RuntimeError(
                "placement materialization changed after inventory validation: "
                f"{o.get('category')}#{o.get('instance')}"
            )
        node = nodes_by_id.get(f"{o.get('category')}#{o.get('instance')}")
        ancestors = ancestor_chain(nodes_by_id, node)
        pcand = o.get("pristine_glb")
        objs.append(
            {
                "glb": glb,
                "name": name,
                "center": o["center"],
                "placement": o,
                "support": ancestors[0] if ancestors else None,
                "ancestors": ancestors,
                "pristine_glb": pcand if (pcand and os.path.exists(pcand)) else None,
            }  # fmt: skip
        )
    if not objs:
        return {}
    # Ground-claim ordering: LARGEST footprint first within each support level (topo
    # preserves input order per round). Big objects take the desk; small clutter that
    # overlaps them lift-to-clears against a committed big body instead of the big
    # body perching on a small one (the wendy1 monitor-on-box defect).
    objs.sort(key=lambda o: -(float(o["placement"]["size"][0])
                              * float(o["placement"]["size"][1])))  # fmt: skip
    ordered = topo_by_support(objs)
    work = Path(out_dir) / "physics"
    work.mkdir(parents=True, exist_ok=True)
    _decompose_all(ordered, work)
    for o in ordered:
        o["stabilize"] = _stabilize_params(o["parts_npz"])
    # VLM physics (preprocess estimate, physics_estimate.py): per-object friction +
    # mass for every ladder rung. Estimation just ran on this same placed geometry,
    # so the mass needs no extents rescale here; ICP/scale changes later in this
    # loop ride the server's det-based s^3 mass rule.
    from lib.tools.geometry.physics_estimate import load_estimates

    vlm_est = load_estimates(work / "physics_vlm.json")

    def _vlm_fields(name: str, keep_friction: bool = True) -> dict:
        e = vlm_est.get(name) or {}
        out = {}
        if keep_friction and e.get("friction") is not None:
            out["friction"] = float(e["friction"])
        if e.get("mass_kg") is not None:
            out["mass"] = float(e["mass_kg"])
        return out

    # placed centres (mesh name -> world center) for the rim-topple nudge direction
    center_by_name = {oo["name"]: np.asarray(oo["center"], float) for oo in ordered}

    points = np.load(os.path.join(out_dir, "moge", "points.npy"))
    valid = np.load(os.path.join(out_dir, "moge", "mask.npy"))

    client = SettleClient(work, isaac_python, shared_dir=work)
    records: dict[str, dict[str, Any]] = {}
    icp_log = []
    try:
        for o in ordered:
            name = o["name"]
            anc = [a for a in o["ancestors"] if a in records]  # committed ancestors
            rec: dict[str, Any] = {"chosen": "raw", "fell": False}
            drop_state: dict[str, Any] = {"attempts": [], "retained": None}

            def _drop(req: dict, stage: str, role: str = "candidate"):
                """Run and record one drop-shaped RPC; clear_along returns the same fields."""
                result = client.rpc(req)
                attempt_id = len(drop_state["attempts"])
                drop_state["attempts"].append(
                    _drop_attempt(result, stage, role, attempt_id)
                )
                return result, attempt_id

            def _retain(attempt_id: int) -> None:
                drop_state["retained"] = attempt_id

            def _retry(swap_args: dict, stage: str):
                """Replace the ladder geometry/physics, then record its isolated drop."""
                client.rpc({"cmd": "swap", "name": name, **swap_args})
                return _drop({"cmd": "drop", "name": name, "scene": anc}, stage)

            # VLM state-aware rollability from the placement record (None on
            # pre-field runs -> omit; the server falls back to its orientation-
            # aware extents heuristic at the add pose).
            rv = o["placement"].get("rollable")
            # STANDING VETO: a rollable=True on a body PLACED standing on end is the
            # VLM misreading an upright slender object as lying, and it disables the
            # only guard against settle toppling it (see placed_upright). Clear it and
            # record why. One-directional: never grants rollable.
            if rv:
                try:
                    tilt_deg, elongation = placed_long_axis(
                        load_parts_vertices(o["parts_npz"])
                    )
                except (OSError, ValueError) as exc:  # unreadable npz: keep the VLM flag
                    print(f"[settle] {name}: upright check skipped ({exc})")
                    tilt_deg, elongation = 90.0, 1.0
                if tilt_deg <= UPRIGHT_VETO_DEG and elongation >= UPRIGHT_VETO_MIN_ELONGATION:
                    rv = False
                    rec["rollable_veto"] = {
                        "placed_tilt_deg": round(tilt_deg, 1),
                        "elongation": round(elongation, 2),
                    }
                    # The placement record is what composition_physics._rollable_flags
                    # reads, so write the RESOLVED flag back: one source of truth, per
                    # the CAPSIZE note above (the drifted-copies lesson).
                    o["placement"]["rollable"] = False
                    print(
                        f"[settle] {name}: rollable=True vetoed — placed standing "
                        f"({tilt_deg:.1f} deg from vertical, elongation {elongation:.2f})"
                    )
            roll_kv = {} if rv is None else {"rollable": bool(rv)}
            client.rpc({"cmd": "add", "name": name, "npz": o["parts_npz"],
                        **roll_kv, **_vlm_fields(name)})  # fmt: skip
            # LADDER in isolation (slab + ancestors only): a cluttered neighborhood
            # must not read as intrinsic instability (wendy1 monitor slid down a
            # mis-placed box's flank and wrongly took the pristine rung).
            r, r_attempt = _drop(
                {"cmd": "drop", "name": name, "scene": anc}, "ladder_raw"
            )
            _retain(r_attempt)
            # the RESOLVED flag (VLM override, else server extents heuristic)
            roll = bool(r.get("rollable"))
            rec["rollable"] = roll
            # signed ladder release (negative = snap-down of a floating MoGE pose)
            if "release_dz_mm" in r:
                rec["ladder_release_dz_mm"] = round(r["release_dz_mm"], 1)

            def _unstable(rr):
                # Class-aware rung gate (see capsized): a ROLLABLE is unstable only
                # when it rolled/slid AWAY — rolling in place reads as a huge tilt
                # but is benign (0724 bin_condiments: the lying ketchup bottle
                # rolled 90deg in place, tripped the tilt gate, took the pristine
                # UPRIGHT rung, and was baked standing). A cap-hit snapshot is not a
                # demonstrated rest even if it has not crossed either motion cap yet.
                return drop_unstable(rr, roll)

            rec["tilt_raw"] = r["tilt_deg"]
            rec["disp_raw_mm"] = r.get("disp_xy_mm")
            tilt, disp, glb = r["tilt_deg"], r.get("disp_xy_mm", 0.0), o["glb"]
            # Rule 1 (owner 2026-07-31): beyond FLIP_ACCEPT_DEG the body flips to its
            # stable face regardless of rung — accept the flip, skip the rescues.
            raw_converged = bool(r.get("converged", True))
            flip = raw_converged and flip_accepted(r["tilt_deg"], roll)
            if flip:
                rec["flip_accepted"] = True
            if _unstable(r) and roll:
                # ROLL-BACK rung (rollables only, terminal): the raw drop found a
                # genuine rest ORIENTATION but rolled/slid away past the disp cap.
                # Keep the rotation physics chose, cancel the travel: translate the
                # drift back and re-seat from contact. Rollables NEVER take the
                # CoM/flatten stabilize rung — freezing the PLACED pose bakes
                # cap-edge rests with the visual mesh hovering (0724 bin_condiments
                # ketchup), and the override bundle then rides into every
                # register/composition settle holding the unphysical pose up.
                dvec = r.get("disp_xy_m") or [0.0, 0.0]
                M = np.eye(4)
                M[0, 3], M[1, 3] = -float(dvec[0]), -float(dvec[1])
                client.rpc({"cmd": "transform", "name": name, "M": M.tolist()})
                r, r_attempt = _drop(
                    {"cmd": "drop", "name": name, "scene": anc, "budget": "micro"},
                    "ladder_rollback",
                )
                _retain(r_attempt)
                rec["rollback_xy_mm"] = round(float(np.hypot(*dvec)) * 1000.0, 1)
                rec["tilt_rollback"] = r["tilt_deg"]
                rec["disp_rollback_mm"] = r.get("disp_xy_mm")
                tilt, disp = r["tilt_deg"], r.get("disp_xy_mm", 0.0)
                rec["chosen"] = "rolled_back"
                if _unstable(r):
                    # the placed spot can't hold it (slope/rim: it rolled away
                    # AGAIN): accept the fallen re-seat pose — a natural rest
                    # beats a corner balance (same call as the sliver guard).
                    rec["fell"] = True
            # pristine = canonical UPRIGHT (SAM3D rotation ditched): never offered
            # to a rollable — it would stand a lying object up.
            if _unstable(r) and not roll and not flip and o.get("pristine_parts_npz"):
                rp, rp_attempt = _retry(
                    {"npz": o["pristine_parts_npz"], **_vlm_fields(name)},
                    "ladder_pristine",
                )
                rec["tilt_pristine"] = rp["tilt_deg"]
                if (
                    raw_converged
                    and bool(rp.get("converged", True))
                    and pristine_confirms_raw(rec["tilt_raw"], rp["tilt_deg"])
                ):
                    # Rule 2 (owner 2026-07-31): pristine CONFIRMS the raw rest — the
                    # rotation is not a raw-mesh artifact, so no mesh swap. Swap raw
                    # parts back in and re-drop so the accepted pose matches the raw
                    # glb the bake uses; the stabilized rung below still gets its shot.
                    rec["raw_confirmed_by_pristine"] = True
                    r, r_attempt = _retry(
                        {"npz": o["parts_npz"], **roll_kv, **_vlm_fields(name)},
                        "ladder_raw_restore",
                    )
                    _retain(r_attempt)
                    tilt, disp = r["tilt_deg"], r.get("disp_xy_mm", 0.0)
                else:
                    r = rp
                    if not _unstable(rp):
                        tilt, glb, rec["chosen"] = (
                            rp["tilt_deg"],
                            o["pristine_glb"],
                            "pristine",
                        )
                        disp = rp.get("disp_xy_mm", 0.0)
                        _retain(rp_attempt)
            stab = o.get("stabilize")
            if (
                _unstable(r)
                and not roll
                and not flip
                and rec["chosen"] == "raw"
                and stab
            ):
                r, r_attempt = _retry({
                    "npz": o["parts_npz"], "com": stab["com_world"],
                    "friction": stab["friction"], "damping": stab["angular_damping"],
                    "flatten_mm": stab["flatten_base_mm"], **roll_kv,
                    **_vlm_fields(name, keep_friction=False),  # stab friction wins
                }, "ladder_stabilized")  # fmt: skip
                _retain(r_attempt)
                rec["tilt_stabilized"] = r["tilt_deg"]
                if not _unstable(r):
                    tilt, disp = r["tilt_deg"], r.get("disp_xy_mm", 0.0)
                    rec["chosen"] = "stabilized"
                elif rec.get("raw_confirmed_by_pristine"):
                    # Rule-2 terminal: the pristine-confirmed RAW rest is the honest
                    # pose — restore it rather than accepting the stabilized attempt's
                    # fall (which can be WORSE than raw: 0730_full moma tape raw 69.1
                    # -> stabilized attempt fell 169.5).
                    r, r_attempt = _retry(
                        {"npz": o["parts_npz"], **roll_kv, **_vlm_fields(name)},
                        "ladder_raw_restore",
                    )
                    _retain(r_attempt)
                    tilt, disp = r["tilt_deg"], r.get("disp_xy_mm", 0.0)
                    rec["fell"] = True
                else:
                    # nothing stood: accept the FALLEN pose of the stabilized attempt —
                    # a resting wrong-orientation object beats a floating "correct" one.
                    tilt, disp = r["tilt_deg"], r.get("disp_xy_mm", 0.0)
                    rec["chosen"], rec["fell"] = "stabilized", True
            elif _unstable(r) and rec["chosen"] == "raw":
                # No stabilization rung (guard-refused sliver contact, or degenerate
                # footprint), a rule-1 flip-accept, or a rule-2 confirmed raw rest:
                # accept the RAW fallen pose. If a pristine attempt ran and failed,
                # the server currently holds the PRISTINE fallen parts — swap back to
                # raw and re-drop so the accepted pose matches the raw glb the bake
                # will use (a rule-2 confirm already restored raw parts).
                if "tilt_pristine" in rec and not rec.get("raw_confirmed_by_pristine"):
                    r, r_attempt = _retry(
                        {"npz": o["parts_npz"], **roll_kv, **_vlm_fields(name)},
                        "ladder_raw_restore",
                    )
                    _retain(r_attempt)
                    tilt, disp = r["tilt_deg"], r.get("disp_xy_mm", 0.0)
                rec["fell"] = True
            if not roll and rec["chosen"] == "pristine":
                # METASTABLE pristine: a body whose NATURAL (uniform-density) tip
                # margin is below the stabilization ladder's own threshold stands a
                # short isolated drop but topples in a long JOINT sim (0725_snapdown
                # vase: pristine 1.9-5.7deg, then TOPPLED at preprocess certify in
                # one run and composition certify in the other). Gate on the
                # INTRINSIC margin, NOT on how the raw drop went — a raw fall often
                # just means a weird SAM3D pose flopping to a natural rest, and a
                # stable-once-upright object must NOT get a physics override. When
                # armed, re-run the pristine rung WITH the bundle (solved on the
                # PRISTINE frame) so it holds the certify AND, via
                # physics_overrides, every composition settle (the boot re-derives
                # the CoM on its fresh collider; friction/damping/flatten reused).
                # chosen stays "pristine": the swap npz IS the pristine parts, so
                # the bake baseline (pristine glb) is unchanged.
                stab_p = _stabilize_params(o["pristine_parts_npz"])
                if stab_p and stab_p["tip_theta_uniform_deg"] >= STABILIZE_THETA_DEG:
                    stab_p = None  # stable under its own mass model: leave it alone
                if stab_p:
                    rr, rr_attempt = _retry({
                        "npz": o["pristine_parts_npz"], "com": stab_p["com_world"],
                        "friction": stab_p["friction"],
                        "damping": stab_p["angular_damping"],
                        "flatten_mm": stab_p["flatten_base_mm"], **roll_kv,
                        **_vlm_fields(name, keep_friction=False),
                    }, "ladder_pristine_stabilized")  # fmt: skip
                    rec["tilt_stabilized"] = rr["tilt_deg"]
                    if not _unstable(rr):
                        r = rr
                        _retain(rr_attempt)
                        tilt, disp = rr["tilt_deg"], rr.get("disp_xy_mm", 0.0)
                        rec["physics_overrides"] = stab_p
                    else:
                        # the bundle didn't hold either: the retry left the body
                        # fallen — restore the accepted plain-pristine rest.
                        r, r_attempt = _retry(
                            {"npz": o["pristine_parts_npz"], **roll_kv,
                             **_vlm_fields(name)},
                            "ladder_pristine_restore",
                        )  # fmt: skip
                        _retain(r_attempt)
                        tilt, disp = r["tilt_deg"], r.get("disp_xy_mm", 0.0)
            rec["ladder_tilt_deg"] = tilt
            rec["ladder_disp_mm"] = disp
            # stabilized rung -> its raw-frame bundle; otherwise PRESERVE a bundle the
            # metastable-pristine branch armed (clobbering it to None here dropped the
            # 0725_tipfix vase's overrides from the record: the live body held certify
            # at 0.06deg but composition would have re-booted it uniform-density).
            rec["physics_overrides"] = (
                stab if rec["chosen"] == "stabilized" else rec.get("physics_overrides")
            )
            o["glb"], rec["glb"] = glb, glb

            # SCENE ENTRY: the same chosen pose released into the full committed
            # scene (release from ~contact vs everyone: lift out of penetration,
            # snap-down when floating). ``ancestors`` arms the server's down-
            # preference: a big lift forced by a NON-ancestor resolves to a clear
            # pose BELOW (chair-under-table) instead of stacking on the neighbor.
            r, r_attempt = _drop(
                {"cmd": "drop", "name": name, "ancestors": anc}, "scene_entry"
            )
            _retain(r_attempt)
            rec["lift_mm"] = r["lift_mm"]
            rec["release_dz_mm"] = round(r.get("release_dz_mm", r["lift_mm"]), 1)
            if r.get("down_cleared"):
                rec["down_cleared"] = True
            rec["tilt_deg"] = rec["cum_tilt_deg"] = r["cum_tilt_deg"]

            # TOPPLE RETRY: it stood in the isolation ladder but fell here because a
            # neighbor forced a big vertical lift+drop. Restore the upright pose and
            # SLIDE it to the nearest clear xy within a HCLEAR_MAX-radius DISK (own
            # support/ancestors excluded), then settle from contact. A disk, not a
            # camera ray: the clear opening is often off-axis to any single ray
            # (0715 wendy1 marker cleared in -x while its ray pointed -y).
            _drift_xy_mm = float(np.hypot(*np.asarray(r["t"], float)[:2])) * 1000.0
            if roll:
                # a rollable rotating in place inflates the frame translation t
                # (rotation is not about the centroid); use the centroid drift
                _drift_xy_mm = float(r.get("disp_xy_mm", _drift_xy_mm))
            rec["drift_xy_mm"] = round(_drift_xy_mm, 1)

            # ATOP-A-NON-PARENT detection + slide-off. supports_of uses a base-
            # underneath test (not resting_on's top-below-base), so it catches a
            # support wedged/tilted so its TOP rises above this object's base — which
            # resting_on misses (wendy1 scissors on a 33deg box). Highest-top non-
            # static, non-ancestor support carrying at least ATOP_FRAC of the footprint.
            def _atop_support():
                sof = (client.rpc({"cmd": "supports_of", "name": name}) or {}).get(
                    "supports", []
                )
                cand = [
                    s for s in sof
                    if not s["static"] and s["name"] not in anc
                    and s.get("footprint_frac", 0.0) >= ATOP_FRAC
                ]  # fmt: skip
                return cand[0]["name"] if cand else None  # supports_of is top-z sorted

            def _slide_off_atop(cur):
                # revert the current settled delta (start upright) then slide to the
                # nearest clear cell OFF the neighbor (ATOP radius, ancestors excluded)
                Rd = np.asarray(cur["R"], float)
                inv = np.eye(4)
                inv[:3, :3], inv[:3, 3] = Rd.T, -Rd.T @ np.asarray(cur["t"], float)
                client.rpc({"cmd": "transform", "name": name, "M": inv.tolist()})
                rr, rr_attempt = _drop(
                    {"cmd": "clear_along", "name": name,
                     "cap": HCLEAR_MAX_ATOP, "exclude": anc},
                    "clear_along_atop",
                )  # fmt: skip
                _retain(rr_attempt)
                rec["lift_mm"] = rr["lift_mm"]
                rec["tilt_deg"] = rec["cum_tilt_deg"] = rr["cum_tilt_deg"]
                rec["ray_push_mm"] = round(rr.get("push_mm", 0.0), 1)
                rec["ray_cleared"] = rec["atop_cleared"] = bool(rr.get("cleared"))
                return rr

            _sup = _atop_support()
            atop_nonparent = _sup is not None
            rec["atop_of"] = _sup
            # "stood in isolation" is class-aware: an accepted in-place roll reads
            # as a huge ladder tilt, so a rollable qualifies via its ladder disp;
            # the entry TOPPLE arm (tilt-based) is likewise meaningless for it —
            # its drift/atop arms still fire.
            _stood = (
                rec["ladder_disp_mm"] <= CAPSIZE_DISP_MM
                if roll
                else rec["ladder_tilt_deg"] <= HCLEAR_LADDER_OK
            )
            if (
                _stood
                and not rec["fell"]
                and (
                    (
                        not roll
                        and r["cum_tilt_deg"] > TILT_CAP
                        and r["lift_mm"] > HCLEAR_LIFT_MM
                    )
                    or _drift_xy_mm > HCLEAR_XY_MM
                    or atop_nonparent
                )
            ):
                if atop_nonparent:
                    r = _slide_off_atop(r)  # detected up front: big-radius slide off
                else:
                    Rd = np.asarray(r["R"], float)
                    inv = np.eye(4)
                    inv[:3, :3], inv[:3, 3] = Rd.T, -Rd.T @ np.asarray(r["t"], float)
                    client.rpc({"cmd": "transform", "name": name, "M": inv.tolist()})
                    rr, rr_attempt = _drop(
                        {"cmd": "clear_along", "name": name,
                         "cap": HCLEAR_MAX, "exclude": anc},
                        "clear_along",
                    )  # fmt: skip
                    _retain(rr_attempt)
                    rec["lift_mm"], r = rr["lift_mm"], rr
                    rec["tilt_deg"] = rec["cum_tilt_deg"] = rr["cum_tilt_deg"]
                    rec["ray_push_mm"] = round(rr.get("push_mm", 0.0), 1)
                    rec["ray_cleared"] = bool(rr.get("cleared"))
                    # POST-CLEAR re-check: a drift/topple clear (small radius, no-
                    # penetration target) can RE-DEPOSIT the object atop a non-parent
                    # (resting on top isn't a penetration), and the pre-clear atop
                    # check can miss it at the transient scene-entry pose. Re-query the
                    # settled pose; if now stacked, slide it off with the ATOP radius
                    # (wendy1 scissors: drift-clear put it back on the box, unchecked).
                    _sup2 = _atop_support()
                    if _sup2 is not None:
                        atop_nonparent = True
                        rec["atop_of"] = _sup2
                        r = _slide_off_atop(r)

            # RIM-TOPPLE NUDGE: still down after the retries, and it topples on the parent
            # but rests flat on a bare slab -> its placed xy sits on the parent's upturned
            # rim. Slide toward the parent centroid (5mm steps, cap 25mm), re-dropping in
            # the full scene; accept the FIRST offset whose tilt <= TILT_CAP. Skips ICP if
            # nudged (its MoGE cloud is at the rim xy and would drag it back). 0722 abc2 knife.
            nudged = False
            # rollables skip the rim nudge: their cum tilt is dominated by the
            # accepted benign roll, so the trigger (and its bare-slab tilt probe)
            # cannot distinguish a rim topple from rolling in place.
            if (
                not roll
                and rec["tilt_deg"] > TILT_CAP
                and anc
                and anc[0] in center_by_name
            ):
                # Every rim probe resets/re-drops the live body. Preserve the exact
                # converged full-scene pose that triggered the probes so a failed search
                # can return to it instead of doing one more reset-and-drop that may land
                # much farther away (0811 abc_5 bottle: 20mm scene-entry -> 85mm restore).
                pre_rim = np.asarray(client.rpc({"cmd": "pose", "name": name})["total"])
                pre_rim_attempt = drop_state["retained"]
                dvec = (center_by_name[anc[0]] - np.asarray(o["center"], float))[:2]
                nrm = float(np.hypot(*dvec))
                u = dvec / nrm if nrm > 1e-6 else np.zeros(2)
                swap_args = _chosen_swap(o, rec)

                def _drop_shifted(scene, mm=0, stage="rim_shift_probe", role="probe"):
                    client.rpc(
                        {"cmd": "swap", "name": name, **swap_args}
                    )  # reset to placed
                    if mm:
                        M = np.eye(4)
                        M[0, 3], M[1, 3] = u[0] * mm / 1000.0, u[1] * mm / 1000.0
                        client.rpc({"cmd": "transform", "name": name, "M": M.tolist()})
                    return _drop(
                        {"cmd": "drop", "name": name, "scene": scene,
                         "ancestors": anc}, stage, role,
                    )  # fmt: skip

                def _rim_tilt(result):
                    # A low-tilt cap-hit is still a snapshot, so it cannot prove a
                    # stable slab or shifted-rim candidate.
                    return (
                        result["cum_tilt_deg"]
                        if bool(result.get("converged", True))
                        else float("inf")
                    )

                # gate: is the topple the parent's fault? (flat on a bare slab)
                bare = None
                if nrm > 1e-6:
                    bare, _ = _drop_shifted([], stage="rim_slab_probe")
                if bare is not None and _rim_tilt(bare) <= TILT_CAP:
                    mm = _first_stable_shift(
                        lambda m: _rim_tilt(_drop_shifted(None, m)[0])
                    )
                    if mm is not None:
                        rr, rr_attempt = _drop_shifted(
                            None, mm, stage="rim_accept", role="candidate"
                        )  # land on the accepted offset
                        if _rim_tilt(rr) <= TILT_CAP:
                            _retain(rr_attempt)
                            rec["rim_nudge_mm"] = mm
                            rec["tilt_deg"] = rec["cum_tilt_deg"] = rr["cum_tilt_deg"]
                            rec["lift_mm"] = rr["lift_mm"]
                            nudged = True
                if not nudged:  # no offset cleared it -> restore the proven prior rest
                    cur = np.asarray(client.rpc({"cmd": "pose", "name": name})["total"])
                    Mrev = pre_rim @ np.linalg.inv(cur)
                    client.rpc({"cmd": "transform", "name": name, "M": Mrev.tolist()})
                    if pre_rim_attempt is not None:
                        _retain(pre_rim_attempt)

            # interleaved ICP on the settled pose (yaw/x/y/scale; z owned by re-descent).
            # SKIP for an object we slid off a non-parent (or rim-nudged): its MoGE cloud
            # sits at the original xy and ICP would drag it straight back.
            cstats: dict = {}
            cloud = object_cloud_for(
                o["placement"], out_dir, points, valid, R, T, stats=cstats
            )
            # icp_locked: the object's pose came from a calibrated estimator (asset
            # matched + --asset-pose foundationpose); the MoGE-cloud ICP would only
            # re-fit yaw/xy against the same evidence, so it is skipped per object.
            icp_locked = bool(o["placement"].get("icp_locked"))
            if icp_locked:
                rec["icp_skipped"] = "icp_locked"
            if (
                cloud is not None
                and not atop_nonparent
                and not nudged
                and not disable_icp
                and not icp_locked
            ):
                pre_icp = np.asarray(client.rpc({"cmd": "pose", "name": name})["total"])
                mesh = load_placed_mesh(glb)
                mesh.vertices = mesh.vertices @ pre_icp[:3, :3].T + pre_icp[:3, 3]
                # Same-size objects carry a frozen shared size: align yaw/xy only, never
                # rescale (scale locked to 1). --icp-freeze DOFs stack on top.
                icp_kw = dict(freeze_kw)
                if o["placement"].get("same_size") or o["placement"].get(
                    "scale_locked"
                ):
                    icp_kw.update(scale_range=(1.0, 1.0), scale_blend=0.0)
                try:
                    fit = fit_similarity(mesh, cloud, cloud_stats=cstats, **icp_kw)
                except Exception as e:  # noqa: BLE001 - per-object best effort
                    fit = None
                    icp_log.append(
                        {"obj": name.removeprefix("obj_"), "skipped": f"ERR {e}"}
                    )
                if fit:
                    fit_log = {
                        "obj": name.removeprefix("obj_"),
                        **{
                            k: fit[k]
                            for k in (
                                "n_pts",
                                "rms_before",
                                "rms_after",
                                "yaw_deg",
                                "scale",
                                "t",
                                "dz_dropped",
                                "accepted",
                                "kept_frac",
                                "footprint_frac",
                                "scale_at_cap",
                            )
                            if k in fit
                        },
                        "skipped": fit["skipped"],
                    }
                    icp_log.append(fit_log)  # fmt: skip
                    if fit["accepted"]:
                        C = correction_matrix(fit, bottom_center_pivot(mesh.vertices))
                        client.rpc({"cmd": "transform", "name": name, "M": C.tolist()})
                        # overlap budget: alignment must not buy a costly lift — if the
                        # corrected pose needs > 3cm of clearance, skip ICP for this object.
                        dz = client.rpc({"cmd": "clearance", "name": name})["dz"]
                        if dz > 0.03:
                            Ci = np.linalg.inv(C)
                            client.rpc(
                                {"cmd": "transform", "name": name, "M": Ci.tolist()}
                            )
                            rec["icp_skipped_overlap_mm"] = round(dz * 1000.0, 1)
                            fit_log["physics_status"] = "skipped_overlap"
                        else:
                            r, r_attempt = _drop(
                                {"cmd": "drop", "name": name, "ancestors": anc},
                                "icp_redrop",
                            )  # re-descent
                            reject_reason = icp_redrop_rejection_reason(
                                r, roll, bool(rec["fell"])
                            )
                            if reject_reason is not None:
                                # The alignment's physical validation regressed: return
                                # exactly to the converged pose captured before ICP.
                                cur = np.asarray(
                                    client.rpc({"cmd": "pose", "name": name})["total"]
                                )
                                Mrev = pre_icp @ np.linalg.inv(cur)
                                client.rpc({"cmd": "transform", "name": name,
                                            "M": Mrev.tolist()})  # fmt: skip
                                rec["icp_reverted_tilt_deg"] = round(
                                    r["cum_tilt_deg"], 1
                                )
                                rec["icp_reverted_disp_mm"] = round(
                                    float(r.get("disp_xy_mm", 0.0)), 1
                                )
                                rec["icp_reverted_reason"] = reject_reason
                                fit_log["physics_status"] = "reverted"
                                fit_log["physics_reverted_reason"] = reject_reason
                            else:
                                _retain(r_attempt)
                                rec["redrop_dz_mm"] = float(r["t"][2]) * 1000.0
                                rec["icp"] = {
                                    k: fit[k] for k in ("yaw_deg", "scale", "t")
                                }
                                rec["cum_tilt_deg"] = r["cum_tilt_deg"]
                                fit_log["physics_status"] = "retained"
            _finalize_drop_bookkeeping(
                rec, drop_state["attempts"], drop_state["retained"]
            )
            records[name] = rec
            print(f"[incremental-settle] {name}: {rec['chosen']}"
                  + (" ROLLABLE" if rec.get("rollable") else "")
                  + (" FELL" if rec["fell"] else "")
                  + (" SETTLE-FAILED" if rec.get("settle_failed") else "")
                  + (" SETTLE-RECOVERED" if rec.get("settle_recovered") else "")
                  + (" DROP-EXTENDED" if rec.get("drop_continued") else "")
                  + f" ladder {rec['ladder_tilt_deg']:.1f}deg cum {rec['cum_tilt_deg']:.1f}deg"
                  + (f" disp {rec['ladder_disp_mm']:.0f}mm"
                     if rec.get("rollable") and rec.get("ladder_disp_mm") is not None
                     else "")
                  + f" lift {rec['lift_mm']:.0f}mm"
                  + (" ICP-SKIP" if "icp_skipped_overlap_mm" in rec else "")
                  + (" ICP-REVERT" if "icp_reverted_tilt_deg" in rec else ""),
                  flush=True)  # fmt: skip
        # pre-certify totals of every body the certify-topple rescue could act on
        # (raw stander, no bundle, marginal tip margin): swap resets ``total`` to the
        # placed pose, so this is what puts a rescued body back where certify found it.
        by_name = {o["name"]: o for o in ordered}
        watch = certify_rescue_candidates(
            {n: {"tilt_deg": float("inf")} for n in records}, ordered, records
        )
        pre_cert_totals = {
            n: np.asarray(client.rpc({"cmd": "pose", "name": n})["total"], float)
            for n in watch
        }
        cert = client.rpc({"cmd": "certify"})
        rescue = certify_rescue_candidates(cert["drift"], ordered, records)
        if rescue:
            for name in rescue:
                o, stab = by_name[name], by_name[name]["stabilize"]
                client.rpc({
                    "cmd": "swap", "name": name, "npz": o["parts_npz"],
                    "com": stab["com_world"], "friction": stab["friction"],
                    "damping": stab["angular_damping"],
                    "flatten_mm": stab["flatten_base_mm"],
                    **_vlm_fields(name, keep_friction=False),
                })  # fmt: skip
                client.rpc({"cmd": "transform", "name": name,
                            "M": pre_cert_totals[name].tolist()})  # fmt: skip
            before = {n: float(cert["drift"][n]["tilt_deg"]) for n in rescue}
            cert = client.rpc({"cmd": "certify"})
            for name in rescue:
                after = float((cert["drift"].get(name) or {}).get("tilt_deg", 0.0))
                held = after <= CAPSIZE_DEG
                records[name]["certify_rescue"] = {
                    "tilt_before_deg": round(before[name], 1),
                    "tilt_after_deg": round(after, 1),
                    "held": held,
                }
                if held:
                    records[name]["chosen"] = "stabilized"
                    records[name]["physics_overrides"] = by_name[name]["stabilize"]
                print(
                    f"[incremental-settle] {name}: toppled {before[name]:.1f}deg in the "
                    f"joint certify after standing the isolated ladder; re-certified with "
                    f"its CoM bundle -> {after:.1f}deg "
                    + (
                        "(bundle kept as physics_overrides)"
                        if held
                        else "(still down; bundle NOT kept)"
                    ),
                    flush=True,
                )
        for name, d in cert["drift"].items():
            records[name]["certify"] = d
        totals = {n: np.asarray(m) for n, m in cert["total"].items()}
        # o.total is cumulative from the PLACED pose (ladder + ICP + re-drop + this
        # certify) — the left half of the delivered-vs-placed measure. Persisted here
        # because the composition certify runs in another process on a fresh session
        # whose own totals restart at the composed pose.
        for name, M in totals.items():
            if name in records:
                records[name]["settle_total"] = M.tolist()
        extended = sorted(n for n, rr in records.items() if rr.get("drop_continued"))
        recovered = sorted(n for n, rr in records.items() if rr.get("settle_recovered"))
        failed = sorted(n for n, rr in records.items() if rr.get("settle_failed"))
        cert_converged = bool(cert.get("converged", True))
        print(
            "[incremental-settle] convergence summary: "
            f"{len(extended)} extended, {len(recovered)} recovered by a later drop, "
            f"{len(failed)} entered certify without a demonstrated rest; "
            f"joint certify converged={cert_converged}",
            flush=True,
        )
        for name in failed:
            d = cert["drift"].get(name, {})
            outcome = (
                "late-settled in joint certify"
                if cert_converged
                else "not cleared by joint certify"
            )
            print(
                f"[incremental-settle] {name}: {outcome}; "
                f"dxy={float(d.get('dxy', 0.0)) * 1000.0:.1f}mm "
                f"tilt={float(d.get('tilt_deg', 0.0)):.1f}deg",
                flush=True,
            )
    finally:
        client.disconnect()  # server stays warm for composition/certify

    # single texture-preserving bake of each cumulative matrix + placement update
    jobs, pending = [], []
    for o in ordered:
        name = o["name"]
        M = totals.get(name)
        if M is None:
            continue
        out_glb = o["glb"][:-4] + "_pm.glb"
        jobs.append(
            {
                "glb_in": _bake_baseline_glb(o, records[name]),  # honor swap baselines
                "glb_out": out_glb,
                "matrix": M.tolist(),
            }
        )
        pending.append((o["placement"], M, out_glb, records[name]))
    bake_world_matrices(jobs, out_dir, blender_cmd)
    for p, M, out_glb, rec in pending:
        p["mesh_glb"] = out_glb
        registered = p.get("asset_registration_to_world")
        if registered is not None:
            p["asset_settle_delta_world"] = M.tolist()
            p["asset_canonical_to_settled_world"] = (
                M @ np.asarray(registered, dtype=np.float64)
            ).tolist()
        if p.get("center"):
            c = M @ np.array([*p["center"], 1.0])
            p["center"] = [float(v) for v in c[:3]]
        sc = (rec.get("icp") or {}).get("scale")
        if sc and p.get("size"):
            p["size"] = [float(v) * float(sc) for v in p["size"]]

    with open(os.path.join(out_dir, "pose_match.json"), "w") as f:
        json.dump(icp_log, f, indent=2)
    write_pose_changes(work, run_id_for(out_dir), records)
    return records
