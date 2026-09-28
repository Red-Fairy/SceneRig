"""Conservative support witnesses from measured PhysX contact pairs.

This module is pure: the verifier supplies contact events and collider transforms.
Cached sleeping contacts remain usable only while their two local anchors still
coincide at the final geometry. A bounding box or sleep flag is not a witness.
"""

from __future__ import annotations

import numpy as np


def _valid_transforms(matrices: list[np.ndarray]) -> bool:
    """Reject malformed native transforms before inverse-anchor calculations."""
    return all(
        m.shape == (4, 4)
        and np.isfinite(m).all()
        and np.array_equal(m[3], [0, 0, 0, 1])
        and np.isfinite(np.linalg.det(m))
        and np.linalg.det(m) != 0
        for m in matrices
    )


class ContactSupport:
    """Keep the latest supporting contacts until their collider pair is lost."""

    def __init__(self) -> None:
        self.pairs: dict[tuple[str, str], list[dict]] = {}
        self.error: str | None = None
        self.event_count = 0

    def update(self, events: list[dict], colliders: dict[str, dict], step: int) -> None:
        """Consume native events after the step's transform writeback.

        Args:
            events: Actor/collider paths and world-space native contact data.
            colliders: Current enabled collision shapes, identities and matrices.
            step: Existing verification step number; no extra simulation is run.
        """
        for event in events:
            self.event_count += 1
            paths = event["colliders"]
            key = tuple(sorted(paths))
            self.pairs.pop(key, None)
            if event["type"] == "lost" or len(set(paths)) != 2:
                continue
            if any(path not in colliders for path in paths):
                continue
            shapes = [colliders[path] for path in paths]
            if shapes[0]["owner"] == shapes[1]["owner"]:
                continue
            if any(
                actor != shape["actor_path"]
                for actor, shape in zip(event["actors"], shapes)
            ):
                continue
            records = []
            for contact in event["contacts"]:
                point = np.asarray(contact["position"], dtype=float)
                normal = np.asarray(contact["normal"], dtype=float)
                impulse = np.asarray(contact["impulse"], dtype=float)
                separation = float(contact["separation"])
                if any(v.shape != (3,) for v in (point, normal, impulse)):
                    continue
                if not np.isfinite(np.r_[point, normal, impulse, separation]).all():
                    continue
                if not -0.02 <= separation <= 0.03:
                    continue
                for i, sign in ((0, 1), (1, -1)):
                    own, other = shapes[i], shapes[1 - i]
                    force, direction = sign * impulse, sign * normal
                    if own["static"] or force[2] <= 0 or direction[2] <= 0:
                        continue
                    matrices = [
                        np.asarray(s["matrix"], dtype=float) for s in (own, other)
                    ]
                    if not _valid_transforms(matrices):
                        continue
                    anchors = [np.linalg.solve(m, np.r_[point, 1]) for m in matrices]
                    records.append(
                        {
                            "body": own["owner"],
                            "other": other["owner"],
                            "paths": [paths[i], paths[1 - i]],
                            "geometry": [own["geometry"], other["geometry"]],
                            "anchors": [a.tolist() for a in anchors],
                            "normal_local": (
                                matrices[0][:3, :3].T @ direction
                            ).tolist(),
                            "support_normal_local": (
                                matrices[1][:3, :3].T @ direction
                            ).tolist(),
                            "point": point.tolist(),
                            "separation": separation,
                            "upward_impulse": float(force[2]),
                            "step": step,
                        }
                    )
            if records:
                self.pairs[key] = records

    def support_for(
        self,
        name: str,
        colliders: dict[str, dict],
        motion_ok: dict[str, bool],
        sleeping: dict[str, bool],
        step: int,
    ) -> dict | None:
        """Return a current upward-contact witness, or leave support unproved.

        Args:
            name: Pipeline rigid-body identity being verified.
            colliders: Final collision shapes including fresh geometry identities.
            motion_ok: Existing per-body motion/rest gate, excluding support.
            sleeping: Live PhysX sleeping status, never inferred from USD velocity.
            step: Final verification step number.

        Returns:
            Serializable contact evidence for another support, or None.
        """
        if self.error:
            return None
        for records in self.pairs.values():
            for record in records:
                if record["body"] != name:
                    continue
                paths = record["paths"]
                if any(path not in colliders for path in paths):
                    continue
                own, other = [colliders[path] for path in paths]
                if [own["geometry"], other["geometry"]] != record["geometry"]:
                    continue
                if own["owner"] != name or other["owner"] != record["other"]:
                    continue
                if not other["static"] and not motion_ok.get(other["owner"], False):
                    continue
                cached = record["step"] < step - 1
                if cached and not (
                    sleeping.get(name) is True
                    and (other["static"] or sleeping.get(other["owner"]) is True)
                ):
                    continue
                matrices = [np.asarray(s["matrix"], dtype=float) for s in (own, other)]
                if not _valid_transforms(matrices):
                    continue
                anchors = [m @ a for m, a in zip(matrices, record["anchors"])]
                # Only float32 transform-writeback roundoff is admitted here;
                # this is not a new physical gap or motion allowance.
                scale = max(
                    1.0, float(np.max(np.abs(anchors))), max(map(abs, record["point"]))
                )
                tolerance = 32 * float(np.finfo(np.float32).eps) * scale
                distance = float(np.linalg.norm(anchors[0][:3] - anchors[1][:3]))
                if not np.isfinite(distance) or distance > tolerance:
                    continue
                normal = np.linalg.solve(matrices[0][:3, :3].T, record["normal_local"])
                length = float(np.linalg.norm(normal))
                if not np.isfinite(length) or length == 0 or normal[2] <= 0:
                    continue
                normal /= length
                support_normal = np.linalg.solve(
                    matrices[1][:3, :3].T, record["support_normal_local"]
                )
                support_length = float(np.linalg.norm(support_normal))
                if not np.isfinite(support_length) or support_length == 0:
                    continue
                support_normal /= support_length
                if (
                    np.linalg.norm(normal - support_normal)
                    > 32 * np.finfo(np.float32).eps
                ):
                    continue
                separation = record["separation"] + float(
                    np.dot(anchors[0][:3] - anchors[1][:3], normal)
                )
                if not -0.02 <= separation <= 0.03:
                    continue
                return {
                    "name": other["owner"],
                    "method": "physx_contact",
                    "collider": paths[0],
                    "support_collider": paths[1],
                    "separation_m": separation,
                    "normal_on_body": normal.tolist(),
                    "upward_impulse_ns": record["upward_impulse"],
                    "observed_step": record["step"],
                    "sleeping_contact_retained": cached,
                    "anchor_error_m": distance,
                    "anchor_roundoff_bound_m": tolerance,
                }
        return None
