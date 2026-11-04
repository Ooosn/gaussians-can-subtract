"""Mesh-aware child placement for the main Gaussian densification path.

The projector is intentionally a small, CPU-side query object.  It is used
only for selected main-GS split parents, while the NDF/gaussolve model keeps
the existing split path.  The mesh is frozen during the normal densification
window (before the 16k joint mesh phase), so one immutable query is sufficient
for the short diagnostic and the full training contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh


class MeshAwareSplitProjector:
    """Prevent split children from crossing nearby mesh triangle planes.

    ``nearest_faces`` is deliberately small (1, 2, or 3).  The default one
    face preserves the light-weight historical behavior; checking two or
    three faces is a more conservative option near a mesh edge/corner.
    """

    def __init__(
        self,
        mesh_path: str | Path,
        scene_extent: float,
        tau_factor: float = 0.005,
        nearest_faces: int = 1,
    ) -> None:
        if int(nearest_faces) not in (1, 2, 3):
            raise ValueError("nearest_faces must be one of {1, 2, 3}")
        self.mesh_path = str(mesh_path)
        self.scene_extent = float(scene_extent)
        self.tau = float(tau_factor) * self.scene_extent
        self.nearest_faces = int(nearest_faces)
        loaded = trimesh.load(self.mesh_path, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected a triangle mesh at {self.mesh_path}")
        if len(loaded.faces) == 0 or len(loaded.vertices) == 0:
            raise ValueError(f"Empty mesh at {self.mesh_path}")
        self.mesh = loaded
        self.query = trimesh.proximity.ProximityQuery(self.mesh)
        self.triangles = np.asarray(self.mesh.triangles, dtype=np.float64)
        # R-tree lookup is used only for K>1.  K=1 keeps ProximityQuery's
        # exact nearest-surface implementation and its existing behavior.
        self.triangles_tree = None
        self.face_normals = np.asarray(self.mesh.face_normals, dtype=np.float64)
        self.calls = 0
        self.parent_queries = 0
        self.eligible_parents = 0
        self.crossing_parents = 0
        self.adjusted_children = 0
        self.projection_fallback_children = 0

    def _nearest_planes(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return closest points, unit normals, and face ids for K faces.

        The R-tree returns a small candidate set by triangle bounding-box
        distance.  We then compute exact point-to-triangle distances for that
        set and always include the exact global nearest face from
        ``ProximityQuery``.  This avoids treating a nearby triangle's box as
        an exact surface distance.
        """
        points = np.asarray(points, dtype=np.float64)
        n_points = int(points.shape[0])
        k = self.nearest_faces
        if k == 1:
            closest, _distance, triangle_id = self.query.on_surface(points)
            triangle_id = np.asarray(triangle_id, dtype=np.int64).reshape(n_points)
            normals = self.face_normals[triangle_id]
            normals /= np.maximum(
                np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
            )
            return (
                np.asarray(closest, dtype=np.float64)[:, None, :],
                normals[:, None, :],
                triangle_id[:, None],
            )

        closest_out = np.empty((n_points, k, 3), dtype=np.float64)
        normals_out = np.empty((n_points, k, 3), dtype=np.float64)
        ids_out = np.empty((n_points, k), dtype=np.int64)
        candidate_count = max(8, k * 4)
        triangles_tree = self.mesh.triangles_tree
        self.triangles_tree = triangles_tree
        # The exact nearest face is queried in one batch.  Only the extra
        # K-1 candidates need per-point R-tree lookups.
        _exact_closest_all, _distance_all, exact_ids_all = self.query.on_surface(
            points
        )
        exact_ids_all = np.asarray(exact_ids_all, dtype=np.int64).reshape(n_points)
        for point_index, point in enumerate(points):
            exact_id = int(exact_ids_all[point_index])
            candidate_ids = [exact_id]
            candidate_ids.extend(
                int(face_id)
                for face_id in triangles_tree.nearest(
                    tuple(float(value) for value in point),
                    num_results=candidate_count,
                )
            )
            candidate_ids = list(dict.fromkeys(candidate_ids))
            if not candidate_ids:
                raise RuntimeError("mesh triangle query returned no candidates")
            candidate_triangles = self.triangles[candidate_ids]
            candidate_points = np.repeat(point[None, :], len(candidate_ids), axis=0)
            candidate_closest = trimesh.triangles.closest_point(
                candidate_triangles, candidate_points
            )
            candidate_distance2 = np.sum(
                (candidate_closest - candidate_points) ** 2, axis=1
            )
            order = np.argsort(candidate_distance2, kind="stable")
            selected_ids = [candidate_ids[int(index)] for index in order[:k]]
            if len(selected_ids) < k:
                selected_ids.extend([selected_ids[-1]] * (k - len(selected_ids)))
            selected_ids_array = np.asarray(selected_ids, dtype=np.int64)
            selected_triangles = self.triangles[selected_ids_array]
            selected_points = np.repeat(point[None, :], k, axis=0)
            selected_closest = trimesh.triangles.closest_point(
                selected_triangles, selected_points
            )
            selected_normals = self.face_normals[selected_ids_array].copy()
            selected_normals /= np.maximum(
                np.linalg.norm(selected_normals, axis=1, keepdims=True), 1e-12
            )
            closest_out[point_index] = selected_closest
            normals_out[point_index] = selected_normals
            ids_out[point_index] = selected_ids_array
        return closest_out, normals_out, ids_out

    def adjust_child_positions(
        self,
        parent_xyz: torch.Tensor,
        child_xyz: torch.Tensor,
        children_per_parent: int,
    ) -> tuple[torch.Tensor, dict[str, int | float]]:
        """Return child positions satisfying every checked mesh half-space.

        ``child_xyz`` follows GaussianModel's existing repeat layout: all
        children for the first repeat, followed by all children for the next
        repeat.  For a parent whose split can reach its nearest triangle, any
        child on the opposite side is moved along the triangle normal back to
        the parent plane.  Tangential displacement and all far parents remain
        unchanged.
        """
        if children_per_parent <= 0:
            raise ValueError("children_per_parent must be positive")
        parent_count = int(parent_xyz.shape[0])
        if parent_count == 0:
            return child_xyz, {
                "surface_parent_queries": 0,
                "surface_eligible_parents": 0,
                "surface_crossing_parents": 0,
                "surface_adjusted_children": 0,
                "surface_nearest_faces": self.nearest_faces,
                "surface_checked_planes": 0,
                "surface_projection_fallback_children": 0,
            }
        expected = parent_count * int(children_per_parent)
        if int(child_xyz.shape[0]) != expected:
            raise ValueError(
                f"child/parent shape mismatch: {int(child_xyz.shape[0])} != {expected}"
            )

        parent_np = parent_xyz.detach().float().cpu().numpy()
        closest, normals, _triangle_ids = self._nearest_planes(parent_np)
        signed_parent = np.sum(
            (parent_np[:, None, :] - closest) * normals, axis=2
        )

        # Keep the operation on CPU: this branch runs after backward and is
        # copied back once, avoiding a mesh query inside the CUDA renderer.
        child_np = child_xyz.detach().float().cpu().numpy().reshape(
            int(children_per_parent), parent_count, 3
        )
        offsets = child_np - parent_np[None, :, :]
        normal_offsets = np.einsum("cni,nki->ckn", offsets, normals)
        child_signed = signed_parent.T[None, :, :] + normal_offsets
        max_normal_offset = np.max(np.abs(normal_offsets), axis=0)
        eligible = np.any(
            np.abs(signed_parent.T) <= (max_normal_offset + self.tau), axis=0
        )

        # The side is defined by the Gaussian center, not by its scale.  A
        # center exactly on a plane uses the deterministic positive-side tie
        # break; there is no near-plane exemption.
        side = np.where(signed_parent >= 0.0, 1.0, -1.0)
        eps = max(self.tau * 1e-6, 1e-8)
        violations = side.T[None, :, :] * child_signed < -eps
        crossing = np.any(violations, axis=(0, 1))

        adjusted = child_np.copy()
        adjusted_count = 0
        fallback_count = 0
        # Cyclic half-space projection: each offending child is moved only
        # along a checked triangle normal.  A few passes handle edge/corner
        # constraints; fallback to the parent center makes the final
        # same-side contract unconditional even for non-convex local planes.
        max_passes = max(4, self.nearest_faces * 4)
        # Minimal correction: a violating child is clipped to epsilon on the
        # parent's side of the plane.  Do not move it back to the parent's
        # signed distance (that can over-shoot by a large amount and erase the
        # purpose of split); tangential displacement and scale stay intact.
        target_margin = np.full_like(signed_parent, eps)
        for child_index in range(int(children_per_parent)):
            original = adjusted[child_index].copy()
            child = adjusted[child_index]
            for _pass in range(max_passes):
                current_signed = np.sum(
                    (child[:, None, :] - closest) * normals, axis=2
                )
                constraint = side * current_signed
                changed = False
                for face_index in range(self.nearest_faces):
                    # Only an actual opposite-side center is corrected.  A
                    # child that is already on the parent's side but closer
                    # to the plane is left in place.
                    mask = constraint[:, face_index] < -eps
                    if not np.any(mask):
                        continue
                    correction = (
                        target_margin[mask, face_index]
                        - constraint[mask, face_index]
                    )[:, None] * side[mask, face_index][:, None] * normals[
                        mask, face_index
                    ]
                    child[mask] += correction
                    changed = True
                if not changed:
                    break
            final_signed = np.sum(
                (child[:, None, :] - closest) * normals, axis=2
            )
            final_constraint = side * final_signed
            remaining = np.any(final_constraint < -eps, axis=1)
            if np.any(remaining):
                child[remaining] = parent_np[remaining]
                fallback_count += int(remaining.sum())
            changed_mask = np.linalg.norm(child - original, axis=1) > eps
            adjusted_count += int(changed_mask.sum())

        final_signed_all = np.sum(
            (adjusted[:, :, None, :] - closest[None, :, :, :]) * normals[None, :, :, :],
            axis=3,
        )
        final_constraint_all = side[None, :, :] * final_signed_all
        if np.any(final_constraint_all < -eps):
            raise RuntimeError("mesh-aware split failed same-side projection")

        self.calls += 1
        self.parent_queries += parent_count
        self.eligible_parents += int(eligible.sum())
        self.crossing_parents += int(crossing.sum())
        self.adjusted_children += adjusted_count
        self.projection_fallback_children += fallback_count
        stats: dict[str, int | float | str] = {
            "surface_parent_queries": parent_count,
            "surface_eligible_parents": int(eligible.sum()),
            "surface_crossing_parents": int(crossing.sum()),
            "surface_adjusted_children": adjusted_count,
            "surface_nearest_faces": self.nearest_faces,
            "surface_checked_planes": parent_count * self.nearest_faces,
            "surface_projection_fallback_children": fallback_count,
            "surface_tau": self.tau,
            "surface_projection_mode": "minimal_same_side",
        }
        result = torch.as_tensor(
            adjusted.reshape(expected, 3),
            device=child_xyz.device,
            dtype=child_xyz.dtype,
        )
        return result, stats

    def summary(self) -> dict[str, int | float | str]:
        return {
            "mesh_path": self.mesh_path,
            "scene_extent": self.scene_extent,
            "tau": self.tau,
            "calls": self.calls,
            "parent_queries": self.parent_queries,
            "eligible_parents": self.eligible_parents,
            "crossing_parents": self.crossing_parents,
            "adjusted_children": self.adjusted_children,
            "nearest_faces": self.nearest_faces,
            "projection_fallback_children": self.projection_fallback_children,
        }
