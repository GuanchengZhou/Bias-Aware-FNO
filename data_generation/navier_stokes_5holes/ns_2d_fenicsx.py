#!/usr/bin/env python3
"""Generate five-hole Navier-Stokes data with gmsh + dolfinx + PETSc."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fenicsx_runtime import sanitize_current_process_build_env

sanitize_current_process_build_env(verbose=True)

import gmsh
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem, geometry
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

try:
    from dolfinx.io.gmsh import model_to_mesh
except Exception:
    from dolfinx.io import gmshio

    model_to_mesh = gmshio.model_to_mesh


TAG_NOSLIP = 1
TAG_LEFT = 2
TAG_RIGHT = 3
TAG_DOMAIN = 10
MAT_V5_BYTE_LIMIT = np.iinfo(np.uint32).max


def format_float_token(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("+", "")


def default_file_name(split: str, n_samples: int, nu: float, eta: float, record_steps: int, resolution: int) -> str:
    return (
        f"ns_5holes_V{format_float_token(nu)}_E{format_float_token(eta)}_"
        f"N{n_samples}_T{record_steps}_R{resolution}_{split}.mat"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FEniCSx five-hole Navier-Stokes datasets.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-ood", type=int, default=0)
    parser.add_argument("--grid-resolution", type=int, default=256)
    parser.add_argument("--record-steps", type=int, default=20)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument(
        "--eta",
        type=float,
        default=5e-3,
        help="Unused compatibility field retained in filenames so the FNO scripts keep the same dataset path convention.",
    )
    parser.add_argument("--mesh-size-min", type=float, default=0.05)
    parser.add_argument("--mesh-size-max", type=float, default=0.08)
    parser.add_argument("--gamma-drive", type=float, default=0.25)
    parser.add_argument("--forcing-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-holes", type=int, default=5)
    parser.add_argument("--radius-min", type=float, default=0.05)
    parser.add_argument("--radius-max", type=float, default=0.08)
    parser.add_argument("--boundary-margin", type=float, default=0.05)
    parser.add_argument("--hole-gap-min", type=float, default=0.03)
    parser.add_argument("--max-placement-attempts", type=int, default=2000)
    parser.add_argument("--stream-modes", type=int, default=4)
    parser.add_argument("--stream-scale", type=float, default=1.0)
    parser.add_argument("--save-aux-fields", action="store_true")
    parser.add_argument("--ood-radius-min", type=float, default=None)
    parser.add_argument("--ood-radius-max", type=float, default=None)
    parser.add_argument("--ood-hole-gap-min", type=float, default=None)
    parser.add_argument(
        "--file-format",
        choices=["auto", "mat", "hdf5"],
        default="auto",
        help="Dataset storage format. 'auto' writes MAT v5 when possible and falls back to HDF5-backed .mat for large arrays.",
    )
    return parser.parse_args()


def build_split_spec(args: argparse.Namespace) -> list[tuple[str, int, tuple[float, float], float]]:
    specs = [
        ("train", int(args.n_train), (float(args.radius_min), float(args.radius_max)), float(args.hole_gap_min)),
        ("test", int(args.n_test), (float(args.radius_min), float(args.radius_max)), float(args.hole_gap_min)),
    ]
    if int(args.n_ood) > 0:
        ood_radius_min = float(args.ood_radius_min if args.ood_radius_min is not None else args.radius_min * 1.1)
        ood_radius_max = float(args.ood_radius_max if args.ood_radius_max is not None else args.radius_max * 1.2)
        ood_gap = float(args.ood_hole_gap_min if args.ood_hole_gap_min is not None else max(0.0, args.hole_gap_min * 0.75))
        specs.append(("ood", int(args.n_ood), (ood_radius_min, ood_radius_max), ood_gap))
    return specs


def sample_theta(
    rng: np.random.Generator,
    n_holes: int,
    radius_range: tuple[float, float],
    boundary_margin: float,
    hole_gap_min: float,
    max_attempts: int,
) -> np.ndarray:
    centers: list[tuple[float, float, float]] = []
    radius_min, radius_max = radius_range
    for _ in range(n_holes):
        placed = False
        for _attempt in range(max_attempts):
            radius = float(rng.uniform(radius_min, radius_max))
            low = boundary_margin + radius
            high = 1.0 - boundary_margin - radius
            if low >= high:
                raise ValueError("Geometry constraints are infeasible; reduce the boundary margin or radius range.")
            cx = float(rng.uniform(low, high))
            cy = float(rng.uniform(low, high))
            ok = True
            for ox, oy, oradius in centers:
                if math.hypot(cx - ox, cy - oy) < radius + oradius + hole_gap_min:
                    ok = False
                    break
            if ok:
                centers.append((cx, cy, radius))
                placed = True
                break
        if not placed:
            raise RuntimeError("Failed to sample non-overlapping holes.")
    return np.asarray(centers, dtype=np.float64)


def sample_theta_batch(
    n_samples: int,
    rng: np.random.Generator,
    n_holes: int,
    radius_range: tuple[float, float],
    boundary_margin: float,
    hole_gap_min: float,
    max_attempts: int,
) -> np.ndarray:
    theta = np.empty((n_samples, n_holes, 3), dtype=np.float64)
    for idx in range(n_samples):
        theta[idx] = sample_theta(rng, n_holes, radius_range, boundary_margin, hole_gap_min, max_attempts)
    return theta


def build_mesh(theta: np.ndarray, mesh_size_min: float, mesh_size_max: float):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("five_hole_ns")

    rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    disks = [gmsh.model.occ.addDisk(float(cx), float(cy), 0.0, float(r), float(r)) for cx, cy, r in theta]
    fluid, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk) for disk in disks], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    curves = gmsh.model.getBoundary(fluid, oriented=False, recursive=False)
    left: list[int] = []
    right: list[int] = []
    noslip: list[int] = []
    for dim, tag in curves:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        x = com[0]
        if np.isclose(x, 0.0):
            left.append(tag)
        elif np.isclose(x, 1.0):
            right.append(tag)
        else:
            noslip.append(tag)

    gmsh.model.addPhysicalGroup(1, noslip, tag=TAG_NOSLIP)
    gmsh.model.addPhysicalGroup(1, left, tag=TAG_LEFT)
    gmsh.model.addPhysicalGroup(1, right, tag=TAG_RIGHT)
    surfaces = [entity[1] for entity in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfaces, tag=TAG_DOMAIN)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    gmsh.model.mesh.generate(2)

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()

    if hasattr(mesh_data, "mesh"):
        return mesh_data.mesh, mesh_data.facet_tags
    if isinstance(mesh_data, (tuple, list)):
        return mesh_data[0], mesh_data[2] if len(mesh_data) > 2 else None
    return mesh_data, None


def g(x):
    return x**2 * (1.0 - x) ** 2


def gp(x):
    return 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x)


def sample_stream_coefficients(rng: np.random.Generator, n_modes: int, scale: float) -> np.ndarray:
    kx = np.arange(1, n_modes + 1, dtype=np.float64)[:, None]
    ky = np.arange(1, n_modes + 1, dtype=np.float64)[None, :]
    decay = (kx**2 + ky**2) ** 1.5
    coeffs = rng.normal(scale=scale, size=(n_modes, n_modes)) / decay
    return coeffs.astype(np.float64)


def build_initial_velocity_callable(coeffs: np.ndarray):
    n_modes = coeffs.shape[0]
    kx = np.arange(1, n_modes + 1, dtype=np.float64)[:, None, None]
    ky = np.arange(1, n_modes + 1, dtype=np.float64)[None, :, None]
    coeffs = coeffs[:, :, None]

    def velocity(x):
        xs = x[0][None, None, :]
        ys = x[1][None, None, :]
        sinx = np.sin(np.pi * kx * xs)
        cosx = np.cos(np.pi * kx * xs)
        siny = np.sin(np.pi * ky * ys)
        cosy = np.cos(np.pi * ky * ys)

        stream_sum = np.sum(coeffs * sinx * siny, axis=(0, 1))
        stream_dx = np.sum(coeffs * (np.pi * kx) * cosx * siny, axis=(0, 1))
        stream_dy = np.sum(coeffs * (np.pi * ky) * sinx * cosy, axis=(0, 1))

        gx = g(x[0])
        gy = g(x[1])
        gpx = gp(x[0])
        gpy = gp(x[1])

        u_x = gx * (gpy * stream_sum + gy * stream_dy)
        u_y = -(gpx * gy * stream_sum + gx * gy * stream_dx)
        return np.vstack([u_x, u_y]).astype(default_real_type)

    return velocity


def build_regular_grid(resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    eval_coords = coords.copy()
    if resolution > 1:
        eps = 1e-10
        eval_coords[0] = eps
        eval_coords[-1] = 1.0 - eps
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")
    eval_x, eval_y = np.meshgrid(eval_coords, eval_coords, indexing="ij")
    points = np.column_stack([eval_x.ravel(), eval_y.ravel(), np.zeros(resolution * resolution, dtype=np.float64)])
    return coords, points, np.stack([grid_x, grid_y], axis=-1)


def build_mask(theta: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    x = grid_xy[..., 0]
    y = grid_xy[..., 1]
    mask = np.zeros(x.shape, dtype=bool)
    for cx, cy, radius in theta:
        mask |= (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    return mask


def build_forcing_scalar_grid(resolution: int, forcing_scale: float) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")
    phase = 2.0 * np.pi * (grid_x + grid_y)
    return forcing_scale * (np.sin(phase) + np.cos(phase))


def build_body_force_ufl(x_coord, forcing_scale: float):
    phase = 2.0 * np.pi * (x_coord[0] + x_coord[1])
    force_x = forcing_scale * (ufl.sin(phase) + ufl.cos(phase))
    return ufl.as_vector((force_x, ufl.as_ufl(0.0)))


def locate_valid_points(msh, points: np.ndarray, mask: np.ndarray):
    flat_mask = mask.reshape(-1)
    fluid_indices = np.flatnonzero(~flat_mask)
    fluid_points = points[fluid_indices]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, fluid_points)
    collisions = geometry.compute_colliding_cells(msh, candidates, fluid_points)

    selected_points = []
    selected_cells = []
    selected_indices = []
    for local_idx, flat_idx in enumerate(fluid_indices):
        links = collisions.links(local_idx)
        if len(links) == 0:
            continue
        selected_points.append(fluid_points[local_idx])
        selected_cells.append(links[0])
        selected_indices.append(flat_idx)

    return (
        np.asarray(selected_indices, dtype=np.int64),
        np.asarray(selected_points, dtype=np.float64),
        np.asarray(selected_cells, dtype=np.int32),
    )


def evaluate_function_on_grid(func, resolution: int, selected_indices, selected_points, selected_cells, value_size: int) -> np.ndarray:
    out = np.zeros((resolution * resolution, value_size), dtype=np.float64)
    if len(selected_indices) > 0:
        values = func.eval(selected_points, selected_cells)
        values = np.asarray(values, dtype=np.float64).reshape(len(selected_indices), value_size)
        out[selected_indices] = values
    out = out.reshape(resolution, resolution, value_size)
    return out[..., 0] if value_size == 1 else out


def compute_vorticity_from_velocity(velocity_grid: np.ndarray, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    u_x = velocity_grid[..., 0]
    u_y = velocity_grid[..., 1]
    du_y_dx = np.gradient(u_y, coords, axis=0, edge_order=2)
    du_x_dy = np.gradient(u_x, coords, axis=1, edge_order=2)
    vort = du_y_dx - du_x_dy
    vort = np.asarray(vort, dtype=np.float64)
    vort[mask] = 0.0
    return vort


def build_problem(msh, facet_tags, dt: float, nu: float, gamma_drive: float, forcing_scale: float):
    P2 = element("Lagrange", msh.basix_cell(), degree=2, shape=(msh.geometry.dim,), dtype=default_real_type)
    P1 = element("Lagrange", msh.basix_cell(), degree=1, dtype=default_real_type)
    W = functionspace(msh, mixed_element([P2, P1]))

    W0 = W.sub(0)
    V, _ = W0.collapse()
    W1 = W.sub(1)
    Q, _ = W1.collapse()

    u_zero = Function(V)
    u_zero.x.array[:] = 0.0

    facets_noslip = facet_tags.find(TAG_NOSLIP)
    dofs_u = locate_dofs_topological((W0, V), msh.topology.dim - 1, facets_noslip)
    bc_u = dirichletbc(u_zero, dofs_u, W0)

    def pressure_pin_marker(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    pressure_pin_dofs = locate_dofs_geometrical((W1, Q), pressure_pin_marker)
    if len(pressure_pin_dofs[0]) == 0:
        raise RuntimeError("Failed to locate a pressure pin dof.")
    bc_p = dirichletbc(np.array(0.0, dtype=default_real_type), pressure_pin_dofs[0], W1)
    bcs = [bc_u, bc_p]

    up = Function(W)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    u_prev = Function(V)

    x_coord = ufl.SpatialCoordinate(msh)
    n = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    p0 = gamma_drive * (1.0 - x_coord[0])
    body_force = build_body_force_ufl(x_coord, forcing_scale)

    a = (
        (1.0 / dt) * ufl.inner(u, v) * ufl.dx
        + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u_prev, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )
    L = (
        (1.0 / dt) * ufl.inner(u_prev, v) * ufl.dx
        + ufl.inner(body_force, v) * ufl.dx
        + ufl.inner(p0 * n, v) * (ds(TAG_LEFT) + ds(TAG_RIGHT))
    )

    problem = LinearProblem(
        a,
        L,
        u=up,
        bcs=bcs,
        petsc_options_prefix="bias_aware_ns_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    u_eval = Function(V)
    p_eval = Function(Q)
    return {
        "W": W,
        "V": V,
        "Q": Q,
        "up": up,
        "u_prev": u_prev,
        "problem": problem,
        "u_eval": u_eval,
        "p_eval": p_eval,
    }


def solve_one_sample(theta: np.ndarray, args: argparse.Namespace, rng: np.random.Generator):
    msh, facet_tags = build_mesh(theta, float(args.mesh_size_min), float(args.mesh_size_max))
    state = build_problem(
        msh,
        facet_tags,
        float(args.dt),
        float(args.nu),
        float(args.gamma_drive),
        float(args.forcing_scale),
    )

    coeffs = sample_stream_coefficients(rng, int(args.stream_modes), float(args.stream_scale))
    state["u_prev"].interpolate(build_initial_velocity_callable(coeffs))
    state["u_prev"].x.scatter_forward()

    coords, points, grid_xy = build_regular_grid(int(args.grid_resolution))
    mask = build_mask(theta, grid_xy)
    forcing_grid = build_forcing_scalar_grid(int(args.grid_resolution), float(args.forcing_scale))
    selected_indices, selected_points, selected_cells = locate_valid_points(msh, points, mask)

    initial_velocity_grid = evaluate_function_on_grid(
        state["u_prev"], int(args.grid_resolution), selected_indices, selected_points, selected_cells, value_size=2
    )
    initial_vorticity = compute_vorticity_from_velocity(initial_velocity_grid, coords, mask)

    n_steps = int(math.ceil(float(args.final_time) / float(args.dt)))
    if n_steps < int(args.record_steps):
        raise ValueError("record_steps cannot exceed the number of implicit time steps.")
    record_targets = np.rint(np.linspace(1, n_steps, int(args.record_steps))).astype(np.int64)
    if len(np.unique(record_targets)) != int(args.record_steps):
        raise ValueError("record_steps is too large for the chosen final_time and dt.")

    vorticity = np.zeros((int(args.grid_resolution), int(args.grid_resolution), int(args.record_steps)), dtype=np.float32)
    velocity_u = np.zeros_like(vorticity) if args.save_aux_fields else None
    velocity_v = np.zeros_like(vorticity) if args.save_aux_fields else None
    pressure = np.zeros_like(vorticity) if args.save_aux_fields else None
    times = np.zeros(int(args.record_steps), dtype=np.float32)

    record_idx = 0
    time = 0.0
    for step_idx in range(n_steps):
        solution = state["problem"].solve()
        solution.x.scatter_forward()
        u_sol, p_sol = solution.split()
        u_sol.x.scatter_forward()
        p_sol.x.scatter_forward()

        state["u_eval"].interpolate(u_sol)
        state["p_eval"].interpolate(p_sol)
        state["u_eval"].x.scatter_forward()
        state["p_eval"].x.scatter_forward()

        if step_idx + 1 == record_targets[record_idx]:
            velocity_grid = evaluate_function_on_grid(
                state["u_eval"], int(args.grid_resolution), selected_indices, selected_points, selected_cells, value_size=2
            )
            pressure_grid = evaluate_function_on_grid(
                state["p_eval"], int(args.grid_resolution), selected_indices, selected_points, selected_cells, value_size=1
            )
            vorticity[..., record_idx] = compute_vorticity_from_velocity(velocity_grid, coords, mask).astype(np.float32)
            if args.save_aux_fields:
                velocity_u[..., record_idx] = velocity_grid[..., 0].astype(np.float32)
                velocity_v[..., record_idx] = velocity_grid[..., 1].astype(np.float32)
                pressure[..., record_idx] = pressure_grid.astype(np.float32)
            time = min((step_idx + 1) * float(args.dt), float(args.final_time))
            times[record_idx] = np.float32(time)
            record_idx += 1
            if record_idx == int(args.record_steps):
                break

        state["u_prev"].interpolate(u_sol)
        state["u_prev"].x.scatter_forward()

    return {
        "a": initial_vorticity.astype(np.float32),
        "u": vorticity.astype(np.float32),
        "mask": mask.astype(np.float32),
        "theta": theta.astype(np.float32),
        "t": times.astype(np.float32),
        "f": forcing_grid.astype(np.float32),
        "velocity_u": velocity_u,
        "velocity_v": velocity_v,
        "pressure": pressure,
    }


def save_split(out_path: Path, split: str, payload: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    mdict = {
        "a": payload["a"].astype(np.float32),
        "u": payload["u"].astype(np.float32),
        "mask": payload["mask"].astype(np.float32),
        "theta": payload["theta"].astype(np.float32),
        "t": payload["t"].astype(np.float32),
        "f": payload["f"].astype(np.float32),
        "nu": np.asarray([args.nu], dtype=np.float32),
        "eta": np.asarray([args.eta], dtype=np.float32),
        "final_time": np.asarray([args.final_time], dtype=np.float32),
        "dt": np.asarray([args.dt], dtype=np.float32),
        "gamma_drive": np.asarray([args.gamma_drive], dtype=np.float32),
        "forcing_scale": np.asarray([args.forcing_scale], dtype=np.float32),
        "split": np.asarray([split], dtype=object),
    }
    if args.save_aux_fields:
        mdict["velocity_u"] = payload["velocity_u"].astype(np.float32)
        mdict["velocity_v"] = payload["velocity_v"].astype(np.float32)
        mdict["pressure"] = payload["pressure"].astype(np.float32)
    file_format = resolve_storage_format(mdict, args.file_format)
    if file_format == "mat":
        scipy.io.savemat(out_path, mdict)
    else:
        save_hdf5_mat(out_path, mdict, split)
    print(f"Saved {split} dataset using {file_format} format -> {out_path}")


def resolve_storage_format(mdict: dict[str, np.ndarray], requested: str) -> str:
    if requested in {"mat", "hdf5"}:
        return requested

    max_array_bytes = 0
    total_bytes = 0
    for value in mdict.values():
        if isinstance(value, np.ndarray):
            max_array_bytes = max(max_array_bytes, int(value.nbytes))
            total_bytes += int(value.nbytes)

    if max_array_bytes >= MAT_V5_BYTE_LIMIT or total_bytes >= MAT_V5_BYTE_LIMIT:
        return "hdf5"
    return "mat"


def save_hdf5_mat(out_path: Path, mdict: dict[str, np.ndarray], split: str) -> None:
    with h5py.File(out_path, "w") as handle:
        handle.attrs["bias_aware_layout"] = "numpy"
        handle.attrs["split"] = split
        for key, value in mdict.items():
            if isinstance(value, np.ndarray) and value.dtype == object:
                encoded = np.asarray([str(item) for item in value.reshape(-1)], dtype="S")
                handle.create_dataset(key, data=encoded)
            else:
                handle.create_dataset(key, data=value)


def generate_split(
    split: str,
    n_samples: int,
    radius_range: tuple[float, float],
    hole_gap_min: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Path:
    theta_all = sample_theta_batch(
        n_samples=n_samples,
        rng=rng,
        n_holes=int(args.n_holes),
        radius_range=radius_range,
        boundary_margin=float(args.boundary_margin),
        hole_gap_min=hole_gap_min,
        max_attempts=int(args.max_placement_attempts),
    )

    a_all = np.zeros((n_samples, int(args.grid_resolution), int(args.grid_resolution)), dtype=np.float32)
    u_all = np.zeros((n_samples, int(args.grid_resolution), int(args.grid_resolution), int(args.record_steps)), dtype=np.float32)
    mask_all = np.zeros((n_samples, int(args.grid_resolution), int(args.grid_resolution)), dtype=np.float32)
    t_all = None

    velocity_u_all = np.zeros_like(u_all) if args.save_aux_fields else None
    velocity_v_all = np.zeros_like(u_all) if args.save_aux_fields else None
    pressure_all = np.zeros_like(u_all) if args.save_aux_fields else None

    for idx in range(n_samples):
        sample_payload = solve_one_sample(theta_all[idx], args, rng)
        a_all[idx] = sample_payload["a"]
        u_all[idx] = sample_payload["u"]
        mask_all[idx] = sample_payload["mask"]
        if t_all is None:
            t_all = sample_payload["t"]
        if args.save_aux_fields:
            velocity_u_all[idx] = sample_payload["velocity_u"]
            velocity_v_all[idx] = sample_payload["velocity_v"]
            pressure_all[idx] = sample_payload["pressure"]
        print(f"[{split}] solved {idx + 1}/{n_samples}")

    out_path = args.output_dir / default_file_name(
        split=split,
        n_samples=n_samples,
        nu=float(args.nu),
        eta=float(args.eta),
        record_steps=int(args.record_steps),
        resolution=int(args.grid_resolution),
    )
    save_split(
        out_path,
        split,
        {
            "a": a_all,
            "u": u_all,
            "mask": mask_all,
            "theta": theta_all,
            "t": t_all,
            "f": sample_payload["f"],
            "velocity_u": velocity_u_all,
            "velocity_v": velocity_v_all,
            "pressure": pressure_all,
        },
        args,
    )
    return out_path


def main() -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This generator currently supports serial execution only.")

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    for split, n_samples, radius_range, hole_gap_min in build_split_spec(args):
        if n_samples <= 0:
            continue
        generate_split(split, n_samples, radius_range, hole_gap_min, args, rng)


if __name__ == "__main__":
    main()
