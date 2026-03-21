#!/usr/bin/env python3
"""Generate paired coarse/fine Darcy FEM datasets on a shared reference grid."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.io

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
FENICS_HELPER_DIR = REPO_ROOT / "data_generation" / "navier_stokes_5holes"
for path in (SCRIPT_DIR, FENICS_HELPER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fenicsx_runtime import sanitize_current_process_build_env
from grf import sample_threshold_coefficients

sanitize_current_process_build_env(verbose=True)

import ufl
from dolfinx import default_real_type, fem, geometry, mesh
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI


MAT_V5_BYTE_LIMIT = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class SplitShapes:
    coeff: tuple[int, int, int]
    sol_coarse: tuple[int, int, int]
    sol_fine: tuple[int, int, int]
    error_hf_lf: tuple[int, int, int]
    a_coarse: tuple[int, int]
    a_fine: tuple[int, int]


@dataclass
class MeshBundle:
    name: str
    n: int
    mesh: mesh.Mesh
    V: any
    DG0: any
    coeff_fn: Function
    problem: LinearProblem
    cell_centers: np.ndarray
    mesh_points: np.ndarray
    mesh_cells: np.ndarray
    selected_indices: np.ndarray
    selected_points: np.ndarray
    selected_cells: np.ndarray


def default_file_name(split: str, n_samples: int, coeff_resolution: int, coarse_n: int, fine_n: int) -> str:
    return f"darcy_fem_r{coeff_resolution}_C{coarse_n}_F{fine_n}_N{n_samples}_{split}.mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Darcy datasets with coarse/fine FEM solves on a shared grid.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--coeff-resolution", type=int, default=256)
    parser.add_argument("--coarse-n", type=int, default=256)
    parser.add_argument("--fine-n", type=int, default=512)
    parser.add_argument("--grf-alpha", type=float, default=2.0)
    parser.add_argument("--grf-tau", type=float, default=3.0)
    parser.add_argument("--coeff-low", type=float, default=4.0)
    parser.add_argument("--coeff-high", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--file-format",
        choices=["auto", "mat", "hdf5"],
        default="auto",
        help="Use MAT v5 when possible; auto falls back to HDF5-backed .mat for large datasets.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.coeff_resolution) < 2:
        raise ValueError("coeff_resolution must be at least 2.")
    if int(args.coarse_n) <= 0 or int(args.fine_n) <= 0:
        raise ValueError("coarse_n and fine_n must be positive.")
    if int(args.fine_n) <= int(args.coarse_n):
        raise ValueError("fine_n must be greater than coarse_n.")
    if float(args.coeff_low) >= float(args.coeff_high):
        raise ValueError("coeff_low must be smaller than coeff_high.")


def build_shapes(n_samples: int, coeff_resolution: int, coarse_cells: int, fine_cells: int) -> SplitShapes:
    return SplitShapes(
        coeff=(n_samples, coeff_resolution, coeff_resolution),
        sol_coarse=(n_samples, coeff_resolution, coeff_resolution),
        sol_fine=(n_samples, coeff_resolution, coeff_resolution),
        error_hf_lf=(n_samples, coeff_resolution, coeff_resolution),
        a_coarse=(n_samples, coarse_cells),
        a_fine=(n_samples, fine_cells),
    )


def estimated_total_bytes(shapes: SplitShapes) -> int:
    total = 0
    for shape in (
        shapes.coeff,
        shapes.sol_coarse,
        shapes.sol_fine,
        shapes.error_hf_lf,
        shapes.a_coarse,
        shapes.a_fine,
    ):
        total += int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    return total


def resolve_storage_format(requested: str, shapes: SplitShapes) -> str:
    if requested == "mat":
        return "mat"
    if requested == "hdf5":
        return "hdf5"

    max_array_bytes = max(
        int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        for shape in (
            shapes.coeff,
            shapes.sol_coarse,
            shapes.sol_fine,
            shapes.error_hf_lf,
            shapes.a_coarse,
            shapes.a_fine,
        )
    )
    if max_array_bytes >= MAT_V5_BYTE_LIMIT or estimated_total_bytes(shapes) >= MAT_V5_BYTE_LIMIT:
        return "hdf5"
    return "mat"


def save_hdf5_metadata(handle: h5py.File, split: str, args: argparse.Namespace, coarse_bundle: MeshBundle, fine_bundle: MeshBundle) -> None:
    handle.attrs["bias_aware_layout"] = "numpy"
    handle.attrs["split"] = split
    handle.create_dataset("mesh_coarse_points", data=coarse_bundle.mesh_points.astype(np.float32))
    handle.create_dataset("mesh_coarse_cells", data=coarse_bundle.mesh_cells.astype(np.int32))
    handle.create_dataset("mesh_fine_points", data=fine_bundle.mesh_points.astype(np.float32))
    handle.create_dataset("mesh_fine_cells", data=fine_bundle.mesh_cells.astype(np.int32))
    handle.create_dataset("grid_coords", data=np.linspace(0.0, 1.0, int(args.coeff_resolution), dtype=np.float32))
    handle.create_dataset("grf_alpha", data=np.asarray([args.grf_alpha], dtype=np.float32))
    handle.create_dataset("grf_tau", data=np.asarray([args.grf_tau], dtype=np.float32))
    handle.create_dataset("coeff_low", data=np.asarray([args.coeff_low], dtype=np.float32))
    handle.create_dataset("coeff_high", data=np.asarray([args.coeff_high], dtype=np.float32))
    handle.create_dataset("coeff_resolution", data=np.asarray([args.coeff_resolution], dtype=np.int32))
    handle.create_dataset("coarse_n", data=np.asarray([args.coarse_n], dtype=np.int32))
    handle.create_dataset("fine_n", data=np.asarray([args.fine_n], dtype=np.int32))


def reference_grid(coeff_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(0.0, 1.0, coeff_resolution, dtype=np.float64)
    eval_coords = coords.copy()
    if coeff_resolution > 1:
        eps = min(1e-10, 0.25 / (coeff_resolution - 1))
        eval_coords[0] = eps
        eval_coords[-1] = 1.0 - eps
    eval_x, eval_y = np.meshgrid(eval_coords, eval_coords, indexing="ij")
    points = np.column_stack([eval_x.ravel(), eval_y.ravel(), np.zeros(coeff_resolution * coeff_resolution, dtype=np.float64)])
    return coords, points


def lookup_coefficients(coeff: np.ndarray, coords: np.ndarray) -> np.ndarray:
    resolution = int(coeff.shape[0])
    indices_x = np.clip((coords[:, 0] * resolution).astype(np.int64), 0, resolution - 1)
    indices_y = np.clip((coords[:, 1] * resolution).astype(np.int64), 0, resolution - 1)
    return coeff[indices_x, indices_y].astype(np.float64)


def extract_mesh_arrays(msh) -> tuple[np.ndarray, np.ndarray]:
    cell_dim = msh.topology.dim
    msh.topology.create_connectivity(cell_dim, 0)
    connectivity = msh.topology.connectivity(cell_dim, 0)
    num_cells = msh.topology.index_map(cell_dim).size_local
    cells = np.vstack([connectivity.links(cell) for cell in range(num_cells)]).astype(np.int32)
    points = np.asarray(msh.geometry.x[:, :2], dtype=np.float64)
    return points, cells


def locate_points_on_mesh(msh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(msh, candidates, points)

    selected_points = []
    selected_cells = []
    selected_indices = []
    for idx in range(len(points)):
        links = collisions.links(idx)
        if len(links) == 0:
            continue
        selected_points.append(points[idx])
        selected_cells.append(int(links[0]))
        selected_indices.append(idx)

    return (
        np.asarray(selected_indices, dtype=np.int64),
        np.asarray(selected_points, dtype=np.float64),
        np.asarray(selected_cells, dtype=np.int32),
    )


def evaluate_function_on_grid(
    func: Function,
    coeff_resolution: int,
    selected_indices: np.ndarray,
    selected_points: np.ndarray,
    selected_cells: np.ndarray,
) -> np.ndarray:
    values = np.zeros(coeff_resolution * coeff_resolution, dtype=np.float64)
    if len(selected_indices) > 0:
        eval_values = np.asarray(func.eval(selected_points, selected_cells), dtype=np.float64).reshape(-1)
        values[selected_indices] = eval_values
    values = values.reshape(coeff_resolution, coeff_resolution)
    values[0, :] = 0.0
    values[-1, :] = 0.0
    values[:, 0] = 0.0
    values[:, -1] = 0.0
    return values


def build_bundle(name: str, n: int, ref_points: np.ndarray) -> MeshBundle:
    msh = mesh.create_unit_square(MPI.COMM_SELF, n, n)
    V = functionspace(msh, ("Lagrange", 1))
    DG0 = functionspace(msh, ("DG", 0))
    coeff_fn = Function(DG0)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    rhs = fem.Constant(msh, default_real_type(1.0))

    def on_boundary(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        )

    boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
    bc = fem.dirichletbc(np.array(0.0, dtype=default_real_type), boundary_dofs, V)

    a = ufl.inner(coeff_fn * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = rhs * v * ufl.dx
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"bias_aware_darcy_{name}_",
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": 1e-10,
            "pc_type": "gamg",
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_pc_type": "jacobi",
        },
    )

    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_centers = mesh.compute_midpoints(
        msh,
        msh.topology.dim,
        np.arange(num_cells, dtype=np.int32),
    )[:, :2]
    mesh_points, mesh_cells = extract_mesh_arrays(msh)
    selected_indices, selected_points, selected_cells = locate_points_on_mesh(msh, ref_points)
    if len(selected_indices) != len(ref_points):
        raise RuntimeError(f"Failed to locate all reference points on the {name} mesh.")

    return MeshBundle(
        name=name,
        n=n,
        mesh=msh,
        V=V,
        DG0=DG0,
        coeff_fn=coeff_fn,
        problem=problem,
        cell_centers=cell_centers,
        mesh_points=mesh_points,
        mesh_cells=mesh_cells,
        selected_indices=selected_indices,
        selected_points=selected_points,
        selected_cells=selected_cells,
    )


def solve_on_bundle(bundle: MeshBundle, coeff_grid: np.ndarray, coeff_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    coeff_cells = lookup_coefficients(coeff_grid, bundle.cell_centers)
    bundle.coeff_fn.x.array[:] = coeff_cells
    bundle.coeff_fn.x.scatter_forward()

    solution = bundle.problem.solve()
    solution.x.scatter_forward()
    solution_grid = evaluate_function_on_grid(
        solution,
        coeff_resolution,
        bundle.selected_indices,
        bundle.selected_points,
        bundle.selected_cells,
    )
    return solution_grid.astype(np.float32), coeff_cells.astype(np.float32)


def run_sample(
    rng: np.random.Generator,
    coeff_resolution: int,
    grf_alpha: float,
    grf_tau: float,
    coeff_low: float,
    coeff_high: float,
    coarse_bundle: MeshBundle,
    fine_bundle: MeshBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coeff_grid, _ = sample_threshold_coefficients(
        rng=rng,
        resolution=coeff_resolution,
        alpha=grf_alpha,
        tau=grf_tau,
        coeff_low=coeff_low,
        coeff_high=coeff_high,
    )
    coarse_grid, coarse_cells = solve_on_bundle(coarse_bundle, coeff_grid, coeff_resolution)
    fine_grid, fine_cells = solve_on_bundle(fine_bundle, coeff_grid, coeff_resolution)
    error_grid = fine_grid - coarse_grid
    return (
        coeff_grid.astype(np.float32),
        coarse_grid.astype(np.float32),
        fine_grid.astype(np.float32),
        error_grid.astype(np.float32),
        np.asarray(coarse_cells, dtype=np.float32),
        np.asarray(fine_cells, dtype=np.float32),
    )


def generate_split_hdf5(
    out_path: Path,
    split: str,
    n_samples: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    coarse_bundle: MeshBundle,
    fine_bundle: MeshBundle,
    shapes: SplitShapes,
) -> None:
    with h5py.File(out_path, "w") as handle:
        save_hdf5_metadata(handle, split, args, coarse_bundle, fine_bundle)
        handle.create_dataset("coeff", shape=shapes.coeff, dtype="f4")
        handle.create_dataset("sol_coarse", shape=shapes.sol_coarse, dtype="f4")
        handle.create_dataset("sol_fine", shape=shapes.sol_fine, dtype="f4")
        handle.create_dataset("error_hf_lf", shape=shapes.error_hf_lf, dtype="f4")
        handle.create_dataset("a_coarse", shape=shapes.a_coarse, dtype="f4")
        handle.create_dataset("a_fine", shape=shapes.a_fine, dtype="f4")

        for idx in range(n_samples):
            coeff_grid, coarse_grid, fine_grid, error_grid, coarse_cells, fine_cells = run_sample(
                rng=rng,
                coeff_resolution=int(args.coeff_resolution),
                grf_alpha=float(args.grf_alpha),
                grf_tau=float(args.grf_tau),
                coeff_low=float(args.coeff_low),
                coeff_high=float(args.coeff_high),
                coarse_bundle=coarse_bundle,
                fine_bundle=fine_bundle,
            )
            handle["coeff"][idx] = coeff_grid
            handle["sol_coarse"][idx] = coarse_grid
            handle["sol_fine"][idx] = fine_grid
            handle["error_hf_lf"][idx] = error_grid
            handle["a_coarse"][idx] = coarse_cells
            handle["a_fine"][idx] = fine_cells
            print(
                f"[{split}] {idx + 1:04d}/{n_samples:04d} "
                f"coarse_mean={float(np.mean(np.abs(coarse_grid))):.4e} "
                f"fine_mean={float(np.mean(np.abs(fine_grid))):.4e}"
            )


def generate_split_mat(
    out_path: Path,
    split: str,
    n_samples: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    coarse_bundle: MeshBundle,
    fine_bundle: MeshBundle,
    shapes: SplitShapes,
) -> None:
    coeff = np.zeros(shapes.coeff, dtype=np.float32)
    sol_coarse = np.zeros(shapes.sol_coarse, dtype=np.float32)
    sol_fine = np.zeros(shapes.sol_fine, dtype=np.float32)
    error_hf_lf = np.zeros(shapes.error_hf_lf, dtype=np.float32)
    a_coarse = np.zeros(shapes.a_coarse, dtype=np.float32)
    a_fine = np.zeros(shapes.a_fine, dtype=np.float32)

    for idx in range(n_samples):
        coeff_grid, coarse_grid, fine_grid, error_grid, coarse_cells, fine_cells = run_sample(
            rng=rng,
            coeff_resolution=int(args.coeff_resolution),
            grf_alpha=float(args.grf_alpha),
            grf_tau=float(args.grf_tau),
            coeff_low=float(args.coeff_low),
            coeff_high=float(args.coeff_high),
            coarse_bundle=coarse_bundle,
            fine_bundle=fine_bundle,
        )
        coeff[idx] = coeff_grid
        sol_coarse[idx] = coarse_grid
        sol_fine[idx] = fine_grid
        error_hf_lf[idx] = error_grid
        a_coarse[idx] = coarse_cells
        a_fine[idx] = fine_cells
        print(
            f"[{split}] {idx + 1:04d}/{n_samples:04d} "
            f"coarse_mean={float(np.mean(np.abs(coarse_grid))):.4e} "
            f"fine_mean={float(np.mean(np.abs(fine_grid))):.4e}"
        )

    scipy.io.savemat(
        out_path,
        {
            "coeff": coeff,
            "sol_coarse": sol_coarse,
            "sol_fine": sol_fine,
            "error_hf_lf": error_hf_lf,
            "a_coarse": a_coarse,
            "a_fine": a_fine,
            "mesh_coarse_points": coarse_bundle.mesh_points.astype(np.float32),
            "mesh_coarse_cells": coarse_bundle.mesh_cells.astype(np.int32),
            "mesh_fine_points": fine_bundle.mesh_points.astype(np.float32),
            "mesh_fine_cells": fine_bundle.mesh_cells.astype(np.int32),
            "grid_coords": np.linspace(0.0, 1.0, int(args.coeff_resolution), dtype=np.float32),
            "grf_alpha": np.asarray([args.grf_alpha], dtype=np.float32),
            "grf_tau": np.asarray([args.grf_tau], dtype=np.float32),
            "coeff_low": np.asarray([args.coeff_low], dtype=np.float32),
            "coeff_high": np.asarray([args.coeff_high], dtype=np.float32),
            "coeff_resolution": np.asarray([args.coeff_resolution], dtype=np.int32),
            "coarse_n": np.asarray([args.coarse_n], dtype=np.int32),
            "fine_n": np.asarray([args.fine_n], dtype=np.int32),
        },
    )


def generate_split(
    split: str,
    n_samples: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    coarse_bundle: MeshBundle,
    fine_bundle: MeshBundle,
) -> Path | None:
    if n_samples <= 0:
        return None

    shapes = build_shapes(
        n_samples=n_samples,
        coeff_resolution=int(args.coeff_resolution),
        coarse_cells=coarse_bundle.cell_centers.shape[0],
        fine_cells=fine_bundle.cell_centers.shape[0],
    )
    out_path = Path(args.output_dir) / default_file_name(
        split=split,
        n_samples=n_samples,
        coeff_resolution=int(args.coeff_resolution),
        coarse_n=int(args.coarse_n),
        fine_n=int(args.fine_n),
    )
    storage_format = resolve_storage_format(args.file_format, shapes)
    if storage_format == "hdf5":
        generate_split_hdf5(out_path, split, n_samples, args, rng, coarse_bundle, fine_bundle, shapes)
    else:
        generate_split_mat(out_path, split, n_samples, args, rng, coarse_bundle, fine_bundle, shapes)

    print(f"Saved {split} dataset using {storage_format} format -> {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _, ref_points = reference_grid(int(args.coeff_resolution))
    coarse_bundle = build_bundle("coarse", int(args.coarse_n), ref_points)
    fine_bundle = build_bundle("fine", int(args.fine_n), ref_points)
    rng = np.random.default_rng(int(args.seed))

    for split, n_samples in (("train", int(args.n_train)), ("test", int(args.n_test))):
        generate_split(split, n_samples, args, rng, coarse_bundle, fine_bundle)


if __name__ == "__main__":
    main()
