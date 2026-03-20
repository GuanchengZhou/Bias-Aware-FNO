#!/usr/bin/env python3
"""Generate raw coarse/fine Stokes solutions for notebook postprocessing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random-parameter Stokes raw data on coarse/fine meshes. "
            "The companion notebook handles grid projection, error computation, and visualization."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "jupyter-notebook" / "stokes-random-mesh-error-256",
        help="Directory for raw data artifacts.",
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of random parameter samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parameter sampling.")
    parser.add_argument("--geometry-seed", type=int, default=42, help="Mesh seed for hole placement.")
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--gamma-min", type=float, default=0.0)
    parser.add_argument("--gamma-max", type=float, default=5.0)
    parser.add_argument("--coarse-mesh-size-min", type=float, default=0.03)
    parser.add_argument("--coarse-mesh-size-max", type=float, default=0.05)
    parser.add_argument("--fine-mesh-size-min", type=float, default=0.01)
    parser.add_argument("--fine-mesh-size-max", type=float, default=0.03)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the raw data files already exist.",
    )
    return parser.parse_args()


def repo_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    lrtor_root = repo_root.parent / "LRTOR"
    return repo_root, lrtor_root


def setup_runtime(lrtor_root: Path, seed: int) -> None:
    if str(lrtor_root) not in sys.path:
        sys.path.insert(0, str(lrtor_root))

    from lrtor.runtime_env import sanitize_current_process_build_env
    from lrtor.utils.misc import seed_everything

    sanitize_current_process_build_env(verbose=False)
    seed_everything(int(seed))


def build_required_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "sample_params": output_dir / "sample_params.npy",
        "coarse_mixed": output_dir / "coarse_mixed_coeff.npy",
        "fine_mixed": output_dir / "fine_mixed_coeff.npy",
        "coarse_mesh_info": output_dir / "coarse_mesh_info.npz",
        "fine_mesh_info": output_dir / "fine_mesh_info.npz",
        "coarse_solver_stats": output_dir / "coarse_solver_stats.npz",
        "fine_solver_stats": output_dir / "fine_solver_stats.npz",
        "generation_summary": output_dir / "generation_summary.json",
    }


def outputs_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def sample_parameters(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(int(args.seed))
    params = np.column_stack(
        [
            rng.uniform(float(args.alpha_min), float(args.alpha_max), size=int(args.n_samples)),
            rng.uniform(float(args.beta_min), float(args.beta_max), size=int(args.n_samples)),
            rng.uniform(float(args.gamma_min), float(args.gamma_max), size=int(args.n_samples)),
        ]
    )
    return np.asarray(params, dtype=np.float64)


def build_generator(mesh_size_min: float, mesh_size_max: float, seed: int):
    from lrtor.data.stokes_5holes import Stokes5HolesGenerator

    generator = Stokes5HolesGenerator(
        mesh_size_min=float(mesh_size_min),
        mesh_size_max=float(mesh_size_max),
        seed=int(seed),
    )
    generator.build_mesh()
    generator.setup_problem()
    return generator


def extract_mesh_info(generator) -> dict[str, np.ndarray | float | int]:
    msh = generator._msh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    num_cells = msh.topology.index_map(tdim).size_local

    cell_vertices = np.asarray([c2v.links(i) for i in range(num_cells)], dtype=np.int32)
    vertex_coords = np.asarray(msh.geometry.x[:, :2], dtype=np.float64)
    u_coords = np.asarray(generator._Vx.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    p_coords = np.asarray(generator._Q.tabulate_dof_coordinates()[:, :2], dtype=np.float64)

    return {
        "vertex_coords": vertex_coords,
        "cell_vertices": cell_vertices,
        "u_coords": u_coords,
        "p_coords": p_coords,
        "u_dof_count": int(u_coords.shape[0]),
        "p_dof_count": int(p_coords.shape[0]),
        "cell_count": int(num_cells),
        "vertex_count": int(vertex_coords.shape[0]),
        "hole_centers": np.asarray(generator._centers, dtype=np.float64),
        "hole_radius": float(generator._hole_radius),
        "ux_bc_dofs": np.asarray(generator._ux_bc_dofs, dtype=np.int64),
        "uy_bc_dofs": np.asarray(generator._uy_bc_dofs, dtype=np.int64),
        "p_pin_dofs": np.asarray(generator._p_pin_dofs, dtype=np.int64),
    }


def set_constant(constant, value: float) -> None:
    from petsc4py import PETSc

    scalar = PETSc.ScalarType(float(value))
    try:
        constant.value = scalar
    except Exception:
        constant.value[:] = scalar


def build_solver_state(generator, prefix: str) -> dict[str, object]:
    from dolfinx.fem import Function
    from dolfinx.fem.petsc import NonlinearProblem

    problem = NonlinearProblem(
        generator._F,
        generator._Up,
        bcs=generator._bcs,
        J=generator._J,
        petsc_options_prefix=prefix,
    )
    snes = problem.solver
    snes.setTolerances(rtol=1e-8, atol=1e-10, max_it=50)
    try:
        snes.getLineSearch().setType("bt")
    except Exception:
        pass
    ksp = snes.getKSP()
    ksp.setType("gmres")
    pc = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass

    return {
        "problem": problem,
        "snes": snes,
        "ux_plot": Function(generator._Vx),
        "uy_plot": Function(generator._Vy),
        "p_plot": Function(generator._Q),
    }


def capture_mixed_coefficients(generator, solver_state: dict[str, object]) -> np.ndarray:
    uh, ph = generator._Up.split()
    uh.x.scatter_forward()
    ph.x.scatter_forward()
    solver_state["ux_plot"].interpolate(uh.sub(0))
    solver_state["uy_plot"].interpolate(uh.sub(1))
    solver_state["p_plot"].interpolate(ph)
    solver_state["ux_plot"].x.scatter_forward()
    solver_state["uy_plot"].x.scatter_forward()
    solver_state["p_plot"].x.scatter_forward()
    ux_coeff = np.asarray(solver_state["ux_plot"].x.array, dtype=np.float64)
    uy_coeff = np.asarray(solver_state["uy_plot"].x.array, dtype=np.float64)
    p_coeff = np.asarray(solver_state["p_plot"].x.array, dtype=np.float64)
    return np.concatenate([ux_coeff, uy_coeff, p_coeff], axis=0)


def solve_one_parameter(
    generator,
    solver_state: dict[str, object],
    alpha: float,
    beta: float,
    gamma: float,
    zero_initial_guess: bool = False,
) -> tuple[np.ndarray, int]:
    def _run(reset_state: bool) -> tuple[np.ndarray, int]:
        if reset_state:
            generator._Up.x.array[:] = 0.0
            generator._Up.x.scatter_forward()
        set_constant(generator._alpha_c, alpha)
        set_constant(generator._beta_c, beta)
        set_constant(generator._gamma_c, gamma)
        solver_state["problem"].solve()
        generator._Up.x.scatter_forward()
        reason = int(solver_state["snes"].getConvergedReason())
        if reason <= 0:
            raise RuntimeError(f"SNES did not converge, reason={reason}")
        coeff = capture_mixed_coefficients(generator, solver_state)
        return coeff, reason

    try:
        return _run(reset_state=zero_initial_guess)
    except Exception:
        if zero_initial_guess:
            raise
        return _run(reset_state=True)


def solve_parameter_batch(generator, params: np.ndarray, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solver_state = build_solver_state(generator, prefix=prefix)
    n_u = int(generator._Vx.dofmap.index_map.size_global)
    n_p = int(generator._Q.dofmap.index_map.size_global)
    mixed = np.empty((len(params), 2 * n_u + n_p), dtype=np.float64)
    reasons = np.empty(len(params), dtype=np.int32)
    durations = np.empty(len(params), dtype=np.float64)

    for idx, (alpha, beta, gamma) in enumerate(np.asarray(params, dtype=np.float64)):
        start = time.perf_counter()
        mixed[idx], reasons[idx] = solve_one_parameter(
            generator,
            solver_state,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            zero_initial_guess=(idx == 0),
        )
        durations[idx] = time.perf_counter() - start
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == len(params):
            print(f"[{prefix}] solved {idx + 1}/{len(params)}")

    return mixed, reasons, durations


def ensure_same_geometry(coarse_mesh_info: dict[str, object], fine_mesh_info: dict[str, object]) -> None:
    if not np.allclose(coarse_mesh_info["hole_centers"], fine_mesh_info["hole_centers"]):
        raise RuntimeError("coarse and fine meshes do not share the same hole centers.")
    if not np.isclose(float(coarse_mesh_info["hole_radius"]), float(fine_mesh_info["hole_radius"])):
        raise RuntimeError("coarse and fine meshes do not share the same hole radius.")


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    coarse_mesh_info: dict[str, object],
    fine_mesh_info: dict[str, object],
    coarse_reasons: np.ndarray,
    fine_reasons: np.ndarray,
    coarse_durations: np.ndarray,
    fine_durations: np.ndarray,
) -> dict[str, object]:
    return {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "geometry_seed": int(args.geometry_seed),
        "param_ranges": {
            "alpha": [float(args.alpha_min), float(args.alpha_max)],
            "beta": [float(args.beta_min), float(args.beta_max)],
            "gamma": [float(args.gamma_min), float(args.gamma_max)],
        },
        "coarse_mesh": {
            "mesh_size_min": float(args.coarse_mesh_size_min),
            "mesh_size_max": float(args.coarse_mesh_size_max),
            "cell_count": int(coarse_mesh_info["cell_count"]),
            "vertex_count": int(coarse_mesh_info["vertex_count"]),
            "u_dof_count": int(coarse_mesh_info["u_dof_count"]),
            "p_dof_count": int(coarse_mesh_info["p_dof_count"]),
            "mean_solve_seconds": float(np.mean(coarse_durations)),
            "max_solve_seconds": float(np.max(coarse_durations)),
            "min_converged_reason": int(np.min(coarse_reasons)),
        },
        "fine_mesh": {
            "mesh_size_min": float(args.fine_mesh_size_min),
            "mesh_size_max": float(args.fine_mesh_size_max),
            "cell_count": int(fine_mesh_info["cell_count"]),
            "vertex_count": int(fine_mesh_info["vertex_count"]),
            "u_dof_count": int(fine_mesh_info["u_dof_count"]),
            "p_dof_count": int(fine_mesh_info["p_dof_count"]),
            "mean_solve_seconds": float(np.mean(fine_durations)),
            "max_solve_seconds": float(np.max(fine_durations)),
            "min_converged_reason": int(np.min(fine_reasons)),
        },
        "generated_files": [
            "sample_params.npy",
            "coarse_mixed_coeff.npy",
            "fine_mixed_coeff.npy",
            "coarse_mesh_info.npz",
            "fine_mesh_info.npz",
            "coarse_solver_stats.npz",
            "fine_solver_stats.npz",
            "generation_summary.json",
        ],
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root, lrtor_root = repo_paths()
    del repo_root  # resolved for consistency; not otherwise needed here

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = build_required_paths(output_dir)

    if outputs_complete(paths) and not args.force:
        print(f"Raw data already exist in {output_dir}. Use --force to regenerate.")
        print(paths["generation_summary"].read_text(encoding="utf-8"))
        return

    setup_runtime(lrtor_root=lrtor_root, seed=int(args.seed))

    sample_params = sample_parameters(args)
    print("Building coarse generator...")
    coarse_generator = build_generator(
        mesh_size_min=float(args.coarse_mesh_size_min),
        mesh_size_max=float(args.coarse_mesh_size_max),
        seed=int(args.geometry_seed),
    )
    print("Building fine generator...")
    fine_generator = build_generator(
        mesh_size_min=float(args.fine_mesh_size_min),
        mesh_size_max=float(args.fine_mesh_size_max),
        seed=int(args.geometry_seed),
    )

    coarse_mesh_info = extract_mesh_info(coarse_generator)
    fine_mesh_info = extract_mesh_info(fine_generator)
    ensure_same_geometry(coarse_mesh_info, fine_mesh_info)

    print("Solving coarse batch...")
    coarse_mixed, coarse_reasons, coarse_durations = solve_parameter_batch(
        coarse_generator,
        sample_params,
        prefix="stokes_coarse_script_",
    )
    print("Solving fine batch...")
    fine_mixed, fine_reasons, fine_durations = solve_parameter_batch(
        fine_generator,
        sample_params,
        prefix="stokes_fine_script_",
    )

    np.save(paths["sample_params"], sample_params)
    np.save(paths["coarse_mixed"], coarse_mixed)
    np.save(paths["fine_mixed"], fine_mixed)
    np.savez_compressed(paths["coarse_mesh_info"], **coarse_mesh_info)
    np.savez_compressed(paths["fine_mesh_info"], **fine_mesh_info)
    np.savez_compressed(
        paths["coarse_solver_stats"],
        converged_reasons=coarse_reasons,
        solve_seconds=coarse_durations,
    )
    np.savez_compressed(
        paths["fine_solver_stats"],
        converged_reasons=fine_reasons,
        solve_seconds=fine_durations,
    )

    summary = build_summary(
        args=args,
        output_dir=output_dir,
        coarse_mesh_info=coarse_mesh_info,
        fine_mesh_info=fine_mesh_info,
        coarse_reasons=coarse_reasons,
        fine_reasons=fine_reasons,
        coarse_durations=coarse_durations,
        fine_durations=fine_durations,
    )
    save_json(paths["generation_summary"], summary)

    print(f"Raw data saved to {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
