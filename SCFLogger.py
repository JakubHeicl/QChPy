from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
from time import perf_counter
from basis_set import BasisSet
import numpy as np
from utils import Z_TO_SYMBOL
from pathlib import Path

vec3 = np.ndarray
Array = np.ndarray

@dataclass
class SCFIter:
    it: int
    E_elec: float
    E_tot: float
    C: Array
    eps: Array
    dE: float
    p_rms: float

class SCFLogger:
    out_filename: str
    _meta: dict[str, Any]

    t0: float
    basis_set: BasisSet
    atoms: list[tuple[int, vec3]]

    tS: float
    tX: float
    tH: float
    tE: float
    tEnuc: float

    tscf0: float
    history: list[SCFIter]

    def __init__(self, xyz_filename: str, out_filename: str):
        
        self.xyz_filename = xyz_filename
        self.out_filename = out_filename

    def __enter__(self) -> SCFLogger:
        return self
    
    def __exit__(self, exc_type, exc, tb) -> None:
        self.end()

    def _open(self) -> None:
        Path(self.out_filename).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_filename, "w", encoding="utf-8")

    def _close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def section(self, title: str) -> None:
        self.log("\n" + title + "\n")
        self.log("-" * len(title) + "\n")

    def start(self, basis_type: str, basis_set: BasisSet, atoms: list[tuple[int, vec3]], n_elec: float, e_tol: float, p_rms_tol: float, maxiter: int):

        self._meta = dict(
            xyz_filename=self.xyz_filename,
            basis_type=basis_type,
            n_elec=n_elec,
            n_functions = len(basis_set.cgtos),
            n_atoms=len(atoms),
            maxiter=maxiter,
            e_tol=e_tol,
            p_rms_tol=p_rms_tol,
        )

        self.history = []

        self.t0 = perf_counter()
        self.basis_set = basis_set
        self.atoms = atoms

        self._open()

        self.log("HF PYTHON PROGRAM OUTPUT FILE\n")
        self.log("=" * 30 + "\n")

        self.log_meta()
        self.log_atoms()
        self.log_basis_set()

    def log(self, message: str) -> None:
        if self._fh is None:
            raise RuntimeError("SCFLogger: file is not open. Call start() first.")
        self._fh.write(message)
        if message.endswith("\n"):
            print(message[:-1])
        self._fh.flush()

    def log_meta(self) -> None:
        self.section("RUN INFO")
        m = self._meta
        self.log(
            f"input file: {m['xyz_filename']}\n"
            f"basis type: {m['basis_type']}\n"
            f"number of atoms: {m['n_atoms']}\n"
            f"number of electrons: {m['n_elec']}\n"
            f"number of basis functions: {m['n_functions']}\n"
            f"maximum number of iterations: {m['maxiter']}\n"
            f"energy tolerance |dE| < {m['e_tol']:.3e}\n"
            f"density tolerance p_rms < {m['p_rms_tol']:.3e}\n"
        )

    def log_atoms(self) -> None:
        assert self.atoms is not None
        self.section("ATOM COORDINATES (BOHR)")
        self.log(" Z  sym          x               y               z\n")
        self.log("--- ----  ---------------  ---------------  ---------------\n")
        for (Z, coor) in self.atoms:
            sym = Z_TO_SYMBOL.get(Z, f"Z{Z}")
            self.log(f"{Z:3d} {sym:>4s}  {coor[0]:15.8f}  {coor[1]:15.8f}  {coor[2]:15.8f}\n")

    def log_basis_set(self, max_print: Optional[int] = None) -> None:
        assert self.basis_set is not None
        self.section("CGTO BASIS SET")
        self.log("[      X            Y            Z     ]  (i, j, k)  (    coeff   ,     alpha   )\n")
        self.log("----------------------------------------  ---------  ----------------------------\n")

        for cgto in self.basis_set.cgtos:
            self.log(f"{cgto}")

    def integrals_done(self) -> None:
        self.section("INTEGRAL BUILD TIMINGS")
        self.log(f"S:    {self.tS:10.3f} s\n")
        self.log(f"X:    {self.tX:10.3f} s\n")
        self.log(f"H:    {self.tH:10.3f} s\n")
        self.log(f"ERI:  {self.tE:10.3f} s\n")
        self.log(f"Enuc: {self.tEnuc:10.3f} s\n")

    def scf_begin(self, eps, C) -> None:
        self.section("SCF ITERATIONS")
        self.log("Starting with a guess from hamiltonian:\n\n")
        self.log("    epsilon    |            coefficients             \n")
        self.log("---------------+-------------------------------------\n")
        for epsilon, c in zip(eps, C.T):
            line = " ".join(f"{number:12.8f}" for number in c)
            self.log(f"{epsilon:15.8f}|{line}\n")
        self.log("\n")
        self.log(" it |        E_tot (Eh) |          dE |     p_rms \n")
        self.log("----+-------------------+-------------+-----------\n")
        self.tscf0 = perf_counter()

    def scf_end(self, eps, C) -> None:
        self.log("\n")
        self.log("Computed molecular orbitals and energies:\n\n")
        self.log("    epsilon    |            coefficients             \n")
        self.log("---------------+-------------------------------------\n")
        for epsilon, c in zip(eps, C.T):
            line = " ".join(f"{number:12.8f}" for number in c)
            self.log(f"{epsilon:15.8f}|{line}\n")
        self.log("\n")

    def iteration(self, it: int, E_elec: float, E_tot: float, C: Array, eps: Array, dE: float, p_rms: float) -> None:
        rec = SCFIter(it=it, E_elec=E_elec, E_tot=E_tot, dE=dE, p_rms=p_rms, C=C, eps = eps)
        self.history.append(rec)

        dE_str = f"{dE:11.3e}"
        self.log(f"{it:3d} | {E_tot:17.10f} | {dE_str:>11s} | {p_rms:9.2e}\n")

    def converged(self, it: int, E_tot: float) -> None:
        self.log(f"SCF converged in {it} iterations\n")
        self.log(f"Final E_tot = {E_tot:.12f} Eh\n")

    def failed(self, maxiter: int, last_E_tot: float) -> None:
        self.log(f"WARNING: SCF did not converge in {maxiter} iterations\n")
        self.log(f"Last E_tot = {last_E_tot:.12f} Eh\n")

    def end(self) -> None:
        if self._fh is None:
            return
        self.section("RUN END")
        self.log(f"Total run time: {perf_counter() - self.t0:.2f} s\n")
        self._close()