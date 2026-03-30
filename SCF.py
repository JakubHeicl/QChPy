import numpy as np
from gaussian import *
from basis_set import *
from SCFLogger import SCFLogger
from tqdm import tqdm
from time import perf_counter

Array = np.ndarray

class SCF:
    atoms: list[tuple[int, vec3]]
    n_elec: int
    basis_set: BasisSet

    e_tol: float
    p_rms_tol: float
    maxiter: int
    damping: float

    logger: SCFLogger

    def __init__(self, atoms: list[tuple[int, vec3]], basis_type: str, n_elec: int, logger: SCFLogger, e_tol: float = 1e-8, p_rms_tol: float = 1e-6, maxiter: int = 50, damping: float = 0.1):

        self.basis_set = build_basis_set(atoms, basis_type)
        self.atoms = atoms
        self.n_elec = n_elec

        if self.n_elec % 2 != 0:
            raise ValueError("Program requires even number of electrons.")
        
        self.e_tol = e_tol
        self.p_rms_tol = p_rms_tol
        self.maxiter = maxiter
        self.damping = damping

        self.logger = logger
        self.logger.start(basis_type, self.basis_set, self.atoms, self.n_elec, self.e_tol, self.p_rms_tol, self.maxiter)

    def run(self) -> None:

        self.build_S()
        self.build_X()
        self.build_H()
        self.build_E()
        self.build_Enuc()
        
        self.logger.integrals_done()

        eps, C = self.solve_roothaan(self.H)
        P = self.build_P(C)
        E_elec = self.energy(self.H, P)
        E_tot = E_elec + self.Enuc

        self.logger.scf_begin(eps, C)

        converged = False

        for it in range(1, self.maxiter + 1):

            F = self.build_F(P)

            eps, C = self.solve_roothaan(F)

            P_new = self.build_P(C)

            if self.damping > 0.0:
                P_new = (1.0 - self.damping) * P_new + self.damping * P

            E_elec_new = self.energy(F, P_new)
            E_tot_new = E_elec_new + self.Enuc

            dP = P_new - P
            p_rms = np.sqrt(np.mean(dP * dP))

            dE = E_tot_new - E_tot

            self.logger.iteration(it, E_elec_new, E_tot_new, C, eps, dE, p_rms)

            P = P_new
            E_tot = E_tot_new

            if abs(dE) < self.e_tol and p_rms < self.p_rms_tol:
                converged = True
                break

        F_final = self.build_F(P)
        eps, C = self.solve_roothaan(F_final)
        E_elec_final = self.energy(F_final, P)
        E_tot_final = E_elec_final + self.Enuc

        self.logger.scf_end(eps, C)

        if converged:
            self.logger.converged(it, E_tot_final)
        if not converged:
            self.logger.failed(self.maxiter, E_tot_final)

        self.logger.end()

    def build_S(self) -> None:
        t0 = perf_counter()

        self.S = np.zeros((len(self.basis_set.cgtos), len(self.basis_set.cgtos)))

        for i, A in enumerate(self.basis_set.cgtos):
            for j, B in enumerate(self.basis_set.cgtos):
                self.S[i][j] = overlap_cgto(A, B)

        self.logger.tS = perf_counter() - t0

    def build_H(self) -> None:
        t0 = perf_counter()

        self.H = np.zeros((len(self.basis_set.cgtos), len(self.basis_set.cgtos)))

        for i, A in enumerate(self.basis_set.cgtos):
            for j, B in enumerate(self.basis_set.cgtos):

                self.H[i][j] = kinetic_cgto(A, B)

                for atom in self.atoms:
                    Z, coor = atom

                    self.H[i][j] += -Z*nucatr_cgto(A, B, coor)

        self.logger.tH = perf_counter() - t0


    def build_E(self) -> None:
        t0 = perf_counter()
        cgtos = self.basis_set.cgtos
        n = len(cgtos)
        E = np.zeros((n, n, n, n), dtype=np.float64)

        def pair_index(a, b):
            return a*(a+1)//2 + b
        
        npair = n * (n + 1) // 2
        bounds = np.zeros(npair)

        for i in tqdm(range(n), desc = "Computing the Schwarz screening..."):
            for j in range(i + 1):
                pij = pair_index(i, j)
                bounds[pij] = np.sqrt(abs(twoel_cgto(cgtos[i], cgtos[j], cgtos[i], cgtos[j])))

        eri_thresh = 1e-10

        bar = tqdm(total=n**4/8, desc = "Computing ERI...", smoothing=0)

        for i in range(n):
            for j in range(i+1):
                pij = pair_index(i, j)
                for k in range(n):
                    for l in range(k+1):
                        pkl = pair_index(k, l)
                        if pij < pkl:
                            continue
                        if bounds[pij] * bounds[pkl] < eri_thresh:
                            bar.update(1)
                            continue

                        v = twoel_cgto(cgtos[i], cgtos[j], cgtos[k], cgtos[l])
                        bar.update(1)

                        E[i,j,k,l] = v
                        E[j,i,k,l] = v
                        E[i,j,l,k] = v
                        E[j,i,l,k] = v
                        E[k,l,i,j] = v
                        E[l,k,i,j] = v
                        E[k,l,j,i] = v
                        E[l,k,j,i] = v
        bar.close()
        self.E = E
        self.logger.tE = perf_counter() - t0

    def build_X(self, thresh: float = 1e-12) -> None:
        t0 = perf_counter()

        s, U = np.linalg.eigh(self.S)
        if np.any(s < -1e-10):
            raise ValueError("Overlap matrix S has significantly negative eigenvalues (basis issue).")
        s_clipped = np.clip(s, thresh, None)
        self.X = (U * (s_clipped ** -0.5)) @ U.T

        self.logger.tX = perf_counter() - t0

    def build_F(self, P: Array) -> Array:

        J = np.einsum("ls,mnls->mn", P, self.E, optimize=True)

        K = np.einsum("ls,mlns->mn", P, self.E, optimize=True)

        F = self.H + J - 0.5 * K
        F = 0.5 * (F + F.T)
        return F
    
    def build_Enuc(self) -> None:
        t0 = perf_counter()

        Enuc = 0.0
        for a in range(len(self.atoms)):
            Za, Ra = self.atoms[a]
            for b in range(a + 1, len(self.atoms)):
                Zb, Rb = self.atoms[b]
                Rab = np.linalg.norm(Ra - Rb)
                if Rab < 1e-12:
                    raise ValueError("Two nuclei are at the same position (R_AB ~ 0).")
                Enuc += Za * Zb / Rab

        self.Enuc = Enuc

        self.logger.tEnuc = perf_counter() - t0

    def solve_roothaan(self, F: Array) -> tuple[Array, Array]:
        Fp = self.X.T @ F @ self.X
        eps, Cp = np.linalg.eigh(Fp)
        C = self.X @ Cp
        return eps, C
    
    def build_P(self, C: Array) -> Array:
        nocc = self.n_elec // 2
        Cocc = C[:, :nocc]
        P = 2.0 * (Cocc @ Cocc.T)
        return P
    
    def energy(self, F: Array, P: Array) -> float:
        return 0.5*np.einsum("mn,mn->", P, (self.H + F), optimize=True)