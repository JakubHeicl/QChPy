import numpy as np
from dataclasses import dataclass, field
from utils import norm2, primitive_norm, F_n
import time
from functools import lru_cache

vec3 = np.ndarray

@dataclass
class PrimitiveGaussian:
    alpha: float
    center: vec3
    ang: tuple[int, int, int]
    coeff: float
    norm: float = field(init=False)

    def __post_init__(self):
        self.norm = primitive_norm(self.alpha, self.ang)

@dataclass
class ContractedGaussian:
    center: vec3
    ang: tuple[int, int, int]
    primitives: list[PrimitiveGaussian] = field(default_factory=list)
    _norm: float | None = None
    
    def __str__(self):
        prims = ", ".join(f"({p.coeff:12.8f}, {p.alpha:12.8f})" for p in self.primitives)
        return (
            f"[{self.center[0]:12.8f} {self.center[1]:12.8f} {self.center[2]:12.8f}], "
            f"({self.ang[0]}, {self.ang[1]}, {self.ang[2]}), "
            f"{prims}\n"
        )
    def add_primitive(self, alpha: float, coeff: float) -> None:
        self.primitives.append(PrimitiveGaussian(alpha, self.center, self.ang, coeff))
        self._norm = None

    @property
    def norm(self):
        if self._norm is None:
            S = 0

            for Ap in self.primitives:
                for Bp in self.primitives:
                    if Ap is Bp: 
                        S += Ap.coeff*Bp.coeff
                    else:
                        S += Ap.coeff*Bp.coeff*overlap_pgto(Ap, Bp)

            self._norm = 1.0/np.sqrt(S)
            return self._norm
        else:
            return self._norm


def productAB(A: PrimitiveGaussian, B: PrimitiveGaussian) -> tuple[float, float, float, vec3]:
    """
    :param A: G_A
    :type A: PrimitiveGaussian
    :param B: G_B
    :type B: PrimitiveGaussian
    :return: Returns p, mu, RAB2, P from the gaussian product theorem G_A*G_B = exp(-mu*RAB2)*exp(-p(r-P)**2)
    :rtype: tuple[float, float, float, vec3]
    """
    p = A.alpha + B.alpha                       #float
    mu = A.alpha*B.alpha/p                      #float
    RAB2 = norm2(A.center-B.center)             #float
    P = (A.alpha*A.center+B.alpha*B.center)/p   #vec3

    return p, mu, RAB2, P

def _S1D(i: int, j: int, mu: float, p: float, Ai: float, Bi: float, Pi: float) -> float:

    if i < 0 or j < 0: 
        return 0.0
    elif i == 0 and j == 0: 
        return np.sqrt(np.pi/p)*np.exp(-mu*(Ai-Bi)**2)
    elif i > 0:
        return (Pi-Ai)*_S1D(i-1, j, mu, p, Ai, Bi, Pi) + (i-1)*_S1D(i-2, j, mu, p, Ai, Bi, Pi)/(2*p) + j*_S1D(i-1, j-1, mu, p, Ai, Bi, Pi)/(2*p)
    else:
        return (Pi-Bi)*_S1D(i, j-1, mu, p, Ai, Bi, Pi) + i*_S1D(i-1, j-1, mu, p, Ai, Bi, Pi)/(2*p) + (j-1)*_S1D(i, j-2, mu, p, Ai, Bi, Pi)/(2*p)

def _T1D(i: int, j: int, a: float, b: float, mu: float, p: float, Ai: float, Bi: float, Pi: float) -> float:
    
    if i < 0 or j < 0: 
        return 0.0
    elif i == 0 and j == 0: 
        return (a-2*a**2*((Pi-Ai)**2 + 1/(2*p)))*_S1D(0, 0, mu, p, Ai, Bi, Pi)
    elif i > 0:
        return (Pi-Ai)*_T1D(i-1, j, a, b, mu, p, Ai, Bi, Pi) + ((i-1)*_T1D(i-2, j, a, b, mu, p, Ai, Bi, Pi) + j*_T1D(i-1, j-1, a, b, mu, p, Ai, Bi, Pi))/(2*p) + b*(2*a*_S1D(i, j, mu, p, Ai, Bi, Pi) - (i-1)*_S1D(i-2, j, mu, p, Ai, Bi, Pi))/p
    else:
        return (Pi-Bi)*_T1D(i, j-1, a, b, mu, p, Ai, Bi, Pi) + (j*_T1D(i-1, j-1, a, b, mu, p, Ai, Bi, Pi) + (j-1)*_T1D(i, j-2, a, b, mu, p, Ai, Bi, Pi))/(2*p) + a*(2*b*_S1D(i, j, mu, p, Ai, Bi, Pi) - (j-1)*_S1D(i, j-1, mu, p, Ai, Bi, Pi))/p 

def overlap_pgto(A: PrimitiveGaussian, B: PrimitiveGaussian) -> float:
    """
    :param A: G_A
    :type A: PrimitiveGaussian
    :param B: G_B
    :type B: PrimitiveGaussian
    :return: Returns the value of the overlap integral between two primitive gaussians <G_A|G_B> using the Obara-Saika recursion relations
    :rtype: float
    """
    p, mu, _, P = productAB(A, B)

    i, k, m = A.ang
    j, l, n = B.ang

    Ax, Ay, Az = A.center
    Bx, By, Bz = B.center
    Px, Py, Pz = P

    Sij = _S1D(i, j, mu, p, Ax, Bx, Px)
    Skl = _S1D(k, l, mu, p, Ay, By, Py)
    Smn = _S1D(m, n, mu, p, Az, Bz, Pz)

    return Sij*Skl*Smn*A.norm*B.norm

def kinetic_pgto(A: PrimitiveGaussian, B: PrimitiveGaussian) -> float:
    """
    :param A: G_A
    :type A: PrimitiveGaussian
    :param B: G_B
    :type B: PrimitiveGaussian
    :return: Returns the value of the kinetic integral between two primitive gaussians -1/2*<G_A|nabla**2|G_B> using the Obara-Saika recursion relations
    :rtype: float
    """
    p, mu, _, P = productAB(A, B)

    a = A.alpha
    b = B.alpha

    i, k, m = A.ang
    j, l, n = B.ang

    Ax, Ay, Az = A.center
    Bx, By, Bz = B.center
    Px, Py, Pz = P

    Sij = _S1D(i, j, mu, p, Ax, Bx, Px)
    Skl = _S1D(k, l, mu, p, Ay, By, Py)
    Smn = _S1D(m, n, mu, p, Az, Bz, Pz)

    Tij = _T1D(i, j, a, b, mu, p, Ax, Bx, Px)
    Tkl = _T1D(k, l, a, b, mu, p, Ay, By, Py)
    Tmn = _T1D(m, n, a, b, mu, p, Az, Bz, Pz)

    return (Tij*Skl*Smn + Sij*Tkl*Smn + Sij*Skl*Tmn)*A.norm*B.norm

def nucatr_pgto(A: PrimitiveGaussian, B: PrimitiveGaussian, C: vec3) -> float:
    """
    :param A: G_A
    :type A: PrimitiveGaussian
    :param B: G_B
    :type B: PrimitiveGaussian
    :param C: Coordinates of center C
    :type C: vec3
    :return: Returns the value of nuclear attraction integral between two primitive gaussians <G_A|1/r_C|G_B>
    :rtype: float
    """
    p, mu, RAB2, P = productAB(A, B)

    XPC, YPC, ZPC = P-C
    XPA, YPA, ZPA = P-A.center
    XPB, YPB, ZPB = P-B.center

    RPC2 = norm2(P-C)
    Kab = np.exp(-mu*RAB2)
    base_pref = (2*np.pi/p)*Kab

    i, k, m = A.ang
    j, l, n = B.ang
    
    @lru_cache(maxsize=None)
    def theta(N, i, j, k, l, m, n) -> float:
        if min(i, j, k, l, m, n) < 0:
            return 0.0
        elif i == j == k == l == m == n == 0:
            return base_pref*F_n(N, p*RPC2)
        
        elif i > 0:
            i -= 1
            return XPA*theta(N, i, j, k, l, m, n) + (i*theta(N, i-1, j, k, l, m, n) + j*theta(N, i, j-1, k, l, m, n))/(2*p) - XPC*theta(N+1, i, j, k, l, m, n) - (i*theta(N+1, i-1, j, k, l, m, n) + j*theta(N+1, i, j-1, k, l, m, n))/(2*p)
        elif j > 0:
            j -= 1
            return XPB*theta(N, i, j, k, l, m, n) + (i*theta(N, i-1, j, k, l, m, n) + j*theta(N, i, j-1, k, l, m, n))/(2*p) - XPC*theta(N+1, i, j, k, l, m, n) - (i*theta(N+1, i-1, j, k, l, m, n) + j*theta(N+1, i, j-1, k, l, m, n))/(2*p)
        
        elif k > 0:
            k -= 1
            return YPA*theta(N, i, j, k, l, m, n) + (k*theta(N, i, j, k-1, l, m, n) + l*theta(N, i, j, k, l-1, m, n))/(2*p) - YPC*theta(N+1, i, j, k, l, m, n) - (k*theta(N+1, i, j, k-1, l, m, n) + l*theta(N+1, i, j, k, l-1, m, n))/(2*p)
        elif l > 0:
            l -= 1
            return YPB*theta(N, i, j, k, l, m, n) + (k*theta(N, i, j, k-1, l, m, n) + l*theta(N, i, j, k, l-1, m, n))/(2*p) - YPC*theta(N+1, i, j, k, l, m, n) - (k*theta(N+1, i, j, k-1, l, m, n) + l*theta(N+1, i, j, k, l-1, m, n))/(2*p)
        
        elif m > 0:
            m -= 1
            return ZPA*theta(N, i, j, k, l, m, n) + (m*theta(N, i, j, k, l, m-1, n) + n*theta(N, i, j, k, l, m, n-1))/(2*p) - ZPC*theta(N+1, i, j, k, l, m, n) - (m*theta(N+1, i, j, k, l, m-1, n) + n*theta(N+1, i, j, k, l, m, n-1))/(2*p)
        else:
            n -= 1
            return ZPB*theta(N, i, j, k, l, m, n) + (m*theta(N, i, j, k, l, m-1, n) + n*theta(N, i, j, k, l, m, n-1))/(2*p) - ZPC*theta(N+1, i, j, k, l, m, n) - (m*theta(N+1, i, j, k, l, m-1, n) + n*theta(N+1, i, j, k, l, m, n-1))/(2*p)

    value = theta(0, i, j, k, l, m, n)
    return value*A.norm*B.norm

def twoel_pgto(A: PrimitiveGaussian, B: PrimitiveGaussian, C: PrimitiveGaussian, D: PrimitiveGaussian) -> float:
    """
    :param A: G_A
    :type A: PrimitiveGaussian
    :param B: G_B
    :type B: PrimitiveGaussian
    :param C: G_C
    :type C: PrimitiveGaussian
    :param D: G_D
    :type D: PrimitiveGaussian
    :return: Returns the value of two electron integral between four primitive gaussians <G_A*G_B|1/r_12|G_C*G_D>
    :rtype: float
    """
    p, mu, RAB2, P = productAB(A, B)
    q, nu, RCD2, Q = productAB(C, D)
    alpha = (p * q) / (p + q)

    ix, iy, iz = A.ang
    jx, jy, jz = B.ang
    kx, ky, kz = C.ang
    lx, ly, lz = D.ang

    XPA, YPA, ZPA = (P - A.center)
    XPB, YPB, ZPB = (P - B.center)
    XQC, YQC, ZQC = (Q - C.center)
    XQD, YQD, ZQD = (Q - D.center)
    XPQ, YPQ, ZPQ = (P - Q)

    RPQ2 = norm2(P - Q)
    Kab = np.exp(-mu * RAB2)
    Kcd = np.exp(-nu * RCD2)
    pre = (2 * np.pi ** (5 / 2)) / (p * q * np.sqrt(p + q))
    base_pref = Kab * Kcd * pre

    @lru_cache(maxsize=None)
    def theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) -> float:
        if min(ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) < 0:
            return 0.0
        elif ix == jx == kx == lx == iy == jy == ky == ly == iz == jz == kz == lz == 0:
            return base_pref*F_n(N, alpha*RPQ2)
        
        elif ix > 0:
            ix -= 1
            result = XPA*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*XPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p \
                    + ix*(theta(N, ix-1, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix-1, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + jx*(theta(N, ix, jx-1, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx-1, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + kx/(2*(p+q))*theta(N+1, ix, jx, kx-1, lx, iy, jy, ky, ly, iz, jz, kz, lz) + lx/(2*(p+q))*theta(N+1, ix, jx, kx, lx-1, iy, jy, ky, ly, iz, jz, kz, lz)
            return result
        elif jx > 0:
            jx -= 1
            result = XPB*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*XPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p \
                    + ix*(theta(N, ix-1, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix-1, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + jx*(theta(N, ix, jx-1, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx-1, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + kx/(2*(p+q))*theta(N+1, ix, jx, kx-1, lx, iy, jy, ky, ly, iz, jz, kz, lz) + lx/(2*(p+q))*theta(N+1, ix, jx, kx, lx-1, iy, jy, ky, ly, iz, jz, kz, lz)
            return result
        elif kx > 0:
            kx -= 1
            result = XQC*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + alpha*XPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q \
                    + kx*(theta(N, ix, jx, kx-1, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx-1, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q)/(2*q) \
                    + lx*(theta(N, ix, jx, kx, lx-1, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx-1, iy, jy, ky, ly, iz, jz, kz, lz)/q)/(2*q) \
                    + ix/(2*(p+q))*theta(N+1, ix-1, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + jx/(2*(p+q))*theta(N+1, ix, jx-1, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)
            return result
        elif lx > 0:
            lx -= 1
            result = XQD*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + alpha*XPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q \
                    + kx*(theta(N, ix, jx, kx-1, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx-1, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q)/(2*q) \
                    + lx*(theta(N, ix, jx, kx, lx-1, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx-1, iy, jy, ky, ly, iz, jz, kz, lz)/q)/(2*q) \
                    + ix/(2*(p+q))*theta(N+1, ix-1, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + jx/(2*(p+q))*theta(N+1, ix, jx-1, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)
            return result
        
        elif iy > 0:
            iy -= 1
            result = YPA*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*YPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p \
                    + iy*(theta(N, ix, jx, kx, lx, iy-1, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy-1, jy, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + jy*(theta(N, ix, jx, kx, lx, iy, jy-1, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy-1, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + ky/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky-1, ly, iz, jz, kz, lz) + ly/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly-1, iz, jz, kz, lz)
            return result
        elif jy > 0:
            jy -= 1
            result = YPB*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*YPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p \
                    + iy*(theta(N, ix, jx, kx, lx, iy-1, jy, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy-1, jy, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + jy*(theta(N, ix, jx, kx, lx, iy, jy-1, ky, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy-1, ky, ly, iz, jz, kz, lz)/p)/(2*p) \
                    + ky/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky-1, ly, iz, jz, kz, lz) + ly/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly-1, iz, jz, kz, lz)
            return result
        elif ky > 0:
            ky -= 1
            result = YQC*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + alpha*YPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q \
                    + ky*(theta(N, ix, jx, kx, lx, iy, jy, ky-1, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky-1, ly, iz, jz, kz, lz)/q)/(2*q) \
                    + ly*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly-1, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly-1, iz, jz, kz, lz)/q)/(2*q) \
                    + iy/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy-1, jy, ky, ly, iz, jz, kz, lz) + jy/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy-1, ky, ly, iz, jz, kz, lz)
            return result
        elif ly > 0:
            ly -= 1
            result = YQD*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + alpha*YPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q \
                    + ky*(theta(N, ix, jx, kx, lx, iy, jy, ky-1, ly, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky-1, ly, iz, jz, kz, lz)/q)/(2*q) \
                    + ly*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly-1, iz, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly-1, iz, jz, kz, lz)/q)/(2*q) \
                    + iy/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy-1, jy, ky, ly, iz, jz, kz, lz) + jy/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy-1, ky, ly, iz, jz, kz, lz)
            return result
        
        elif iz > 0:
            iz -= 1
            result = ZPA*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*ZPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p \
                    + iz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz-1, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz-1, jz, kz, lz)/p)/(2*p) \
                    + jz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz-1, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz-1, kz, lz)/p)/(2*p) \
                    + kz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz-1, lz) + lz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz-1)
            return result
        elif jz > 0:
            jz -= 1
            result = ZPB*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) - alpha*ZPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/p \
                    + iz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz-1, jz, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz-1, jz, kz, lz)/p)/(2*p) \
                    + jz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz-1, kz, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz-1, kz, lz)/p)/(2*p) \
                    + kz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz-1, lz) + lz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz-1)
            return result
        elif kz > 0:
            kz -= 1
            result = ZQC*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + alpha*ZPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q \
                    + kz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz-1, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz-1, lz)/q)/(2*q) \
                    + lz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz-1) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz-1)/q)/(2*q) \
                    + iz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz-1, jz, kz, lz) + jz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz-1, kz, lz)
            return result
        else:
            lz -= 1
            result = ZQD*theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz) + alpha*ZPQ*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)/q \
                    + kz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz-1, lz) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz-1, lz)/q)/(2*q) \
                    + lz*(theta(N, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz-1) - alpha*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz-1)/q)/(2*q) \
                    + iz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz-1, jz, kz, lz) + jz/(2*(p+q))*theta(N+1, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz-1, kz, lz)
            return result

    value = theta(0, ix, jx, kx, lx, iy, jy, ky, ly, iz, jz, kz, lz)
    return value * A.norm * B.norm * C.norm * D.norm

def overlap_cgto(A: ContractedGaussian, B: ContractedGaussian) -> float:

    S = 0

    for Ap in A.primitives:
        for Bp in B.primitives:
            S += Ap.coeff*Bp.coeff*overlap_pgto(Ap, Bp)

    return S*A.norm*B.norm

def kinetic_cgto(A: ContractedGaussian, B: ContractedGaussian) -> float:

    S = 0

    for Ap in A.primitives:
        for Bp in B.primitives:
            S += Ap.coeff*Bp.coeff*kinetic_pgto(Ap, Bp)

    return S*A.norm*B.norm

def nucatr_cgto(A: ContractedGaussian, B: ContractedGaussian, C: vec3) -> float:

    S = 0

    for Ap in A.primitives:
        for Bp in B.primitives:
            S += Ap.coeff*Bp.coeff*nucatr_pgto(Ap, Bp, C)

    return S*A.norm*B.norm

def twoel_cgto(A: ContractedGaussian, B: ContractedGaussian, C: ContractedGaussian, D: ContractedGaussian) -> float:

    S = 0

    for Ap in A.primitives:
        for Bp in B.primitives:
            for Cp in C.primitives:
                for Dp in D.primitives:
                    S += Ap.coeff*Bp.coeff*Cp.coeff*Dp.coeff*twoel_pgto(Ap, Bp, Cp, Dp)

    return S*A.norm*B.norm*C.norm*D.norm