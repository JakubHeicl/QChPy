import numpy as np
from scipy.special import gamma, gammainc

vec3 = np.ndarray

_SYMBOLS = [
    "X",
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og"
]

SYMBOL_TO_Z = {s: i for i, s in enumerate(_SYMBOLS) if s != "X"}
Z_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_Z.items()}

ANGSTROM_TO_BOHR = 1.8897261254535

def cartesian_tuples(l) -> list[tuple[int, int, int]]:

    out = []

    for i in range(l, -1, -1):
        for j in range(l-i, -1, -1):
            out.append((i, j, l-i-j))

    return out

def norm2(v: vec3) -> float:
    return np.dot(v, v)

def F_n(n: float, x: float, eps: float = 1e-12) -> float:
    """
    :param x: x
    :type x: float
    :param n: n
    :type n: float
    :return: Returns the Boys function of x of the order of n F_n(x) using the gamma functions definition
    :rtype: float
    """
    if x < 0.0:
        x = 0.0

    if x < eps:
        return (1.0/(2*n + 1) - x/(2*n + 3) + x*x/(2.0*(2*n + 5)) - x*x*x/(6.0*(2*n + 7)))

    return gamma(n+0.5)*gammainc(n+0.5, x)/(2*x**(n+0.5))

def double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    k = n
    while k > 0:
        out *= k
        k -= 2
    return out

def primitive_norm(alpha: float, ang: tuple[int, int, int]) -> float:
    i, j, k = ang
    alpha = alpha
    denom = double_factorial(2 * i - 1) * double_factorial(2 * j - 1) * double_factorial(2 * k - 1)
    pref = (2 * alpha / np.pi) ** (3 / 4)
    ang_part = (4 * alpha) ** ((i + j + k) / 2) / np.sqrt(denom)
    return float(pref * ang_part)

def read_xyz(filename: str, to_bohr: bool = True) -> tuple[int, list[int], list[vec3]]:
    """
    :param filename: Filename of the .xyz file
    :type filename: str
    :return: Returns the number of atoms, list of atom types and list of atom coordinates from a given .xyz file
    :rtype: tuple[int, list[int], list[vec3]]
    """
    number_of_atoms = None
    atom_types = []
    atom_coors = []

    assert filename.endswith(".xyz"), "File has to be in a format of .xyz!"

    with open(filename, "r") as file:

        lines = file.readlines()
        try:
            number_of_atoms = float(lines[0].split()[0])
        except:
            raise RuntimeError("File is not in a good format")
        for line in lines[2:]:
            
            content = line.strip().split()

            Z = SYMBOL_TO_Z.get(content[0])

            if Z is None:
                raise ValueError(f"Unknown element symbol in XYZ: {content[0]}")

            atom_types.append(Z)
            coors = np.array([float(content[1]),
                              float(content[2]),
                              float(content[3])])
            
            if to_bohr:
                coors *= ANGSTROM_TO_BOHR

            atom_coors.append(coors)
            
    if not (number_of_atoms == len(atom_types) == len(atom_coors)):
        raise RuntimeError("The number of atoms does not match the rest of the file")
            
    return number_of_atoms, atom_types, atom_coors