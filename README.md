# Hartree-Fock in Python

This project is a small educational implementation of a closed-shell restricted Hartree-Fock (RHF) solver written in Python. It reads molecular geometries from `.xyz` files, builds contracted Gaussian basis functions, evaluates one- and two-electron integrals, runs a self-consistent field (SCF) cycle, and writes a detailed text report with energies, orbitals, and timing information.

## What the project does

The main script, `HF_program.py`, performs the following steps:

1. Reads a molecular geometry from an `.xyz` file.
2. Converts coordinates from Angstrom to bohr.
3. Builds a contracted Gaussian basis set for the selected basis.
4. Computes:
   - overlap matrix `S`
   - orthogonalization matrix `X`
   - core Hamiltonian `H`
   - two-electron repulsion tensor, with Schwarz screening used to skip negligible quartets
   - nuclear repulsion energy
5. Solves the Roothaan equations in an SCF loop until the energy and density converge.
6. Writes a human-readable output file with orbitals, energies, convergence history, and timings.

## Main entries

- `HF_program.py` - command-line entry point and top-level RHF workflow
- `SCF.py` - SCF driver, matrix construction, Roothaan solver, and RHF energy evaluation
- `SCFLogger.py` - formatted output writer for run metadata, timings, orbitals, and SCF history
- `gaussian.py` - primitive/contracted Gaussian objects and integral routines
- `basis_set.py` - basis set loading and construction of contracted basis functions
- `utils.py` - geometry parsing, normalization helpers, Boys function, and element tables
- `basis_sets/` - basis set data files used by the solver

## Requirements

- Python 3.10+
- `numpy`
- `scipy`

## Input format

The main solver expects a standard `.xyz` geometry file:

- line 1: number of atoms
- line 2: comment line
- remaining lines: `ElementSymbol x y z`

Coordinates are expected in Angstrom and are internally converted to bohr.

Example:

```text
3
water
O  0.10297483  0.41189931  0.00000000
H  1.06297483  0.41189931  0.00000000
H -0.21747975  1.31683514  0.00000000
```

## Supported basis sets

The solver currently supports:

- `STO-2G`
- `STO-3G`
- `STO-4G`
- `STO-5G`
- `STO-6G`
- `3-21G`
- `6-21G`
- `6-31G`
- `6-311G`
- `6-31++G`
- `6-31++G**`
- `6-311+G`

but the sets can be easily extended in the JSON format for example from https://www.basissetexchange.org/. Note that not all of the types of Gaussian functions present might be compatible tho. You can also print the available basis names from the CLI:

```bash
python HF_program.py --showb
```

## Running the Hartree-Fock solver

Basic example:

```bash
python HF_program.py --input H2O.xyz --basis STO-3G
```

Specify the output file explicitly:

```bash
python HF_program.py --input NH3.xyz --basis 6-31G --output NH3_631g.out
```

Use custom SCF settings:

```bash
python HF_program.py --input H2O.xyz --basis STO-2G --etol 1e-8 --ptol 1e-6 --maxit 50 --damping 0.1
```

### Command-line arguments

- `--input` - input geometry file in `.xyz` format
- `--basis` - basis set name
- `--showb` - print available basis sets and exit
- `--nelec` - total number of electrons; if omitted, the program uses `sum(Z)`
- `--etol` - SCF energy convergence threshold
- `--ptol` - SCF density convergence threshold
- `--maxit` - maximum number of SCF iterations
- `--damping` - density damping factor
- `--output` - output report filename; if omitted, `<input_stem>.out` is used

### Notes

- The implementation is restricted to closed-shell systems.
- The number of electrons must be even.
- If `--nelec` is not provided, the program assumes a neutral system with `n_elec = sum(Z)`.
- Basis names are treated case-insensitively by the CLI.

## Output

The `.out` file written by `HF_program.py` contains:

- run information and chosen SCF settings
- atom coordinates in bohr
- the full contracted Gaussian basis set used in the calculation
- timings for integral construction
- initial orbital energies and coefficients from the core Hamiltonian guess
- SCF iteration table with total energy, energy change, and density RMS error
- final orbital energies and coefficients
- convergence status and final total energy

If no output filename is given, the default output name is based on the input geometry file. For example:

- `H2O.xyz` -> `H2O.out`

## Using the code from Python

The top-level function can also be imported and called directly:

```python
from HF_program import HF_program

HF_program(
    xyz_filename="H2O.xyz",
    basis_type="STO-3G",
    n_elec=None,
    e_tol=1e-8,
    p_rms_tol=1e-6,
    maxiter=50,
    damping=0.1,
    out_filename="H2O.out",
)
```

## Example files in the repository

The repository includes several sample geometries and outputs, such as:

- `H2O.xyz`
- `NH3.xyz`
- `guanine.xyz`

There are also example `.out`, and related files that can be used for testing or comparison.

## Milestones

A notable recent milestone is that the solver now converges a guanine test case in the minimal `STO-2G` basis. The included example output `examples/guanine.out` shows:

- 16 atoms
- 60 basis functions
- SCF convergence in 50 iterations
- final total energy `-516.676802918731 Eh`
- total run time `2652.58 s`

This is a useful regression and performance checkpoint because earlier versions of the project were not practical for a system of this size.

The current ERI build also uses Schwarz screening, which significantly reduces the number of negligible two-electron quartets that need to be evaluated in practice.

## Limitations

- Restricted to closed-shell RHF
- No geometry optimization
- No unrestricted or open-shell treatment
- No DIIS or more advanced SCF acceleration
- Even with Schwarz screening, the full electron repulsion tensor is still built explicitly, so larger basis sets can become slow and memory-intensive

## Summary

If you want to run a small Hartree-Fock calculation from an `.xyz` file and inspect the basis, matrices, orbitals, convergence, and final energy in a transparent way, `HF_program.py` is the main entry point of this project.
