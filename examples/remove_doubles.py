#!/usr/bin/env python3
"""
remove_doubles.py - fast equivalent of Atomsk's `-remove-doubles R` (`-rmd R`) for big
polycrystals. Atomsk's own -rmd takes over an hour on a ~1.5-million-atom box on 4 cores.
This takes seconds and applies the same rule: loop over the atoms in file order; every atom
still present removes all its neighbours closer than R (periodic box).

Checked against Atomsk on a 95,000-atom polycrystal: the same 92,977 atoms kept, at identical
positions.

  atomsk --polycrystal BCC_Ti.xsf polycrystal_30nm.txt poly_raw.lmp -wrap
  python3 examples/remove_doubles.py poly_raw.lmp polycrystal_30nm.lmp 2.0

Reads and writes LAMMPS data files with atom_style atomic and an orthogonal box (as written
by Atomsk).
"""
import sys

import numpy as np
from scipy.spatial import cKDTree


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    R = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    lines = open(src).read().split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.strip().startswith("Atoms"))
    style = lines[i0].split("#")[1].strip() if "#" in lines[i0] else "atomic"
    if style != "atomic":
        sys.exit(f"{src}: atom_style '{style}' - only atomic data files are supported")
    head, rows, i = lines[:i0 + 1], [], i0 + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines) and lines[i].strip():
        rows.append(lines[i])
        i += 1
    rest = lines[i:]                                   # e.g. a Velocities section
    if any(l.strip() for l in rest):
        sys.exit(f"{src}: sections after Atoms are not supported")
    bounds = {l.split()[2][0]: (float(l.split()[0]), float(l.split()[1]))
              for l in head if l.strip().endswith(("xlo xhi", "ylo yhi", "zlo zhi"))}
    if any(l.strip().endswith("xy xz yz") for l in head):
        sys.exit(f"{src}: triclinic boxes are not supported")
    lo = np.array([bounds[k][0] for k in "xyz"])
    L = np.array([bounds[k][1] - bounds[k][0] for k in "xyz"])
    a = np.array([r.split()[:5] for r in rows], float)
    pos = np.mod(a[:, 2:5] - lo, L)
    pairs = cKDTree(pos, boxsize=L).query_pairs(R, output_type="ndarray")
    pairs.sort(axis=1)                                  # (i, j) with i < j
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    start = np.searchsorted(pairs[:, 0], np.arange(len(pos) + 1))
    alive = np.ones(len(pos), bool)
    for k in range(len(pos)):                           # file order, as Atomsk does
        if alive[k] and start[k] < start[k + 1]:
            alive[pairs[start[k]:start[k + 1], 1]] = False
    keep = np.where(alive)[0]
    out = [f"{len(keep):>12d}  atoms" if l.strip().endswith(" atoms") and l.split()[0].isdigit()
           else l for l in head]
    out.append("")
    out += [f"{n:>10d} {int(a[k, 1]):>4d} {a[k, 2]:20.12f} {a[k, 3]:20.12f} {a[k, 4]:20.12f}"
            for n, k in enumerate(keep, 1)]
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"{src}: {len(pos):,} atoms, {len(pairs):,} pairs closer than {R:g} A; removed "
          f"{len(pos) - len(keep):,}, kept {len(keep):,} -> {dst}")


if __name__ == "__main__":
    main()
