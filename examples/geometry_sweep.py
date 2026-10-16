#!/usr/bin/env python3
"""
geometry_sweep.py - compare TPMS architectures against a bulk reference structure.

Builds every (surface, phase, relative density, seed) combination by carving it out of the
SAME structure as your bulk reference (e.g. your MD polycrystal, --matrix), plus the bulk
itself (density 1), and writes one LAMMPS run per case and a run_all.sh. After the runs,
`summarize` fits the apparent Young's modulus of each run, normalises it by the bulk value,
fits Gibson-Ashby scaling E*/E_bulk = C * rho^m per architecture and, with --target, gives
the relative density each architecture needs to reach a target modulus.

  # 1. generate (atomistic, n = 1, from your polycrystal; types of the file = --elements)
  python3 examples/geometry_sweep.py generate --pair adp --potential W_Mo_Nb_Zr_Ti_Ta.adp.alloy \\
      --elements Ti Zr Nb --matrix polycrystal.data --outroot sweep \\
      --surfaces gyroid diamond primitive iwp --sheet gyroid --densities 0.25 0.35 0.5
  # 2. run (on your cluster):  cd sweep && bash run_all.sh   (edit LMP / MPI prefix inside)
  # 3. analyse
  python3 examples/geometry_sweep.py summarize --outroot sweep --target 20

Protocol defaults: 300 K, 3 % compression at 1e-3/ps with lateral 0 bar, two seeds, fit
between 0.25 % and 1.5 %. In ~26k-atom gyroids at 300 K, compressing to only 1 % left the
lattice stress buried in thermal and barostat noise (R^2 < 0.7), hence the larger strain and
the seeds. Check every stress-strain curve and the R^2 printed by `summarize`.
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cg_hea_core as core


def generate(a):
    cases = [("bulk", "network", 1.0)]
    cases += [(s, "network", d) for s in a.surfaces for d in a.densities]
    cases += [(s, "sheet", d) for s in (a.sheet or []) for d in a.densities]
    os.makedirs(a.outroot, exist_ok=True)
    lines = ["#!/bin/bash", "# edit to your LAMMPS command, e.g. LMP='mpirun -np 32 lmp'",
             f"LMP=${{LMP:-'{a.lmp}'}}", "set -e"]
    first = True
    for surf, phase, dens in cases:
        for seed in a.seeds:
            name = (f"bulk_s{seed}" if surf == "bulk" else
                    f"{surf}_{phase}_rho{int(round(100 * dens)):03d}_s{seed}")
            cfg = core.Config(pair=a.pair, potential=a.potential or "", library=a.library or "",
                              params=a.params or "", lib_elements=a.lib_elements,
                              elements=a.elements, lattice=a.lattice, n=a.n, a0=a.a0,
                              matrix=a.matrix, surface="gyroid" if surf == "bulk" else surf,
                              phase=phase, density=dens, temp=a.temp, dt_aa=a.dt_aa,
                              equil=a.equil, crush=True, crush_strain=a.strain,
                              strain_rate=a.rate, crush_axis=a.axis, seed=seed, lmp=a.lmp,
                              outdir=os.path.join(a.outroot, name),
                              verify=a.verify and first, auto_a0=False)
            first = False
            s = core.run_pipeline(cfg, core.Session(log=lambda m: None))
            lines.append(f"(cd {name} && $LMP -in {s['run_input']} -log log.lammps)")
            print(f"{name:32s} {s['n_beads']:>9,d} beads  fill {s['fill']:.3f}")
    with open(os.path.join(a.outroot, "run_all.sh"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {len(lines) - 4} runs; next: cd {a.outroot} && bash run_all.sh")


def summarize(a):
    rows = []
    for name in sorted(os.listdir(a.outroot)):
        d = os.path.join(a.outroot, name)
        js = [f for f in os.listdir(d) if f.endswith("_summary.json")] if os.path.isdir(d) else []
        if not js:
            continue
        s = json.load(open(os.path.join(d, js[0])))
        ss = os.path.join(d, s["prefix"] + "_stress_strain.txt")
        if not os.path.isfile(ss):
            print(f"{name}: not run yet")
            continue
        try:
            E, _, r2, k = core.fit_modulus(ss, *a.fit_window)
        except ValueError as e:
            print(f"{name}: {e}")
            continue
        arch = "bulk" if name.startswith("bulk") else name.rsplit("_rho", 1)[0]
        rows.append(dict(run=name, arch=arch, rho=s["fill"], E=E, R2=r2))
    if not rows:
        sys.exit("no finished runs found")
    bulk = [r["E"] for r in rows if r["arch"] == "bulk"]
    Eb = float(np.mean(bulk)) if bulk else None
    print(f"\n{'architecture':24s} {'rho':>6s} {'E (GPa)':>14s} {'E/E_bulk':>9s}  seeds")
    groups = {}
    for r in rows:
        groups.setdefault((r["arch"], round(r["rho"], 3)), []).append(r)
    table = []
    for (arch, rho), g in sorted(groups.items()):
        Es = np.array([r["E"] for r in g])
        rel = Es.mean() / Eb if Eb else float("nan")
        rho = float(np.mean([r["rho"] for r in g]))              # exact fill, not the key
        table.append(dict(architecture=arch, rho=rho, E_mean=Es.mean(), E_std=Es.std(),
                          E_rel=rel, n_seeds=len(g), min_R2=min(r["R2"] for r in g)))
        print(f"{arch:24s} {rho:6.3f} {Es.mean():8.2f} ± {Es.std():4.2f} {rel:9.3f}  {len(g)}"
              + ("   ** low R^2: check the curve **" if min(r["R2"] for r in g) < 0.9 else ""))
    with open(os.path.join(a.outroot, "moduli.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    if not Eb:
        print("\nno bulk run: relative moduli need the bulk reference")
        return
    print(f"\nGibson-Ashby fits  E*/E_bulk = C rho^m   (E_bulk = {Eb:.2f} GPa)")
    for arch in sorted({t["architecture"] for t in table} - {"bulk"}):
        pts = [(t["rho"], t["E_rel"]) for t in table if t["architecture"] == arch]
        if len(pts) < 2:
            continue
        x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
        m, lnC = np.polyfit(x, y, 1)
        msg = f"  {arch:24s} C = {np.exp(lnC):.3f}  m = {m:.2f}"
        if a.target:
            rho_t = (a.target / Eb / np.exp(lnC)) ** (1 / m)
            msg += f"   -> rho = {rho_t:.3f} for E* = {a.target:g} GPa"
        print(msg)
    print(f"\nwrote {os.path.join(a.outroot, 'moduli.csv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the sweep")
    g.add_argument("--pair", required=True, choices=list(core.PAIR_STYLES))
    g.add_argument("--potential")
    g.add_argument("--library")
    g.add_argument("--params")
    g.add_argument("--lib-elements", nargs="+")
    g.add_argument("--elements", nargs="+", required=True, help="element of type 1, 2, ...")
    g.add_argument("--matrix", required=True, help="bulk reference structure (LAMMPS data)")
    g.add_argument("--lattice", default="bcc", choices=list(core.LATTICES))
    g.add_argument("-n", type=int, default=1, help="CG level (1 = atomistic)")
    g.add_argument("--a0", type=float, default=None, help="only used for the neighbour skin")
    g.add_argument("--surfaces", nargs="+", default=["gyroid", "diamond", "primitive", "iwp"])
    g.add_argument("--sheet", nargs="*", default=["gyroid"], help="surfaces also run as sheets")
    g.add_argument("--densities", nargs="+", type=float, default=[0.25, 0.35, 0.5])
    g.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    g.add_argument("--temp", type=float, default=300.0)
    g.add_argument("--dt-aa", type=float, default=0.001)
    g.add_argument("--equil", type=int, default=10000)
    g.add_argument("--strain", type=float, default=0.03)
    g.add_argument("--rate", type=float, default=1e-3, help="CG strain rate, 1/ps")
    g.add_argument("--axis", default="z", choices=["x", "y", "z"])
    g.add_argument("--verify", action="store_true", help="verify the CG map once (n > 1)")
    g.add_argument("--lmp", default="lmp")
    g.add_argument("--outroot", default="sweep")
    s = sub.add_parser("summarize", help="fit and compare the finished runs")
    s.add_argument("--outroot", default="sweep")
    s.add_argument("--fit-window", nargs=2, type=float, default=(0.0025, 0.015))
    s.add_argument("--target", type=float, help="target modulus, GPa (e.g. 20 for bone)")
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else summarize(a)


if __name__ == "__main__":
    main()
