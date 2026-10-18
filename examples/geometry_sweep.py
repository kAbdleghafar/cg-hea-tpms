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

  # size effect: one sweep per TPMS cell size (e.g. 30 nm and 40 nm polycrystals with
  # --periods 1, so the cell is the box), then compare E/E_bulk between them
  python3 examples/geometry_sweep.py summarize --outroot sweep_30nm sweep_40nm

Protocol defaults: 300 K, 3 % compression at 1e-3/ps with lateral 0 bar, two seeds, fit
between 0.75 % and 2.5 %. Tested on ~26k-atom Ti-Zr-Nb gyroids: switching from isotropic NPT
to lateral control makes the box ring for the first ~0.75 % of strain (the same in every seed,
so averaging does not remove it); after that the curves are linear to 3 %, the fit gives
R^2 = 0.91-0.96 and seeds agree within ~3-5 %. A 1 % compression was noise-dominated
(R^2 < 0.7). Check every stress-strain curve and the R^2 printed by `summarize`.
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cg_hea_core as core


def generate(a):
    cases = [] if a.no_bulk else [("bulk", "network", 1.0)]
    cases += [(s, "network", d) for s in a.surfaces for d in a.densities]
    cases += [(s, "sheet", d) for s in (a.sheet or []) for d in a.densities]
    if not cases:
        sys.exit("nothing to generate: --no-bulk without --surfaces or --sheet")
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
                              matrix=a.matrix, matrix_species=a.matrix_species,
                              composition=a.composition,
                              surface="gyroid" if surf == "bulk" else surf,
                              periods=a.periods, phase=phase, density=dens, temp=a.temp,
                              dt_aa=a.dt_aa, equil=a.equil, crush=True, crush_strain=a.strain, load=a.load,
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


def collect(outroot, fit_window):
    """Fit every finished run of one sweep. Returns (rows, TPMS cell edge in nm or None)."""
    rows, cell = [], None
    for name in sorted(os.listdir(outroot)):
        d = os.path.join(outroot, name)
        js = [f for f in os.listdir(d) if f.endswith("_summary.json")] if os.path.isdir(d) else []
        if not js:
            continue
        s = json.load(open(os.path.join(d, js[0])))
        if s.get("box_vector") and s.get("tpms_periods"):
            cell = s["box_vector"][0] / s["tpms_periods"][0] / 10.0
        ss = os.path.join(d, s["prefix"] + "_stress_strain.txt")
        if not os.path.isfile(ss):
            print(f"{name}: not run yet")
            continue
        try:
            E, _, r2, k = core.fit_modulus(ss, *fit_window)
        except ValueError as e:
            print(f"{name}: {e}")
            continue
        arch = "bulk" if name.startswith("bulk") else name.rsplit("_rho", 1)[0]
        rows.append(dict(run=name, arch=arch, rho=s["fill"], E=E, R2=r2))
    return rows, cell


def summarize_one(outroot, rows, cell, target, Eb_ref=None):
    bulk = [r["E"] for r in rows if r["arch"] == "bulk"]
    Eb = float(np.mean(bulk)) if bulk else Eb_ref
    print(f"\n== {outroot}" + (f"  (TPMS cell {cell:.3g} nm)" if cell else "")
          + ("" if bulk or Eb is None else f"  [E_bulk {Eb:.2f} GPa from --bulk-root]"))
    print(f"{'architecture':24s} {'rho':>6s} {'E (GPa)':>14s} {'E/E_bulk':>9s}  seeds")
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
    with open(os.path.join(outroot, "moduli.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    if not Eb:
        print("\nno bulk run: relative moduli need the bulk reference")
        return table
    print(f"\nGibson-Ashby fits  E*/E_bulk = C rho^m   (E_bulk = {Eb:.2f} GPa)")
    for arch in sorted({t["architecture"] for t in table} - {"bulk"}):
        pts = [(t["rho"], t["E_rel"]) for t in table if t["architecture"] == arch]
        if len(pts) < 2:
            continue
        x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
        m, lnC = np.polyfit(x, y, 1)
        msg = f"  {arch:24s} C = {np.exp(lnC):.3f}  m = {m:.2f}"
        if target:
            rho_t = (target / Eb / np.exp(lnC)) ** (1 / m)
            msg += f"   -> rho = {rho_t:.3f} for E* = {target:g} GPa"
        print(msg)
    print(f"\nwrote {os.path.join(outroot, 'moduli.csv')}")
    return table


def summarize(a):
    Eb_ref = None
    if a.bulk_root:                       # bulk reference from another sweep (e.g. bulk only)
        ref = [r["E"] for r in collect(a.bulk_root, a.fit_window)[0] if r["arch"] == "bulk"]
        if not ref:
            sys.exit(f"{a.bulk_root}: no finished bulk run")
        Eb_ref = float(np.mean(ref))
        print(f"E_bulk = {Eb_ref:.2f} GPa from {a.bulk_root} ({len(ref)} run(s)), used where a "
              f"sweep has no bulk run of its own")
    tables = {}
    for root in a.outroot:
        rows, cell = collect(root, a.fit_window)
        if not rows:
            print(f"{root}: no finished runs found")
            continue
        tables[root] = (cell, summarize_one(root, rows, cell, a.target, Eb_ref))
    if not tables:
        sys.exit("no finished runs found")
    if len(tables) < 2:
        return
    # size effect: the same architecture and density in sweeps with different TPMS cells
    roots = list(tables)
    key = lambda t: (t["architecture"], round(t["rho"], 2))
    rel = {r: {key(t): t["E_rel"] for t in tables[r][1]} for r in roots}
    common = sorted(set.intersection(*(set(v) for v in rel.values())) - {("bulk", 1.0)})
    head = "".join(f"{(f'{tables[r][0]:.3g} nm' if tables[r][0] else r):>10s}" for r in roots)
    print(f"\nSize effect on E/E_bulk (same architecture and density, different TPMS cells)")
    print(f"{'architecture':24s} {'rho':>5s}{head}  spread")
    for k in common:
        v = np.array([rel[r][k] for r in roots])
        print(f"{k[0]:24s} {k[1]:5.2f}" + "".join(f"{x:10.3f}" for x in v)
              + f"  {100 * np.ptp(v) / v.mean():5.1f} %")
    print("A spread within the seed-to-seed scatter means E/E_bulk no longer depends on the "
          "cell size,\nso it carries over to larger (implant-scale) lattices.")


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
    g.add_argument("--matrix-species", default="file", choices=["file", "random"],
                   help="random: alloy a pure template at --composition before carving "
                        "(the same arrangement for the bulk and every lattice of a seed)")
    g.add_argument("--composition", nargs="+", type=float,
                   help="at.%% or fractions per element (with --matrix-species random)")
    g.add_argument("--lattice", default="bcc", choices=list(core.LATTICES))
    g.add_argument("-n", type=int, default=1, help="CG level (1 = atomistic)")
    g.add_argument("--a0", type=float, default=None, help="only used for the neighbour skin")
    g.add_argument("--surfaces", nargs="*", default=["gyroid", "diamond", "primitive", "iwp"],
                   help="network surfaces (none: only the bulk and --sheet)")
    g.add_argument("--sheet", nargs="*", default=["gyroid"], help="surfaces also run as sheets")
    g.add_argument("--no-bulk", action="store_true",
                   help="skip the bulk run (take E_bulk from another sweep: summarize --bulk-root)")
    g.add_argument("--densities", nargs="+", type=float, default=[0.25, 0.35, 0.5])
    g.add_argument("--periods", type=int, default=1,
                   help="TPMS cells along the shortest box edge (cell size = that edge / periods)")
    g.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    g.add_argument("--temp", type=float, default=300.0)
    g.add_argument("--dt-aa", type=float, default=0.001)
    g.add_argument("--equil", type=int, default=10000)
    g.add_argument("--strain", type=float, default=0.03)
    g.add_argument("--rate", type=float, default=1e-3, help="CG strain rate, 1/ps")
    g.add_argument("--axis", default="z", choices=["x", "y", "z"])
    g.add_argument("--load", default="compression", choices=["compression", "tension"])
    g.add_argument("--verify", action="store_true", help="verify the CG map once (n > 1)")
    g.add_argument("--lmp", default="lmp")
    g.add_argument("--outroot", default="sweep")
    s = sub.add_parser("summarize", help="fit and compare the finished runs")
    s.add_argument("--outroot", nargs="+", default=["sweep"],
                   help="one sweep, or several with different TPMS cell sizes to compare")
    s.add_argument("--fit-window", nargs=2, type=float, default=(0.0075, 0.025))
    s.add_argument("--target", type=float, help="target modulus, GPa (e.g. 20 for bone)")
    s.add_argument("--bulk-root", help="sweep whose bulk run(s) give E_bulk for sweeps without one")
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else summarize(a)


if __name__ == "__main__":
    main()
