"""
cg_hea_tpms.py — Coarse-grained TPMS cells for FCC and BCC multicomponent alloys
(high- and medium-entropy alloys), command-line front-end. Run without arguments
(or with --gui) to open the graphical interface instead.
=======================================================================================
One command:  rescale potential (EAM, EAM/FS, ADP or MEAM)  ->  relax random-alloy lattice
              constant (FCC or BCC)  ->  verify the CG mapping  ->  CG lattice + TPMS mask
              + random species  ->  LAMMPS data + run input (equilibration, compression)

Coarse-graining (Chalamet, Rodney, Shibuta, Comput. Mater. Sci. 228 (2023) 112306):
    E_CG(X) = n^3 * E_AA(X/n)  -> thermostat at T_CG = n^3 T, timestep n*dt.
See cg_hea_core.py for the details of the exact mapping.

Examples
  # FCC FeNiCoMn with the Choi et al. 2018 2NN-MEAM (NIST IPR: 2018--Choi-W-M-...-Co-Ni-Cr-Fe-Mn)
  python3 cg_hea_tpms.py --pair meam --library library.meam --params CoNiCrFeMn.meam \\
      --lib-elements Co Ni Cr Fe Mn --elements Fe Ni Co Mn --lattice fcc -n 5 \\
      --surface gyroid --cells 20 --auto-a0 --verify --crush

  # BCC refractory HEA (any eam/alloy file containing the elements)
  python3 cg_hea_tpms.py --pair eam --potential MoNbTaW.eam.alloy --elements Nb Mo Ta W \\
      --lattice bcc -n 5 --auto-a0 --verify --compare-phases --surface gyroid --density 0.3

  # BCC refractory HEA with an angular-dependent potential (ADP, setfl + u,w tables)
  python3 cg_hea_tpms.py --pair adp --potential W_Mo_Nb_Zr_Ti_Ta.adp.alloy \\
      --elements Mo Nb Ta Ti W Zr --lattice bcc -n 5 --box-nm 100 --auto-a0 --verify --crush

  # meta-atom TWIP steel (Wang et al. JMPS 2017, NIST IPR TWIP.eam.fs): one averaged species
  python3 cg_hea_tpms.py --pair eamfs --potential TWIP.eam.fs --elements meta_TWIP -n 5 \\
      --a0 3.6 --auto-a0 --verify --surface gyroid --cells 20 --crush

  # carve TPMS geometries out of YOUR polycrystal (LAMMPS data file, types in --elements order)
  python3 cg_hea_tpms.py --pair adp --potential W_Mo_Nb_Zr_Ti_Ta.adp.alloy --elements Ti Zr Nb \\
      --lattice bcc --matrix polycrystal.data -n 1 --surface gyroid --density 0.4 --crush \\
      --crush-strain 0.03 --outdir gyroid_rho040

  # apparent Young's modulus of finished runs (fit after the start-up ringing, 0.75-2.5 %)
  python3 cg_hea_tpms.py --analyze run*/*_stress_strain.txt --fit-window 0.0075 0.025

  # re-run a configuration saved by the GUI
  python3 cg_hea_tpms.py --config cg_hea_output/gyroid_fcc_FeNiCoMn_cgn5_config.json
"""
import argparse, os, sys

import cg_hea_core as core


def build_parser():
    ap = argparse.ArgumentParser(description="CG-EAM/MEAM TPMS generator for FCC/BCC alloys "
                                             "(run without arguments for the GUI)")
    ap.add_argument("--gui", action="store_true", help="open the graphical interface")
    ap.add_argument("--config", help="load all settings from a JSON file saved by the GUI/CLI "
                                     "(other flags given on the command line override it)")
    ap.add_argument("--save-config", help="write the settings to this JSON file and exit")
    ap.add_argument("--pair", choices=["eam", "eamfs", "adp", "meam"],
                    help="eam = eam/alloy (setfl), eamfs = eam/fs, adp = angular-dependent "
                         "(setfl + u,w), meam = MEAM")
    ap.add_argument("--potential", help="eam/alloy, eam/fs or adp file (atomistic)")
    ap.add_argument("--library", help="MEAM library file (atomistic)")
    ap.add_argument("--params", help="MEAM parameter file (atomistic, optional)")
    ap.add_argument("--lib-elements", nargs="+",
                    help="MEAM: element list after the library in the ORIGINAL pair_coeff "
                         "(sets parameter-file indices), e.g. Co Ni Cr Fe Mn")
    ap.add_argument("--elements", nargs="+", help="atom types, e.g. Fe Ni Co Mn")
    ap.add_argument("--composition", nargs="+", type=float, default=None,
                    help="fraction (or at.%%) per element, normalised (default equiatomic)")
    ap.add_argument("--lattice", choices=sorted(core.LATTICES),
                    help="crystal structure of the alloy (default fcc)")
    ap.add_argument("-n", type=int, help="CG level: one bead = n^3 atoms (default 5)")
    ap.add_argument("--a0", type=float,
                    help="atomistic lattice constant (guess), Å; default: Vegard estimate "
                         "from the potential header")
    ap.add_argument("--auto-a0", action="store_true", default=None,
                    help="relax the random alloy with LAMMPS (E(a) scan + box/relax)")
    ap.add_argument("--verify", action="store_true", default=None,
                    help="run AA vs CG exact-mapping test")
    ap.add_argument("--check-pure", action="store_true", default=None,
                    help="relax each pure element as FCC and BCC (AA potential) and print a0, "
                         "E/atom: confirms each atom TYPE is the intended element. NOTE: neither "
                         "this nor --verify can detect a wrong MEAM --lib-elements order (that "
                         "only affects cross/screening terms) - take it from the potential's docs")
    ap.add_argument("--compare-phases", action="store_true", default=None,
                    help="relax the random alloy as FCC and as BCC and report E_BCC - E_FCC")
    ap.add_argument("--surface", choices=sorted(core.SURFACES))
    ap.add_argument("--cells", type=int, help="CG lattice cells per side (default 20)")
    ap.add_argument("--box-nm", type=float,
                    help="CG box edge in nm instead of --cells (rounded up to whole cells, "
                         "after a0 relaxation)")
    ap.add_argument("--periods", type=int, help="TPMS unit cells per side (default 1)")
    ap.add_argument("--phase", choices=["auto", "network", "sheet"],
                    help="auto (default): t>0 -> sheet |f|<=t, else network f<=0; "
                         "network: f<=t; sheet: |f|<=t")
    ap.add_argument("--t", type=float, help="TPMS level / sheet half-thickness (default 0)")
    ap.add_argument("--density", type=float,
                    help="target relative density (solid fraction, 0-1); sets t automatically")
    ap.add_argument("--min-coord", type=int,
                    help="remove beads with fewer first neighbours than this (0 = off)")
    ap.add_argument("--matrix",
                    help="carve the TPMS out of this LAMMPS data file (atom_style atomic, "
                         "orthogonal box), e.g. your polycrystal; its atom types must follow "
                         "--elements; it is scaled by n (use -n 1 to stay atomistic)")
    ap.add_argument("--analyze", nargs="+", metavar="STRESS_STRAIN",
                    help="fit the apparent Young's modulus of *_stress_strain.txt files and exit")
    ap.add_argument("--fit-window", nargs=2, type=float, default=(0.0075, 0.025),
                    metavar=("LO", "HI"),
                    help="strain window for --analyze (default 0.0075 0.025: after the start-up "
                         "ringing of the box; compress to >= 3 %%)")
    ap.add_argument("--temp", type=float, help="PHYSICAL temperature, K (default 300)")
    ap.add_argument("--dt-aa", type=float, help="atomistic timestep ps (CG uses n*dt)")
    ap.add_argument("--dt-cg", type=float,
                    help="override the CG timestep, ps (default n*dt-aa, the exact map)")
    ap.add_argument("--tdamp", type=float, help="CG thermostat damping, ps (default 100*n*dt-aa)")
    ap.add_argument("--pdamp", type=float, help="CG barostat damping, ps (default 1000*n*dt-aa)")
    ap.add_argument("--equil", type=int, help="NPT equilibration steps (default 20000)")
    ap.add_argument("--crush", action="store_true", default=None,
                    help="add uniaxial compression stage")
    ap.add_argument("--crush-strain", type=float)
    ap.add_argument("--strain-rate", type=float, help="1/ps")
    ap.add_argument("--crush-axis", choices=["x", "y", "z"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lmp", help="LAMMPS command, e.g. lmp or 'mpirun -np 4 lmp_mpi'")
    ap.add_argument("--outdir")
    return ap


def config_from_args(a):
    cfg = core.Config.from_json(a.config) if a.config else core.Config()
    for k, v in vars(a).items():
        if k in ("gui", "config", "save_config", "analyze", "fit_window") or v is None:
            continue
        setattr(cfg, k, v)
    return cfg


def analyze(paths, lo, hi):
    """Print the apparent modulus of each stress-strain file (with its relative density)."""
    import json, os
    print(f"{'run':55s} {'rel. density':>12s} {'E (GPa)':>9s} {'R^2':>6s} {'pts':>4s}")
    for p in paths:
        try:
            E, _, r2, k = core.fit_modulus(p, lo, hi)
        except (OSError, ValueError) as e:
            print(f"{p:55s}  ERROR {e}")
            continue
        summ = p.replace("_stress_strain.txt", "_summary.json")
        rho = json.load(open(summ)).get("fill") if os.path.isfile(summ) else None
        print(f"{p[-55:]:55s} {rho if rho is None else round(rho, 4)!s:>12s} {E:9.2f} {r2:6.3f} {k:4d}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.gui or not argv:
        import cg_hea_gui
        return cg_hea_gui.main()
    if a.analyze:
        return analyze(a.analyze, *a.fit_window)
    cfg = config_from_args(a)
    if not cfg.elements:
        ap.error("--elements is required")
    if cfg.pair in core.SETFL_PAIRS and not cfg.potential:
        ap.error("--potential is required for EAM / ADP")
    if cfg.pair == "meam" and not cfg.library:
        ap.error("--library is required for MEAM")
    try:
        cfg.validate()
    except ValueError as e:
        ap.error(str(e))
    if a.save_config:
        cfg.to_json(a.save_config)
        print(f"settings written to {a.save_config}")
        return 0
    try:
        core.run_pipeline(cfg)
    except core.VerificationError as e:
        print(f"VERIFICATION FAILED: {e}", file=sys.stderr)
        return 2
    except (core.PotentialError, core.LammpsError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
