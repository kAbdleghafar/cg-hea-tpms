"""
Tests for cg_hea_core. Run from the repository folder:

    python3 -m unittest discover -s tests -v        (or: python3 -m pytest tests)

Unit tests need only numpy. The integration tests (exact CG mapping in FCC and BCC,
EAM, ADP and MEAM including erose_form = 1) run LAMMPS and are skipped when no `lmp` executable is found (set CG_HEA_LMP to
point at one).
"""
import json, os, shutil, sys, tempfile, unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cg_hea_core as core

LMP = os.environ.get("CG_HEA_LMP") or core.find_lammps()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def read(path):
    with open(path) as f:
        return f.read()


def synthetic_setfl(path, names=("A", "B"), fs=False, Nr=500, Nrho=500, cut=5.5, adp=False):
    """Smooth, physically shaped EAM tables (not a fitted potential) in setfl format;
    adp=True appends dipole u(r) and quadrupole w(r) tables (ADP format)."""
    nel = len(names)
    r = np.linspace(0, cut, Nr)
    dr = r[1]
    rho_grid = np.linspace(0, 40, Nrho)
    drho = rho_grid[1]
    sw = np.where(r < cut, (1 - r / cut) ** 3, 0.0)                  # smooth cutoff
    re = [2.50 + 0.12 * k for k in range(nel)]
    with open(path, "w") as f:
        f.write("synthetic test potential\nnot for production\n\n")
        f.write(f"{nel} {' '.join(names)}\n{Nrho} {drho:.10e} {Nr} {dr:.10e} {cut:.10e}\n")
        for k, nm in enumerate(names):
            f.write(f"{k+1} {50.0 + 20*k:.4f} {re[k]*np.sqrt(2):.4f} fcc\n")
            F = -(1.5 + 0.3 * k) * np.sqrt(rho_grid)
            f.write("\n".join(f"{v:.12e}" for v in F) + "\n")
            for j in range(nel if fs else 1):
                rho = 3.0 * np.exp(-1.6 * (r / re[k] - 1)) * sw * (1 + 0.05 * j)
                f.write("\n".join(f"{v:.12e}" for v in rho) + "\n")
        for i in range(nel):
            for j in range(i + 1):
                rr = 0.5 * (re[i] + re[j])
                rs = np.maximum(r, 0.5)
                phi = (0.6 * np.exp(-6.0 * (rs / rr - 1)) - 1.1 * np.exp(-3.0 * (rs / rr - 1))) * sw
                f.write("\n".join(f"{v:.12e}" for v in rs * phi) + "\n")
        if adp:
            for amp in (0.05, 0.02):                                   # u, then w
                for i in range(nel):
                    for j in range(i + 1):
                        tab = amp * (1 + 0.1 * (i + j)) * np.exp(-r / 1.5) * sw
                        f.write("\n".join(f"{v:.12e}" for v in tab) + "\n")


MEAM_LIB = """# test library (Ni, W entries of the LAMMPS library.meam)
'Ni'        'fcc'   12.     28           58.71
4.99        2.45    2.20    6.00         2.20  3.52    4.45    0.94
1.0         3.57            1.60         3.70          1.0    1
'W'         'bcc'   8.      74           183.85
5.62777409  3.978   1.00    1.00065894   1.00  3.165   8.66    0.98
1.0         3.16          8.25         -2.7          1.      0
"""


class TestLattices(unittest.TestCase):
    def test_counts_and_nearest_neighbours(self):
        for kind, per, nn in (("fcc", 4, 1 / np.sqrt(2)), ("bcc", 2, np.sqrt(3) / 2)):
            a, nc = 3.3, 3
            pos = core.lattice(kind, a, nc)
            self.assertEqual(len(pos), per * nc**3)
            L = nc * a
            d = pos[1:] - pos[0]
            d -= L * np.round(d / L)
            dist = np.linalg.norm(d, axis=1)
            self.assertAlmostEqual(dist.min(), nn * a, places=10)
            self.assertEqual(int(np.sum(np.isclose(dist, nn * a))), len(core.NN_OFFSETS[kind]))

    def test_original_fcc_order_preserved(self):
        base = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        g = np.stack(np.meshgrid(*[np.arange(3)]*3, indexing="ij"), -1).reshape(-1, 3)
        ref = ((g[:, None, :] + base[None]).reshape(-1, 3) * 3.6)
        np.testing.assert_allclose(core.fcc(3.6, 3), ref)

    def test_vegard_volume_conversion(self):
        ref = {"X": ("fcc", 3.6), "Y": ("bcc", 2.9)}
        self.assertAlmostEqual(core.vegard_a0(ref, ["X"], [1], "fcc"), 3.6)
        self.assertAlmostEqual(core.vegard_a0(ref, ["X"], [1], "bcc"), (2 * 3.6**3 / 4) ** (1/3))
        v = 0.5 * 3.6**3 / 4 + 0.5 * 2.9**3 / 2
        self.assertAlmostEqual(core.vegard_a0(ref, ["X", "Y"], [1, 1], "bcc"), (2 * v) ** (1/3))
        self.assertIsNone(core.vegard_a0({"Z": ("b1", 4.0)}, ["Z"], [1], "fcc"))


class TestChemistry(unittest.TestCase):
    def test_exact_counts(self):
        rng = np.random.default_rng(0)
        for N, comp in ((1000, [0.29, 0.71]), (10, [0.4, 0.35, 0.25]), (32001, [0.2] * 5)):
            t = core.assign_species(N, comp, rng)
            counts = np.bincount(t, minlength=len(comp) + 1)[1:]
            self.assertEqual(counts.sum(), N)
            self.assertTrue(np.all(np.abs(counts - np.array(comp) / sum(comp) * N) < 1))

    def test_entropy_classes(self):
        self.assertIn("HEA", core.entropy_class([1] * 5)[1])
        self.assertIn("MEA", core.entropy_class([1] * 3)[1])
        self.assertIn("low", core.entropy_class([1, 1])[1])
        self.assertEqual(core.entropy_class([0.3, 0.3, 0.3, 0.1])[2], 4)


class TestEAMRescaling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _roundtrip(self, fs):
        src = os.path.join(self.tmp, "ab.eam.fs" if fs else "ab.eam.alloy")
        synthetic_setfl(src, fs=fs)
        info = core.load_potential("eamfs" if fs else "eam", potential=src)
        n = 3
        pair = core.rescale_potential(info, n, self.tmp, ["A", "B"])
        d0 = core.read_setfl(src, fs=fs)
        d1 = core.read_setfl(os.path.join(self.tmp, pair.cg_files[0]), fs=fs)
        self.assertAlmostEqual(d1["dr"], n * d0["dr"], places=12)
        self.assertAlmostEqual(d1["cut"], n * d0["cut"], places=12)
        self.assertEqual(d1["drho"], d0["drho"])
        for e0, e1 in zip(d0["E"], d1["E"]):
            self.assertAlmostEqual(e1["mass"], n**3 * e0["mass"], places=6)
            np.testing.assert_allclose(e1["F"], n**3 * e0["F"], rtol=1e-14)
            for r0, r1 in zip(e0["rho"], e1["rho"]):
                np.testing.assert_allclose(r1, r0, rtol=1e-14)
        for z0, z1 in zip(d0["z2r"], d1["z2r"]):
            np.testing.assert_allclose(z1, n**4 * z0, rtol=1e-14)
        self.assertEqual(pair.coeff("cg", ["B", "A"]), f"pair_coeff * * {pair.cg_files[0]} B A")

    def test_alloy(self):
        self._roundtrip(False)

    def test_adp(self):
        src = os.path.join(self.tmp, "ab.adp")
        synthetic_setfl(src, adp=True)
        self.assertEqual(core.detect_pair_style(src), "adp")
        info = core.load_potential("adp", potential=src)
        n = 4
        pair = core.rescale_potential(info, n, self.tmp, ["A", "B"])
        self.assertEqual(pair.style, "pair_style adp")
        d0 = core.read_setfl(src)
        d1 = core.read_setfl(os.path.join(self.tmp, pair.cg_files[0]))
        self.assertEqual(d1["kind"], "adp")
        for a, b in zip(d0["u"], d1["u"]):
            np.testing.assert_allclose(b, np.sqrt(n) * a, rtol=1e-14)
        for a, b in zip(d0["w"], d1["w"]):
            np.testing.assert_allclose(b, a / np.sqrt(n), rtol=1e-14)
        with self.assertRaises(core.PotentialError):
            core.load_potential("eam", potential=src)          # ADP given as eam/alloy

    def test_fortran_exponents(self):
        src = os.path.join(self.tmp, "d.eam.alloy")
        synthetic_setfl(src)
        write(src, read(src).replace("e-0", "D-0"))
        d = core.read_setfl(src)
        self.assertTrue(np.all(np.isfinite(d["E"][0]["F"])))

    def test_fs(self):
        self._roundtrip(True)

    def test_flavour_detection_and_errors(self):
        src = os.path.join(self.tmp, "x.setfl")
        synthetic_setfl(src, fs=True)
        self.assertEqual(core.detect_pair_style(src), "eamfs")
        with self.assertRaises(core.PotentialError):
            core.load_potential("eam", potential=src)          # fs file given as eam/alloy
        bad = os.path.join(self.tmp, "bad.eam.alloy")
        write(bad, "not\na\npotential\n")
        with self.assertRaises(core.PotentialError):
            core.load_potential("eam", potential=bad)


class TestMEAMRescaling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lib = os.path.join(self.tmp, "lib.meam")
        write(self.lib, MEAM_LIB)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_library_and_params(self):
        par = os.path.join(self.tmp, "NiW.meam")
        write(par, "# no rc here: LAMMPS default 4.0 must be rescaled too\n"
                             "Ec(1,2) = 6.0\nre(1,2) = 2.6\ndelta(1,2) = 0.1\n"
                             "alpha(1,2) = 5.0\nzbl(1,1) = 1\nnn2(1,2) = 1\n")
        info = core.load_potential("meam", library=self.lib, params=par)
        self.assertEqual(info.elements, ["Ni", "W"])
        self.assertEqual(info.param_index_max, 2)
        self.assertEqual(core.classify_meam_file(par), "params")
        self.assertEqual(core.classify_meam_file(self.lib), "library")
        n = 4
        pair = core.rescale_potential(info, n, self.tmp, ["W", "Ni"])
        self.assertEqual(pair.lib_elements, ["Ni", "W"])        # library order (dedicated library)
        lib1 = core.read_meam_library(os.path.join(self.tmp, pair.cg_files[0]))
        for e0, e1 in zip(core.read_meam_library(self.lib), lib1):
            self.assertAlmostEqual(float(e1["atwt"]), n**3 * float(e0["atwt"]))
            self.assertAlmostEqual(float(e1["esub"]), n**3 * float(e0["esub"]))
            self.assertAlmostEqual(float(e1["alat"]), n * float(e0["alat"]))
            self.assertEqual(e1["alpha"], e0["alpha"])
        txt = read(os.path.join(self.tmp, pair.cg_files[1]))
        for line in ("Ec(1,2) = 384.0", "re(1,2) = 10.4", "delta(1,2) = 6.4", "alpha(1,2) = 5.0",
                     "rc = 16.0", "delr = 0.4", "zbl(1,1) = 0", "zbl(1,2) = 0", "zbl(2,2) = 0"):
            self.assertIn(line, txt)
        self.assertNotIn("zbl(1,1) = 1", txt)
        aa0 = read(os.path.join(self.tmp, pair.aa0_files[1]))
        self.assertIn("Ec(1,2) = 6.0", aa0)
        self.assertIn("zbl(2,2) = 0", aa0)
        self.assertEqual(pair.coeff("aa", ["W"]), "pair_coeff * * lib.meam Ni W NiW.meam W")

    def test_no_parameter_file(self):
        info = core.load_potential("meam", library=self.lib)
        pair = core.rescale_potential(info, 2, self.tmp, ["Ni"], ["Ni", "W"])
        self.assertEqual(pair.aa_files[1], "NULL")
        txt = read(os.path.join(self.tmp, pair.cg_files[1]))
        self.assertIn("rc = 8.0", txt)
        self.assertIn("zbl(1,2) = 0", txt)

    def test_erose_form_1_repuls_is_a_length(self):
        for form, repuls in ((0, 0.3), (1, 0.9), (2, 0.3)):      # n = 3
            par = os.path.join(self.tmp, f"f{form}.meam")
            write(par, f"erose_form = {form}\nrepuls(1,1) = 0.3\nattrac(1,1) = 0.1\n")
            info = core.load_potential("meam", library=self.lib, params=par)
            self.assertEqual(info.erose_form, form)
            pair = core.rescale_potential(info, 3, self.tmp, ["Ni"], ["Ni", "W"])
            cg = {k + idx: core._num(v) for k, idx, v, _ in
                  core.parse_meam_params(os.path.join(self.tmp, pair.cg_files[1]))}
            self.assertAlmostEqual(cg["repuls(1,1)"], repuls)
            self.assertAlmostEqual(cg["attrac(1,1)"], 0.1)

    def test_strict_parameter_parsing(self):
        for bad in ("foo(1,1) = 2.0\n", "Ec(1,1) 4.0\n", "Ec(0,1) = 4.0\n"):
            par = os.path.join(self.tmp, "bad.meam")
            write(par, bad)
            with self.assertRaises(core.PotentialError):
                core.load_potential("meam", library=self.lib, params=par)

    def test_zbl_detection(self):
        par = os.path.join(self.tmp, "z.meam")
        write(par, "zbl(1,1) = 0\nzbl(1,2) = 0\n")
        e = core.parse_meam_params(par)
        self.assertEqual(core.meam_zbl_active(e, 2), [(2, 2)])  # default is on
        write(par, "zbl(1,1) = 0\nzbl(1,2) = 0\nzbl(2,2) = 0\n")
        self.assertEqual(core.meam_zbl_active(core.parse_meam_params(par), 2), [])

    def test_index_beyond_element_list(self):
        par = os.path.join(self.tmp, "p.meam")
        write(par, "Ec(1,3) = 5.0\n")
        info = core.load_potential("meam", library=self.lib, params=par)
        with self.assertRaises(core.PotentialError):
            core.rescale_potential(info, 2, self.tmp, ["Ni"], ["Ni", "W"])


class TestGeometry(unittest.TestCase):
    def test_density_target(self):
        for kind in ("fcc", "bcc"):
            for phase in ("network", "sheet"):
                g = core.build_geometry(kind, 10.0, 12, "gyroid", 1, phase, density=0.3)
                self.assertAlmostEqual(g.fill, 0.3, delta=0.005)
                self.assertEqual(g.phase, phase)

    def test_auto_phase_matches_original(self):
        g = core.build_geometry("fcc", 10.0, 10, "gyroid", 1, "auto", 0.0)
        self.assertEqual((g.phase, g.level), ("network", 0.0))
        g = core.build_geometry("fcc", 10.0, 10, "gyroid", 1, "auto", 0.3)
        self.assertEqual(g.phase, "sheet")

    def test_prune_removes_isolated_beads(self):
        for kind in ("fcc", "bcc"):
            idx = core.lattice_index(kind, 4)              # half-lattice indices 0..7
            keep = idx[:, 0] < 4                           # solid slab x < L/2
            j = int(np.flatnonzero(idx[:, 0] == 6)[0])     # a bead in the middle of the gap
            keep[j] = True
            c = core.coordination(idx, keep, kind, 4)
            self.assertEqual(c[j], 0)
            k2, removed = core.prune(idx, keep, kind, 4, 3)
            self.assertFalse(k2[j])
            self.assertEqual(removed, 1)                   # the slab itself is untouched

    def test_full_lattice_coordination(self):
        for kind in ("fcc", "bcc"):
            idx = core.lattice_index(kind, 3)
            c = core.coordination(idx, np.ones(len(idx), bool), kind, 3)
            self.assertTrue(np.all(c == len(core.NN_OFFSETS[kind])))


class TestMatrix(unittest.TestCase):
    """Carving TPMS geometries out of an existing structure (e.g. the user's polycrystal)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        pos = core.lattice("bcc", 3.3, 6) + 0.05 * rng.normal(size=(432, 3))
        self.types = core.assign_species(432, [0.5, 0.25, 0.25], rng)
        L, lo = 6 * 3.3, -2.0
        lines = ["LAMMPS data file via write_data", "", "432 atoms", "3 atom types", ""]
        lines += [f"{lo} {lo + L} {ax}lo {ax}hi" for ax in "xyz"]
        lines += ["", "Masses", "", "1 47.867", "2 91.224", "3 92.906", "", "Atoms # atomic", ""]
        order = rng.permutation(432)                         # unsorted ids, image flags
        for i in order:
            x, y, z = pos[i] + lo + np.array([L, 0, -L]) * (i % 2)
            lines.append(f"{i+1} {self.types[i]} {x:.6f} {y:.6f} {z:.6f} {i % 2} 0 {-(i % 2)}")
        lines += ["", "Velocities", ""] + [f"{i+1} 0.1 0.0 0.0" for i in range(432)]
        self.path = os.path.join(self.tmp, "poly.data")
        write(self.path, "\n".join(lines) + "\n")
        self.L = L

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_reader(self):
        m = core.read_lammps_data(self.path)
        self.assertEqual(len(m["pos"]), 432)
        np.testing.assert_array_equal(m["types"], self.types)       # sorted back by id
        np.testing.assert_allclose(m["box"], self.L)
        self.assertTrue(np.all((m["pos"] >= 0) & (m["pos"] < self.L)))
        self.assertEqual(m["masses"][2], 91.224)

    def test_reader_refuses_unsupported(self):
        txt = read(self.path)
        bad = os.path.join(self.tmp, "bad.data")
        write(bad, txt.replace("Atoms # atomic", "Atoms # sphere"))
        with self.assertRaises(ValueError):
            core.read_lammps_data(bad)
        write(bad, txt.replace("3 atom types\n", "3 atom types\n0.5 0.0 0.0 xy xz yz\n"))
        with self.assertRaises(ValueError):
            core.read_lammps_data(bad)

    def test_carve_density(self):
        m = core.read_lammps_data(self.path)
        for phase in ("network", "sheet"):
            keep, ph, lev, pk = core.carve_matrix(m, "gyroid", 1, phase, 0.0, 0.4)
            self.assertAlmostEqual(keep.mean(), 0.4, delta=0.01)
            self.assertEqual(ph, phase)

    def test_charge_style_random_species_noncubic(self):
        """OVITO/Atomsk-style template: atom_style charge, pure element, 3:3:1 box."""
        pos = core.lattice("bcc", 3.3, 6)
        pos = np.vstack([pos + [dx, dy, 0] for dx in (0, 19.8, 39.6) for dy in (0, 19.8, 39.6)])
        lines = ["# LAMMPS data file written by OVITO", "", f"{len(pos)} atoms", "3 atom types", "",
                 "-29.7 29.7 xlo xhi", "-29.7 29.7 ylo yhi", "-9.9 9.9 zlo zhi", "", "Masses", "",
                 "1 47.867  # Ti", "2 92.906  # Nb", "3 95.95  # Mo", "", "Atoms  # charge", ""]
        lines += [f"{i+1} 1 0.0 {x-29.7:.6f} {y-29.7:.6f} {z-9.9:.6f}" for i, (x, y, z) in enumerate(pos)]
        path = os.path.join(self.tmp, "tpl.data")
        write(path, "\n".join(lines) + "\n")
        m = core.read_lammps_data(path)
        self.assertEqual(m["style"], "charge")
        np.testing.assert_allclose(m["box"], [59.4, 59.4, 19.8])
        keep, ph, lev, pk = core.carve_matrix(m, "gyroid", 1, "network", 0.0, 0.5)
        self.assertEqual(pk.tolist(), [3, 3, 1])                     # cubic 19.8 A cells
        pot = os.path.join(self.tmp, "ABC.eam.alloy")
        synthetic_setfl(pot, names=("A", "B", "C"))
        cfg = core.Config(pair="eam", potential=pot, elements=["A", "B", "C"], n=1, a0=3.3,
                          matrix=path, matrix_species="random", composition=[0.5, 0.3, 0.2],
                          density=1.0, outdir=os.path.join(self.tmp, "o"), seed=5)
        s = core.run_pipeline(cfg, core.Session(log=lambda *_: None))
        self.assertEqual(s["n_beads"], len(pos))
        self.assertEqual([s["counts"][e] for e in "ABC"], [1944, 1166, 778])   # exact counts
        cfg.density, cfg.outdir = 0.4, os.path.join(self.tmp, "o2")
        s2 = core.run_pipeline(cfg, core.Session(log=lambda *_: None))
        self.assertEqual(s2["tpms_periods"], [3, 3, 1])
        self.assertAlmostEqual(s2["fill"], 0.4, delta=0.01)

    def test_fit_modulus(self):
        eps = np.linspace(0, 0.02, 41)
        sig = 25.0 * eps + 0.01 + 0.001 * np.sin(300 * eps)
        f = os.path.join(self.tmp, "x_stress_strain.txt")
        write(f, "# fix ave/time\n# TimeStep v_strain v_stress\n" +
              "".join(f"{i} {e} {s}\n" for i, (e, s) in enumerate(zip(eps, sig))))
        E, c, r2, k = core.fit_modulus(f, 0.002, 0.01)
        self.assertAlmostEqual(E, 25.0, delta=0.3)
        self.assertGreater(r2, 0.99)
        self.assertEqual(k, 17)

    @unittest.skipUnless(LMP, "LAMMPS executable not found")
    def test_pipeline_with_matrix(self):
        pot = os.path.join(self.tmp, "ABC.eam.alloy")
        synthetic_setfl(pot, names=("A", "B", "C"))
        cfg = core.Config(pair="eam", potential=pot, elements=["A", "B", "C"], lattice="bcc",
                          n=2, a0=3.3, matrix=self.path, surface="gyroid", density=0.5,
                          lmp=LMP, outdir=os.path.join(self.tmp, "out"), crush=True,
                          crush_strain=0.03)
        s = core.run_pipeline(cfg, core.Session(log=lambda *_: None))
        self.assertAlmostEqual(s["fill"], 0.5, delta=0.01)
        np.testing.assert_allclose(s["box_vector"], 2 * self.L)
        self.assertEqual(sum(s["counts"].values()), s["n_beads"])
        self.assertAlmostEqual(s["composition"][0], 0.5, delta=1e-9)  # taken from the file
        out = read(os.path.join(s["outdir"], s["prefix"] + ".data"))
        self.assertIn(f"0.0 {2*self.L:.8f} xlo xhi", out)


class TestGeometrySweep(unittest.TestCase):
    """examples/geometry_sweep.py: generate from a matrix, then summarise fake results."""

    def test_generate_and_summarize(self):
        import subprocess
        tmp = tempfile.mkdtemp()
        try:
            pot = os.path.join(tmp, "AB.eam.alloy")
            synthetic_setfl(pot)
            t = core.assign_species(2 * 8**3, [0.5, 0.5], np.random.default_rng(0))
            mat = os.path.join(tmp, "m.data")
            core.write_data(mat, core.lattice("bcc", 3.3, 8), t, 8 * 3.3, [50.0, 70.0], "m")
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples",
                                  "geometry_sweep.py")
            root = os.path.join(tmp, "sw")
            def sweep(root, periods, stiffer):
                out = subprocess.run([sys.executable, script, "generate", "--pair", "eam",
                                      "--potential", pot, "--elements", "A", "B", "--matrix", mat,
                                      "--surfaces", "gyroid", "--sheet", "--densities", "0.3",
                                      "0.5", "--seeds", "1", "--periods", str(periods),
                                      "--outroot", root], capture_output=True, text=True)
                self.assertEqual(out.returncode, 0, out.stderr)
                runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
                self.assertEqual(runs, ["bulk_s1", "gyroid_network_rho030_s1",
                                        "gyroid_network_rho050_s1"])
                self.assertEqual(read(os.path.join(root, "run_all.sh")).count("$LMP -in"), 3)
                for d in runs:
                    s = json.loads(read(os.path.join(root, d, [f for f in os.listdir(
                        os.path.join(root, d)) if f.endswith("_summary.json")][0])))
                    self.assertEqual(s["tpms_periods"], [periods] * 3)
                    E = 50.0 if d.startswith("bulk") else stiffer * 50.0 * 0.6 * s["fill"]**2
                    eps = np.linspace(0, 0.03, 61)                            # C 0.6, m 2
                    write(os.path.join(root, d, s["prefix"] + "_stress_strain.txt"),
                          "# t\n# TimeStep v_strain v_stress\n" +
                          "".join(f"{i} {e} {E * e}\n" for i, e in enumerate(eps)))

            sweep(root, 1, 1.0)
            out = subprocess.run([sys.executable, script, "summarize", "--outroot", root,
                                  "--target", "5"], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("C = 0.600  m = 2.00", out.stdout)
            self.assertIn("rho = 0.408 for E* = 5 GPa", out.stdout)   # (5/30)^(1/2)
            self.assertNotIn("Size effect", out.stdout)
            # a second sweep with half the TPMS cell, 10 % stiffer lattices: size-effect table
            root2 = os.path.join(tmp, "sw2")
            sweep(root2, 2, 1.1)
            out = subprocess.run([sys.executable, script, "summarize", "--outroot", root, root2],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("2.64 nm   1.32 nm", out.stdout)
            # same fill (0.30 in both) -> 0.1 / 1.05 = 9.5 %
            self.assertRegex(out.stdout, r"gyroid_network\s+0\.30\s+0\.055\s+0\.060\s+9\.5 %")
        finally:
            shutil.rmtree(tmp)


class TestSweepSeparateBulk(unittest.TestCase):
    """A bulk-only sweep plus a lattice sweep with --no-bulk, joined by summarize --bulk-root."""

    def test_bulk_root(self):
        import subprocess
        tmp = tempfile.mkdtemp()
        try:
            pot = os.path.join(tmp, "AB.eam.alloy")
            synthetic_setfl(pot)
            t = core.assign_species(2 * 8**3, [0.5, 0.5], np.random.default_rng(0))
            mat = os.path.join(tmp, "m.data")
            core.write_data(mat, core.lattice("bcc", 3.3, 8), t, 8 * 3.3, [50.0, 70.0], "m")
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples",
                                  "geometry_sweep.py")
            base = [sys.executable, script, "generate", "--pair", "eam", "--potential", pot,
                    "--elements", "A", "B", "--matrix", mat, "--seeds", "1", "--load",
                    "compression", "--strain", "0.05", "--rate", "5e-5"]
            bulk, lat = os.path.join(tmp, "bulk"), os.path.join(tmp, "lat")
            out = subprocess.run(base + ["--surfaces", "--sheet", "--outroot", bulk],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual([d for d in os.listdir(bulk) if os.path.isdir(os.path.join(bulk, d))],
                             ["bulk_s1"])
            out = subprocess.run(base + ["--surfaces", "gyroid", "--sheet", "--densities", "0.4",
                                         "0.6", "--no-bulk", "--outroot", lat],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            runs = sorted(d for d in os.listdir(lat) if os.path.isdir(os.path.join(lat, d)))
            self.assertEqual(runs, ["gyroid_network_rho040_s1", "gyroid_network_rho060_s1"])
            for root, d in [(bulk, "bulk_s1")] + [(lat, r) for r in runs]:
                s = json.loads(read(os.path.join(root, d, [f for f in os.listdir(
                    os.path.join(root, d)) if f.endswith("_summary.json")][0])))
                self.assertEqual(s["compression_steps"], round(0.05 / (5e-5 * 0.001)))
                E = 60.0 if d.startswith("bulk") else 60.0 * 0.5 * s["fill"]**2   # C 0.5, m 2
                eps = np.linspace(0, 0.05, 101)
                write(os.path.join(root, d, s["prefix"] + "_stress_strain.txt"),
                      "# t\n# TimeStep v_strain v_stress\n" +
                      "".join(f"{i} {e} {E * e}\n" for i, e in enumerate(eps)))
            out = subprocess.run([sys.executable, script, "summarize", "--outroot", lat,
                                  "--bulk-root", bulk, "--fit-window", "0", "0.01"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("E_bulk = 60.00 GPa from", out.stdout)
            self.assertIn("C = 0.500  m = 2.00", out.stdout)
            out = subprocess.run(base + ["--surfaces", "--sheet", "--no-bulk", "--outroot",
                                         os.path.join(tmp, "none")], capture_output=True, text=True)
            self.assertNotEqual(out.returncode, 0)
        finally:
            shutil.rmtree(tmp)


@unittest.skipUnless(__import__("importlib").util.find_spec("scipy"), "needs SciPy")
class TestRemoveDoubles(unittest.TestCase):
    """examples/remove_doubles.py: Atomsk's -rmd rule (earlier atom stays, periodic box)."""

    def test_doubles_removed(self):
        import subprocess
        tmp = tempfile.mkdtemp()
        try:
            a, nc = 3.3, 4
            lat = core.lattice("bcc", a, nc)
            L = nc * a
            extra = lat[[0, 5, 9]] + [0.3, 0.0, 0.0]               # doubles after the originals
            edge = np.array([[L - 0.2, lat[0, 1], lat[0, 2]]])        # double of atom 0 across x
            pos = np.vstack([lat, extra, edge])
            src, dst = os.path.join(tmp, "in.lmp"), os.path.join(tmp, "out.lmp")
            core.write_data(src, pos, np.ones(len(pos), int), L, [47.867], "doubles")
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples",
                                  "remove_doubles.py")
            out = subprocess.run([sys.executable, script, src, dst, "2.0"], capture_output=True,
                                 text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("removed 4, kept 128", out.stdout)
            m = core.read_lammps_data(dst)
            self.assertEqual(len(m["pos"]), len(lat))
            np.testing.assert_allclose(m["pos"], lat, atol=1e-9)
        finally:
            shutil.rmtree(tmp)


class TestConfig(unittest.TestCase):
    def test_json_and_cli(self):
        cfg = core.Config(pair="eam", potential="x.eam.alloy", elements=["Fe", "Ni"],
                          composition=[0.6, 0.4], lattice="bcc", density=0.3, crush=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            cfg.to_json(p)
            self.assertEqual(core.Config.from_json(p), cfg)
        cli = cfg.to_cli()
        for frag in ("--lattice bcc", "--density 0.3", "--crush", "--composition 0.6 0.4"):
            self.assertIn(frag, cli)

    def test_tension_stage(self):
        tmp = tempfile.mkdtemp()
        try:
            pot = os.path.join(tmp, "AB.eam.alloy")
            synthetic_setfl(pot)
            cfg = core.Config(pair="eam", potential=pot, elements=["A", "B"], a0=3.6, n=2,
                              cells=4, crush=True, load="tension", crush_axis="x",
                              crush_strain=0.01, strain_rate=5e-5, outdir=tmp)
            s = core.run_pipeline(cfg, core.Session(log=lambda *_: None))
            run = read(os.path.join(tmp, s["run_input"]))
            self.assertIn("variable       strain equal (lx-v_l0)/v_l0", run)
            self.assertIn("variable       stress equal -c_prs[1]*1.0e-4", run)
            self.assertIn("deform 1 x erate 5e-05 remap", run)
            self.assertIn("y 0 0", run)
            self.assertEqual(s["compression_steps"], round(0.01 / (5e-5 * 2 * 0.001)))
            self.assertIn("dump_modify    1 every 10000 ", run)       # 100,000 steps / 10
            self.assertIn("fix            br all box/relax aniso 0.0", run)
            self.assertIn("--load tension", cfg.to_cli())
        finally:
            shutil.rmtree(tmp)

    def test_validation(self):
        with self.assertRaises(ValueError):
            core.Config(elements=["A"], lattice="hcp").validate()
        with self.assertRaises(ValueError):
            core.Config(elements=["A"], phase="sheet", t=0).validate()
        with self.assertRaises(ValueError):
            core.Config(elements=["A"], seed=0).validate()
        with self.assertRaises(ValueError):
            core.Config(elements=["A"], crush=True, crush_strain=1.2).validate()
        with self.assertRaises(ValueError):
            core.Config(elements=["A"], box_nm=-3).validate()


@unittest.skipUnless(LMP, "LAMMPS executable not found")
class TestPipelineLAMMPS(unittest.TestCase):
    """Exact-mapping verification through the full pipeline."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pot = os.path.join(self.tmp, "AB.eam.alloy")
        synthetic_setfl(self.pot)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, kind, pair="eam", pot=None, **kw):
        cfg = core.Config(pair=pair, potential=pot or self.pot, elements=["A", "B"],
                          lattice=kind, n=3, auto_a0=True, verify=True, cells=6, lmp=LMP,
                          outdir=os.path.join(self.tmp, f"{kind}_{pair}"), crush=True)
        for k, v in kw.items():
            setattr(cfg, k, v)
        out = []
        s = core.run_pipeline(cfg, core.Session(log=out.append))
        self.assertTrue(s["verify"]["passed"], "\n".join(out))
        self.assertLess(s["verify"]["static_dev"], 1e-10)
        self.assertTrue(s["n_beads"] > 0)
        run = read(os.path.join(s["outdir"], s["run_input"]))
        self.assertIn(f"temp {27*300:.1f}", run)
        self.assertIn("equal c_prs[3]*1.0e-4", run)          # compression positive
        self.assertNotIn("-c_prs", run)
        self.assertIn("write_restart", run)
        summ = json.loads(read(os.path.join(s["outdir"], s["prefix"] + "_summary.json")))
        self.assertEqual(summ["lattice"], kind)
        self.assertEqual(summ["type_to_element"], {"1": "A", "2": "B"})
        return s

    def test_fcc(self):
        self._run("fcc")

    def test_bcc_box_nm(self):
        s = self._run("bcc", box_nm=9.0)
        self.assertGreaterEqual(s["box"], 90.0)
        self.assertLess(s["box"] - s["a_cg"], 90.0)

    def test_adp(self):
        pot = os.path.join(self.tmp, "AB.adp")
        synthetic_setfl(pot, adp=True)
        self._run("fcc", pair="adp", pot=pot)

    def test_meam_erose_form_1(self):
        lib = os.path.join(self.tmp, "lib.meam")
        par = os.path.join(self.tmp, "NiW.meam")
        write(lib, MEAM_LIB)
        write(par, "erose_form = 1\nrc = 4.5\nrepuls(1,1) = 0.3\nattrac(1,1) = 0.05\n"
                   "repuls(1,2) = 0.2\nzbl(1,1) = 0\nzbl(1,2) = 0\nzbl(2,2) = 0\n")
        cfg = core.Config(pair="meam", library=lib, params=par, lib_elements=["Ni", "W"],
                          elements=["Ni"], lattice="fcc", n=3, auto_a0=True, verify=True,
                          cells=4, lmp=LMP, outdir=os.path.join(self.tmp, "meam"))
        out = []
        s = core.run_pipeline(cfg, core.Session(log=out.append))
        self.assertTrue(s["verify"]["passed"], "\n".join(out))
        self.assertIsNone(s["verify"]["zbl_effect"])         # ZBL already off in the original

    def test_wrong_rescaling_is_caught(self):
        good = core.write_setfl

        def broken(path, d, n):                               # embedding energy 0.1 % off
            d = dict(d, E=[dict(e, F=e["F"] * 1.001) for e in d["E"]])
            good(path, d, n)
        core.write_setfl = broken
        try:
            cfg = core.Config(pair="eam", potential=self.pot, elements=["A", "B"], n=3,
                              a0=3.6, verify=True, cells=4, lmp=LMP,
                              outdir=os.path.join(self.tmp, "broken"))
            with self.assertRaises(core.VerificationError):
                core.run_pipeline(cfg, core.Session(log=lambda *_: None))
            self.assertFalse(any(f.endswith("_run.in") for f in os.listdir(cfg.outdir)))
        finally:
            core.write_setfl = good


if __name__ == "__main__":
    unittest.main()
