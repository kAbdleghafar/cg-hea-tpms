"""
cg_hea_core.py — engine for coarse-grained TPMS cells of FCC and BCC multicomponent
alloys (high- and medium-entropy alloys). Shared by the CLI (cg_hea_tpms.py) and the
GUI (cg_hea_gui.py).

Pipeline:  rescale the attached atomistic potential (EAM/alloy, EAM/FS or MEAM)
           -> relax the random-alloy lattice constant in the chosen lattice (FCC or BCC)
           -> verify the CG mapping -> CG lattice + TPMS mask + random species
           -> LAMMPS data + run input (equilibration, optional compression)

Coarse-graining (Chalamet, Rodney, Shibuta, Comput. Mater. Sci. 228 (2023) 112306):
    E_CG(X) = n^3 * E_AA(X/n)      (EAM: phi,F *n^3, rho(r/n); MEAM: Ec,delta,atwt,esub *n^3;
                                     re,alat,rc,delr *n; ZBL off)
This map is EXACT and independent of the crystal structure: a CG run at T_CG = n^3*T with
timestep n*dt is the atomistic trajectory of the same structure shrunk by n, scaled up by n.
Stresses and elastic constants are unchanged. Hence:
  * thermostat at n^3 * T_physical (NOT T/n)
  * a CG cell of edge L behaves exactly like an atomistic cell of edge L/n
    -> size-independent properties transfer; size effects are those of the L/n cell
  * random species per bead == random alloy in the L/n atomistic cell
  * FCC and BCC differ only in the bead lattice (4 vs 2 beads per cubic cell) and in the
    lattice constant, which is relaxed with the ATOMISTIC potential and scaled by n.
"""
from __future__ import annotations

import itertools, json, math, os, re, shlex, shutil, subprocess, threading
from dataclasses import asdict, dataclass, field, fields

import numpy as np

TWO_PI = 2.0 * np.pi
SURFACES = {
    "gyroid":    lambda X, Y, Z: np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X),
    "diamond":   lambda X, Y, Z: (np.sin(X)*np.sin(Y)*np.sin(Z) + np.sin(X)*np.cos(Y)*np.cos(Z)
                                  + np.cos(X)*np.sin(Y)*np.cos(Z) + np.cos(X)*np.cos(Y)*np.sin(Z)),
    "primitive": lambda X, Y, Z: np.cos(X) + np.cos(Y) + np.cos(Z),
    "neovius":   lambda X, Y, Z: 3*(np.cos(X)+np.cos(Y)+np.cos(Z)) + 4*np.cos(X)*np.cos(Y)*np.cos(Z),
    "iwp":       lambda X, Y, Z: (2*(np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z) + np.cos(Z)*np.cos(X))
                                  - (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z))),
    "fischer":   lambda X, Y, Z: (np.cos(2*X)*np.sin(Y)*np.cos(Z) + np.cos(X)*np.cos(2*Y)*np.sin(Z)
                                  + np.sin(X)*np.cos(Y)*np.cos(2*Z)),
    "frd":       lambda X, Y, Z: (4*np.cos(X)*np.cos(Y)*np.cos(Z)
                                  - (np.cos(2*X)*np.cos(2*Y) + np.cos(2*Y)*np.cos(2*Z)
                                     + np.cos(2*Z)*np.cos(2*X))),
}

PAIR_STYLES = {"eam": "eam/alloy", "eamfs": "eam/fs", "adp": "adp", "meam": "meam"}
PAIR_LABELS = {"eam": "EAM  (eam/alloy, setfl)",
               "eamfs": "EAM Finnis-Sinclair  (eam/fs)",
               "adp": "ADP  (angular-dependent, setfl + u,w)",
               "meam": "MEAM  (library + parameter file)"}
SETFL_PAIRS = ("eam", "eamfs", "adp")
SETFL_EXT = {"eam": "eam.alloy", "eamfs": "eam.fs", "adp": "adp"}


class PotentialError(ValueError):
    """The attached potential file cannot be read or does not fit the request."""


class LammpsError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


class VerificationError(RuntimeError):
    """The CG potential does not reproduce the atomistic one: no model is written."""


# ───────────────────────── lattices ─────────────────────────
# basis in units of a/2, so every site has integer "half-lattice" indices
LATTICES = {
    "fcc": np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]),
    "bcc": np.array([[0, 0, 0], [1, 1, 1]]),
}
NN_OFFSETS = {                                   # first neighbours, in units of a/2
    "fcc": np.array([p for p in itertools.product((-1, 0, 1), repeat=3)
                     if sum(map(abs, p)) == 2]),                      # 12
    "bcc": np.array(list(itertools.product((-1, 1), repeat=3))),     # 8
}
ATOMS_PER_CELL = {"fcc": 4, "bcc": 2}
DEFAULT_A0 = {"fcc": 3.60, "bcc": 3.10}          # last-resort guesses, Å
BULK_CELLS = {"fcc": 6, "bcc": 8}                # 864 / 1024 atoms for the a0 relaxation
PURE_CELLS = {"fcc": 4, "bcc": 5}                # 256 / 250 atoms for the pure-element check
SCAN = np.round(np.arange(0.75, 1.3001, 0.01), 4)  # E(a) scan, factors of the a0 guess


def lattice_index(kind, nc):
    """Integer site indices (units of a/2) of an nc^3 cubic supercell of `kind`."""
    g = 2 * np.stack(np.meshgrid(*[np.arange(nc)]*3, indexing="ij"), -1).reshape(-1, 3)
    return (g[:, None, :] + LATTICES[kind][None]).reshape(-1, 3)


def lattice(kind, a, nc):
    return lattice_index(kind, nc) * (0.5 * a)


def fcc(a, nc):          # kept for scripts written against the original module
    return lattice("fcc", a, nc)


def bcc(a, nc):
    return lattice("bcc", a, nc)


def atomic_volume(lat, a):
    """Volume per atom of a reference structure (as written in potential headers)."""
    lat = str(lat).strip("'\"").lower()
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if a <= 0:
        return None
    return {"fcc": a**3 / 4, "bcc": a**3 / 2, "sc": a**3, "dia": a**3 / 8,
            "diamond": a**3 / 8, "hcp": a**3 / math.sqrt(2)}.get(lat)


def vegard_a0(ref, elements, comp, kind):
    """Lattice-constant guess for the random alloy in `kind` from the per-element
    reference structures in the potential header (Vegard's law on atomic volume).
    Header data are sometimes mislabelled, so this is only a starting guess."""
    vols = [atomic_volume(*ref.get(e, (None, None))) for e in elements]
    if any(v is None for v in vols):
        return None
    V = float(np.dot(np.asarray(comp) / np.sum(comp), vols))
    return (ATOMS_PER_CELL[kind] * V) ** (1.0 / 3.0)


# ───────────────────────── alloy chemistry ─────────────────────────
def assign_species(N, comp, rng):
    """Exact-count random assignment (largest-remainder rounding), types 1..len(comp)."""
    comp = np.asarray(comp, float)
    exact = comp / comp.sum() * N
    counts = np.floor(exact).astype(int)
    counts[np.argsort(-(exact - counts), kind="stable")[:N - counts.sum()]] += 1
    t = np.repeat(np.arange(1, len(comp) + 1), counts)
    rng.shuffle(t)
    return t


def entropy_class(comp):
    """Configurational entropy (units of R) and the usual HEA/MEA classification."""
    x = np.asarray(comp, float)
    x = x[x > 0] / x.sum()
    s = float(-(x * np.log(x)).sum())
    principal = int(((x >= 0.05 - 1e-9) & (x <= 0.35 + 1e-9)).sum())
    if s >= 1.5 - 1e-9:
        label = "high-entropy alloy (HEA)"
    elif s >= 1.0 - 1e-9:
        label = "medium-entropy alloy (MEA)"
    else:
        label = "low-entropy alloy"
    return s, label, principal


# ───────────────────────── EAM (setfl) ─────────────────────────
def _read(path):
    with open(path) as f:
        return f.read()


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _num(token):
    """float() that also accepts Fortran 'D' exponents."""
    return float(str(token).replace("D", "E").replace("d", "e"))


def _floats(toks, name):
    try:
        return np.array(toks, float)
    except ValueError:
        try:
            v = np.array([_num(t) for t in toks], float)
        except ValueError as e:
            raise PotentialError(f"{name}: non-numeric table entry ({e})")
    return v


def setfl_sizes(nel, Nrho, Nr):
    """Number of table values after line 5 for each setfl flavour."""
    npair = nel * (nel + 1) // 2
    base = nel * (4 + Nrho + Nr) + npair * Nr
    return {"eam": base,                              # eam/alloy
            "eamfs": base + nel * (nel - 1) * Nr,      # rho_ij for every pair
            "adp": base + 2 * npair * Nr}              # + u(r), w(r) per pair


def read_setfl(path, kind=None, fs=None):
    """Read an eam/alloy, eam/fs or ADP (setfl + u, w tables) file. The number of table
    values must match the flavour exactly; kind=None detects it from that number."""
    if kind is None and fs is not None:               # legacy keyword
        kind = "eamfs" if fs else "eam"
    name = os.path.basename(path)
    try:
        raw = _read(path).replace("\r", "").split("\n")
        elems = raw[3].split(); nel = int(elems[0]); names = elems[1:1 + nel]
        p = raw[4].split()
        Nrho, Nr = int(p[0]), int(p[2])
        drho, dr, cut = _num(p[1]), _num(p[3]), _num(p[4])
    except (OSError, IndexError, ValueError) as e:
        raise PotentialError(f"{name}: not a setfl (eam/alloy, eam/fs, adp) file - lines 4-5 "
                             f"must hold the element list and the table sizes ({e})")
    if nel < 1 or len(names) != nel or len(set(names)) != nel:
        raise PotentialError(f"{name}: header announces {nel} elements but lists {names}")
    if min(Nrho, Nr) < 2 or not all(np.isfinite(v) and v > 0 for v in (drho, dr, cut)):
        raise PotentialError(f"{name}: invalid table grid on line 5")
    toks = " ".join(line.split("#", 1)[0] for line in raw[5:]).split()
    sizes = setfl_sizes(nel, Nrho, Nr)
    match = [k for k in ("eam", "eamfs", "adp") if sizes[k] == len(toks)]
    if kind is None:
        if not match:
            raise PotentialError(f"{name}: {len(toks)} table values fit no supported format "
                                 f"(eam/alloy {sizes['eam']}, eam/fs {sizes['eamfs']}, adp "
                                 f"{sizes['adp']}) - truncated or unsupported file")
        kind = match[0]
    elif kind not in match:
        if match:
            raise PotentialError(f"{name}: the table size matches {PAIR_STYLES[match[0]]}, not "
                                 f"{PAIR_STYLES[kind]} - select '{PAIR_LABELS[match[0]].split('  ')[0]}'")
        raise PotentialError(f"{name}: {len(toks)} table values but {PAIR_STYLES[kind]} needs "
                             f"{sizes[kind]} - truncated file or wrong format")
    fs, adp = kind == "eamfs", kind == "adp"
    i, E = 0, []
    for k in range(nel):
        try:
            Z, m, a0, st = int(_num(toks[i])), _num(toks[i+1]), _num(toks[i+2]), toks[i+3]
        except ValueError:
            raise PotentialError(f"{name}: element block {k+1} must start with 'Z mass a0 lattice'")
        if not (np.isfinite(m) and m > 0):
            raise PotentialError(f"{name}: invalid mass {m} for {names[k]}")
        i += 4
        F = _floats(toks[i:i+Nrho], name); i += Nrho
        rho = [_floats(toks[i+j*Nr:i+(j+1)*Nr], name) for j in range(nel if fs else 1)]
        i += (nel if fs else 1) * Nr
        E.append(dict(Z=Z, mass=m, a0=a0, st=st, F=F, rho=rho))
    npair = nel * (nel + 1) // 2
    tables = [_floats(toks[i+j*Nr:i+(j+1)*Nr], name) for j in range(npair * (3 if adp else 1))]
    return dict(head=raw[0:3], names=names, Nrho=Nrho, drho=drho, Nr=Nr, dr=dr, cut=cut,
                E=E, z2r=tables[:npair], u=tables[npair:2*npair], w=tables[2*npair:],
                kind=kind, fs=fs)


def write_setfl(path, d, n):
    """E_CG(X) = n^3 E(X/n): F*n^3, rho(r/n) (values unchanged), r*phi*n^4; ADP dipole
    u*n^(1/2) and quadrupole w*n^(-1/2) (mu, lambda enter the energy squared)."""
    def block(f, a):
        for k in range(0, len(a), 5):
            f.write(" ".join(f"{v: .16E}" for v in a[k:k+5]) + "\n")
    with open(path, "w") as f:
        f.write(f" CG-{PAIR_STYLES[d.get('kind', 'eam')]} n={n} (Chalamet/Rodney/Shibuta CMS 228 "
                f"(2023) 112306)\n")
        f.write(f" from: {d['head'][0].strip()[:70]}\n")
        f.write(f" F*n^3 rho(r/n) r*phi*n^4 mass*n^3 a0*n"
                f"{' u*n^0.5 w*n^-0.5' if d.get('u') else ''} ; T_CG = n^3 T\n")
        f.write(f" {len(d['names'])} " + " ".join(d['names']) + "\n")
        f.write(f"{d['Nrho']} {d['drho']:.16E} {d['Nr']} {n*d['dr']:.16E} {n*d['cut']:.16E}\n")
        for e in d["E"]:
            f.write(f"{e['Z']} {n**3*e['mass']:.16E} {n*e['a0']:.16E} {e['st']}\n")
            block(f, n**3*e["F"])
            for r in e["rho"]:
                block(f, r)
        for z in d["z2r"]:
            block(f, n**4*z)
        for u in d.get("u") or []:
            block(f, math.sqrt(n)*u)
        for w in d.get("w") or []:
            block(f, w/math.sqrt(n))


# ───────────────────────── MEAM ─────────────────────────
LIB_FIELDS = ["elt", "lat", "z", "ielement", "atwt", "alpha", "b0", "b1", "b2", "b3",
              "alat", "esub", "asub", "t0", "t1", "t2", "t3", "rozero", "ibar"]
MEAM_SCALE = {"Ec": 3, "delta": 3, "re": 1, "rc": 1, "delr": 1}   # power of n
MEAM_DEFAULTS = {"rc": 4.0, "delr": 0.1}       # LAMMPS defaults: dimensional, must be scaled too
MEAM_KEYWORDS = {"Ec", "alpha", "rho0", "delta", "lattce", "attrac", "repuls", "nn2", "Cmin",
                 "Cmax", "rc", "delr", "augt1", "gsmooth_factor", "re", "ialloy",
                 "mixture_ref_t", "erose_form", "zbl", "emb_lin_neg", "bkgd_dyn", "theta"}
_PARAM = re.compile(r"^([A-Za-z_]\w*)\s*(\([^)]*\))?\s*=\s*(\S+)$")


def unquote(s):
    return str(s).strip("'\"")


def read_meam_library(path):
    toks = []
    try:
        for line in _read(path).splitlines():
            toks += shlex.split(line.split("#")[0], posix=False)
    except (OSError, ValueError) as e:
        raise PotentialError(f"{os.path.basename(path)}: cannot read MEAM library ({e})")
    if not toks or len(toks) % 19:
        raise PotentialError(f"{os.path.basename(path)}: token count {len(toks)} is not a "
                             f"multiple of 19 - not a MEAM library file (MS-MEAM is not supported)")
    lib = [dict(zip(LIB_FIELDS, toks[k:k+19])) for k in range(0, len(toks), 19)]
    for e in lib:
        try:
            float(e["atwt"]); float(e["alat"]); float(e["esub"])
        except ValueError:
            raise PotentialError(f"{os.path.basename(path)}: entry {e['elt']} has non-numeric "
                                 f"atwt/alat/esub - not a MEAM library file")
    return lib


def write_meam_library(path, lib, n):
    with open(path, "w") as f:
        f.write(f"# CG-MEAM library n={n}: atwt,esub*n^3 alat*n\n")
        for e in lib:
            e = dict(e)
            for k, p in (("atwt", 3), ("alat", 1), ("esub", 3)):
                e[k] = repr(float(e[k]) * n**p)
            f.write(f"{e['elt']} {e['lat']} {e['z']} {e['ielement']} {e['atwt']}\n")
            f.write(" ".join(e[k] for k in LIB_FIELDS[5:13]) + "\n")
            f.write(" ".join(e[k] for k in LIB_FIELDS[13:]) + "\n")


def parse_meam_params(path):
    """Strictly parse a MEAM parameter file into [(keyword, '(i,j)', value, [i, j])].
    Every keyword LAMMPS knows is accepted; anything else is refused, because a keyword of
    unknown dimension could silently stay unscaled."""
    if not path:
        return []
    name = os.path.basename(path)
    try:
        text = _read(path)
    except OSError as e:
        raise PotentialError(f"cannot read MEAM parameter file ({e})")
    entries = []
    for no, line in enumerate(text.splitlines(), 1):
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        m = _PARAM.match(body)
        if not m:
            raise PotentialError(f"{name}:{no}: expected 'keyword(i,j) = value', got {body!r}")
        k, idx, v = m.group(1), m.group(2) or "", m.group(3)
        if k not in MEAM_KEYWORDS:
            raise PotentialError(f"{name}:{no}: unknown MEAM keyword '{k}' - its units are "
                                 f"unknown, so it cannot be rescaled safely")
        try:
            ids = [int(x) for x in idx.strip("()").split(",")] if idx else []
        except ValueError:
            raise PotentialError(f"{name}:{no}: bad element indices {idx}")
        if any(x < 1 for x in ids):
            raise PotentialError(f"{name}:{no}: element indices start at 1")
        entries.append((k, idx, v, ids))
    return entries


def meam_form(entries):
    """erose_form of a parsed parameter file (LAMMPS default 0)."""
    forms = [v for k, idx, v, _ in entries if k == "erose_form"]
    try:
        form = int(_num(forms[-1])) if forms else 0
    except ValueError:
        raise PotentialError(f"erose_form = {forms[-1]} is not a number")
    if form not in (0, 1, 2):
        raise PotentialError(f"unsupported erose_form = {form}")
    return form


def meam_zbl_active(entries, nlib):
    """Pairs (i, j) for which ZBL is on in the original potential (LAMMPS default: on)."""
    z = {}
    for k, idx, v, ids in entries:
        if k == "zbl" and len(ids) == 2:
            z[tuple(sorted(ids))] = _num(v)
    return [(i, j) for i in range(1, nlib+1) for j in range(i, nlib+1) if z.get((i, j), 1) != 0]


def write_meam_params(src, dst, n, nlib):
    """Rescale a MEAM parameter file (src may be None = LAMMPS defaults only).
    Ec, delta *n^3; re, rc, delr *n; repuls *n when erose_form = 1 (the Rose energy then
    contains repuls/r). rc and delr are written even when src relies on the LAMMPS
    defaults, and ZBL is switched off for every pair of the nlib library elements (its
    screening length is absolute, so it cannot be rescaled)."""
    entries = parse_meam_params(src)
    form = meam_form(entries)
    bad = sorted({x for *_, ids in entries for x in ids if x > nlib})
    if bad:
        raise PotentialError(f"the MEAM parameter file uses element index {bad[-1]} but only "
                             f"{nlib} library elements are listed")
    out, seen = [f"# CG-MEAM params n={n}: Ec,delta*n^3 re,rc,delr*n"
                 f"{', repuls*n (erose_form=1)' if form == 1 else ''}, ZBL off\n"], set()
    for k, idx, v, ids in entries:
        if k == "zbl":
            continue
        seen.add(k)
        power = 1 if (k == "repuls" and form == 1) else MEAM_SCALE.get(k, 0)
        if power and n != 1:
            v = repr(_num(v) * n**power)
        out.append(f"  {k}{idx} = {v}\n")
    for k, v in MEAM_DEFAULTS.items():
        if k not in seen:
            out.append(f"\n# {k} not set in the original -> LAMMPS default {v} Å, rescaled\n"
                       f"  {k} = {v * n!r}\n")
    out.append("\n# ZBL uses an absolute screening length -> not rescalable; disabled\n")
    out += [f"  zbl({i},{j}) = 0\n" for i in range(1, nlib+1) for j in range(i, nlib+1)]
    _write(dst, "".join(out))


def classify_meam_file(path):
    """'library', 'params' or None, from the file contents."""
    try:
        with open(path) as f:
            text = f.read(200000)
    except OSError:
        return None
    body = "\n".join(l.split("#")[0] for l in text.splitlines())
    if re.search(r"^\s*[A-Za-z_]\w*\s*(\([^)]*\))?\s*=", body, re.M):
        return "params"
    toks = body.split()
    if toks and len(toks) % 19 == 0 and toks[0][:1] in "'\"":
        return "library"
    return None


# ───────────────────────── attached potential ─────────────────────────
@dataclass
class PotentialInfo:
    pair: str                      # eam | eamfs | adp | meam
    files: list                    # attached atomistic files
    elements: list                 # unique element names, file order
    masses: dict                   # name -> amu (atomistic)
    ref: dict                      # name -> (reference lattice, a0) from the header
    cutoff: float                  # atomistic cutoff, Å
    data: object = None            # parsed setfl dict or MEAM library list
    param_index_max: int = 0       # MEAM: largest element index in the parameter file
    notes: list = field(default_factory=list)
    erose_form: int = 0            # MEAM

    def describe(self):
        kind = PAIR_STYLES[self.pair]
        return (f"pair_style {kind}: {len(self.elements)} element(s) "
                f"[{' '.join(self.elements[:12])}{' ...' if len(self.elements) > 12 else ''}], "
                f"cutoff {self.cutoff:.3f} Å")


def detect_pair_style(path):
    """Best guess of the format of an attached file: eam | eamfs | adp | meam | None.
    Setfl flavours are told apart by their exact number of table values."""
    low = os.path.basename(path).lower()
    if "meam" in low or low.endswith((".library", ".lib")):
        return "meam"
    try:
        return read_setfl(path)["kind"]
    except PotentialError:
        pass
    if classify_meam_file(path):
        return "meam"
    for suf, kind in ((".adp", "adp"), (".adp.alloy", "adp"), (".fs", "eamfs"),
                      (".alloy", "eam"), (".setfl", "eam")):
        if low.endswith(suf):
            return kind                 # unreadable, but loading will explain why
    return None


def load_potential(pair, potential=None, library=None, params=None):
    """Parse and validate the attached atomistic potential."""
    if pair not in PAIR_STYLES:
        raise PotentialError(f"unsupported pair style '{pair}' (use {', '.join(PAIR_STYLES)})")
    if pair in SETFL_PAIRS:
        if not potential or not os.path.isfile(potential):
            raise PotentialError(f"attach the atomistic {PAIR_STYLES[pair]} potential file")
        if potential.lower().endswith(".eam"):
            raise PotentialError(f"{os.path.basename(potential)} looks like a single-element "
                                 f"funcfl file (pair_style eam); convert it to setfl "
                                 f"(eam/alloy) first, e.g. with atomman or OpenKIM tools")
        d = read_setfl(potential, kind=pair)
        info = PotentialInfo(pair, [potential], list(dict.fromkeys(d["names"])),
                             {nm: e["mass"] for nm, e in zip(d["names"], d["E"])},
                             {nm: (e["st"], e["a0"]) for nm, e in zip(d["names"], d["E"])},
                             d["cut"], d)
        if pair == "eam" and len(d["names"]) == 1:
            info.notes.append("single-element file: eam/alloy and eam/fs are equivalent")
        return info
    if not library or not os.path.isfile(library):
        raise PotentialError("attach the MEAM library file (and the parameter file, if any)")
    if params and not os.path.isfile(params):
        raise PotentialError(f"MEAM parameter file not found: {params}")
    lib = read_meam_library(library)
    masses, ref = {}, {}
    for e in lib:                                   # LAMMPS uses the first matching entry
        nm = unquote(e["elt"])
        if nm not in masses:
            masses[nm] = float(e["atwt"])
            ref[nm] = (unquote(e["lat"]), float(e["alat"]))
    entries = parse_meam_params(params)
    rcs = [v for k, idx, v, _ in entries if k == "rc"]
    try:
        rc = _num(rcs[-1]) if rcs else MEAM_DEFAULTS["rc"]
    except ValueError:
        raise PotentialError(f"MEAM parameter file: rc = {rcs[-1]} is not a number")
    maxidx = max([0] + [x for *_, ids in entries for x in ids])
    info = PotentialInfo("meam", [library] + ([params] if params else []), list(masses),
                         masses, ref, rc, lib, maxidx, erose_form=meam_form(entries))
    if not params:
        info.notes.append("no parameter file: LAMMPS defaults (rc = 4.0 Å) are used")
    if info.erose_form == 1:
        info.notes.append("erose_form = 1: repuls is a length and is rescaled by n")
    return info


@dataclass
class PairSetup:
    """pair_style / pair_coeff lines for the atomistic (AA) and rescaled (CG) potentials.
    'aa0' is the atomistic potential exactly as the CG model sees it (MEAM: ZBL off), the
    reference for the exactness test; for EAM it is identical to 'aa'."""
    pair: str
    aa_files: list
    cg_files: list
    lib_elements: list = None
    masses: dict = None
    aa0_files: list = None
    zbl_active: list = None        # MEAM: pairs with ZBL on in the original potential

    @property
    def style(self):
        return "pair_style " + PAIR_STYLES[self.pair]

    def coeff(self, which, elements, prefix=""):
        """which: 'aa' | 'aa0' | 'cg' (True/False accepted for cg/aa)."""
        which = {True: "cg", False: "aa"}.get(which, which)
        files = {"aa": self.aa_files, "cg": self.cg_files,
                 "aa0": self.aa0_files or self.aa_files}[which]
        p = lambda f: f if f == "NULL" else prefix + f
        if self.pair == "meam":
            return (f"pair_coeff * * {p(files[0])} {' '.join(self.lib_elements)} "
                    f"{p(files[1])} {' '.join(elements)}")
        return f"pair_coeff * * {p(files[0])} {' '.join(elements)}"


def _safe_name(path):
    return re.sub(r"\s+", "_", os.path.basename(path))


def _copy(src, outdir):
    dst = os.path.join(outdir, _safe_name(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    return os.path.basename(dst)


def default_lib_elements(info, elements):
    """MEAM: element list after the library in the ORIGINAL pair_coeff. Parameter-file
    indices refer to this order. Without user input, use the library's own order when it
    is the potential's dedicated library, else the alloy elements."""
    lib = info.elements
    if set(lib) == set(elements):
        return list(lib)
    if info.param_index_max and len(lib) == info.param_index_max and set(elements) <= set(lib):
        return list(lib)
    return list(elements)


def rescale_potential(info, n, outdir, elements, lib_elements=None):
    """Write the CG potential next to a copy of the atomistic one."""
    os.makedirs(outdir, exist_ok=True)
    if info.pair in SETFL_PAIRS:
        src = info.files[0]
        ext = SETFL_EXT[info.pair]
        base = _safe_name(src)
        for suf in (".adp.alloy", ".eam.alloy", ".eam.fs", ".adp", ".alloy", ".fs", ".setfl"):
            if base.lower().endswith(suf):
                base = base[:-len(suf)]; break
        cg = f"{base}_CG_n{n}.{ext}"
        write_setfl(os.path.join(outdir, cg), info.data, n)
        return PairSetup(info.pair, [_copy(src, outdir)], [cg], masses=info.masses)
    lib_src = info.files[0]
    par_src = info.files[1] if len(info.files) > 1 else None
    libE = list(lib_elements) if lib_elements else default_lib_elements(info, elements)
    missing = [e for e in libE if e not in info.masses]
    if missing:
        raise PotentialError(f"MEAM library elements {missing} are not in "
                             f"{os.path.basename(lib_src)}")
    if not set(elements) <= set(libE):
        raise PotentialError(f"alloy elements {sorted(set(elements) - set(libE))} are missing "
                             f"from the MEAM library-element list {libE}")
    if info.param_index_max > len(libE):
        raise PotentialError(f"the MEAM parameter file uses element index {info.param_index_max} "
                             f"but only {len(libE)} library elements are listed ({libE}); give "
                             f"the full list in the order of the potential's original pair_coeff")
    stem = lambda p: os.path.splitext(_safe_name(p))[0]
    cg_lib = f"{stem(lib_src)}_CG_n{n}.meam"
    cg_par = f"{stem(par_src) if par_src else 'params'}_CG_n{n}.meam"
    if cg_par == cg_lib:
        cg_par = cg_par.replace(".meam", "_params.meam")
    aa0_par = f"{stem(par_src) if par_src else 'params'}_AA_noZBL.meam"
    write_meam_library(os.path.join(outdir, cg_lib), info.data, n)
    write_meam_params(par_src, os.path.join(outdir, cg_par), n, len(libE))
    write_meam_params(par_src, os.path.join(outdir, aa0_par), 1, len(libE))
    aa = [_copy(lib_src, outdir), _copy(par_src, outdir) if par_src else "NULL"]
    return PairSetup("meam", aa, [cg_lib, cg_par], lib_elements=libE, masses=info.masses,
                     aa0_files=[aa[0], aa0_par],
                     zbl_active=meam_zbl_active(parse_meam_params(par_src), len(libE)))


# ───────────────────────── TPMS geometry ─────────────────────────
def tpms_field(surface, pos, L, periods):
    X, Y, Z = (TWO_PI * periods * pos[:, k] / L for k in range(3))
    return SURFACES[surface](X, Y, Z)


def resolve_phase(phase, t):
    """'auto' keeps the original convention: t > 0 -> sheet |f| <= t, else network f <= 0."""
    if phase == "auto":
        return ("sheet", t) if t > 0 else ("network", 0.0)
    return phase, t


def level_for_density(f, phase, density):
    """Iso-level that gives the requested solid fraction on the discrete lattice."""
    g = np.abs(f) if phase == "sheet" else f
    return float(np.quantile(g, min(max(density, 1e-6), 1.0), method="higher"))


def solid_mask(f, phase, level):
    return (np.abs(f) <= level) if phase == "sheet" else (f <= level)


def coordination(idx, keep, kind, nc):
    """Number of occupied first neighbours of every site (periodic box)."""
    occ = np.zeros((2*nc,)*3, bool)
    k = idx[keep]
    occ[k[:, 0], k[:, 1], k[:, 2]] = True
    cnt = np.zeros(occ.shape, np.int8)
    for off in NN_OFFSETS[kind]:
        cnt += np.roll(occ, shift=tuple(-off), axis=(0, 1, 2))
    return cnt[idx[:, 0], idx[:, 1], idx[:, 2]]


def prune(idx, keep, kind, nc, min_coord):
    """Iteratively remove beads with fewer than min_coord occupied first neighbours."""
    keep, removed = keep.copy(), 0
    while min_coord > 0:
        bad = keep & (coordination(idx, keep, kind, nc) < min_coord)
        if not bad.any():
            break
        keep &= ~bad
        removed += int(bad.sum())
    return keep, removed


@dataclass
class Geometry:
    pos: np.ndarray
    keep: np.ndarray
    L: float
    phase: str
    level: float
    fill: float
    surface_fraction: float
    pruned: int
    n_sites: int


def build_geometry(kind, a_cg, cells, surface, periods=1, phase="auto", t=0.0,
                   density=None, min_coord=0):
    """CG bead lattice of `kind` with the TPMS solid kept. No LAMMPS needed."""
    idx = lattice_index(kind, cells)
    pos = idx * (0.5 * a_cg)
    L = cells * a_cg
    f = tpms_field(surface, pos, L, periods)
    phase, level = resolve_phase(phase, t)
    if density:
        if phase == "auto":
            phase = "network"
        level = level_for_density(f, phase, density)
    if phase == "sheet" and level <= 0:
        raise ValueError("sheet TPMS needs a positive thickness level t (or a target density)")
    keep = solid_mask(f, phase, level)
    keep, removed = prune(idx, keep, kind, cells, min_coord)
    full = len(NN_OFFSETS[kind])
    surf = float(np.mean(coordination(idx, keep, kind, cells)[keep] < full)) if keep.any() else 0.0
    return Geometry(pos[keep], keep, L, phase, level, float(keep.mean()), surf, removed, len(idx))


# ───────────────────────── configuration ─────────────────────────
@dataclass
class Config:
    pair: str = "eam"                 # eam (eam/alloy) | eamfs | adp | meam
    potential: str = ""               # setfl file (EAM)
    library: str = ""                 # MEAM library
    params: str = ""                  # MEAM parameter file (optional)
    lib_elements: list = None         # MEAM: element order of the ORIGINAL pair_coeff
    elements: list = field(default_factory=list)
    composition: list = None          # fractions (normalised); None = equiatomic
    lattice: str = "fcc"              # fcc | bcc
    n: int = 5
    a0: float = None                  # atomistic lattice constant (guess); None = from potential
    auto_a0: bool = False
    verify: bool = False
    check_pure: bool = False
    compare_phases: bool = False
    surface: str = "gyroid"
    cells: int = 20
    box_nm: float = None              # CG box edge in nm (overrides cells; rounds up)
    periods: int = 1
    phase: str = "auto"               # auto | network | sheet
    t: float = 0.0
    density: float = None             # target solid fraction (overrides t)
    min_coord: int = 0
    temp: float = 300.0
    dt_aa: float = 0.001
    dt_cg: float = None               # CG timestep override, ps (default n*dt_aa = exact map)
    tdamp: float = None               # CG thermostat damping, ps (default 100*n*dt_aa)
    pdamp: float = None               # CG barostat damping, ps (default 1000*n*dt_aa)
    equil: int = 20000
    crush: bool = False
    crush_strain: float = 0.5
    strain_rate: float = 1e-4
    crush_axis: str = "z"
    seed: int = 2024
    lmp: str = "lmp"
    outdir: str = "cg_hea_output"

    def fractions(self):
        comp = self.composition or [1.0] * len(self.elements)
        s = float(sum(comp))
        return [c / s for c in comp]

    def validate(self):
        err = []
        if self.lattice not in LATTICES:
            err.append(f"lattice must be one of {list(LATTICES)}")
        if not self.elements:
            err.append("select at least one element")
        if len(set(self.elements)) != len(self.elements):
            err.append("an element is listed twice")
        if self.composition is not None:
            if len(self.composition) != len(self.elements):
                err.append("composition needs one value per element")
            elif any(c < 0 for c in self.composition) or sum(self.composition) <= 0:
                err.append("composition values must be positive")
        if self.n < 1:
            err.append("n must be >= 1")
        if self.cells < 1 or self.periods < 1:
            err.append("cells and periods must be >= 1")
        if self.surface not in SURFACES:
            err.append(f"surface must be one of {sorted(SURFACES)}")
        if self.phase not in ("auto", "network", "sheet"):
            err.append("phase must be auto, network or sheet")
        if self.density is not None and not 0 < self.density <= 1:
            err.append("target relative density must be in (0, 1]")
        if self.phase == "sheet" and self.t <= 0 and not self.density:
            err.append("sheet TPMS needs t > 0 or a target density")
        if self.crush_axis not in ("x", "y", "z"):
            err.append("crush axis must be x, y or z")
        if self.a0 is not None and self.a0 <= 0:
            err.append("a0 must be positive")
        if self.temp <= 0 or self.dt_aa <= 0:
            err.append("temperature and timestep must be positive")
        for k in ("box_nm", "dt_cg", "tdamp", "pdamp"):
            v = getattr(self, k)
            if v is not None and not (math.isfinite(v) and v > 0):
                err.append(f"{k} must be positive")
        if self.crush and not 0 < self.crush_strain < 1:
            err.append("final compression strain must be between 0 and 1")
        if self.crush and self.strain_rate <= 0:
            err.append("strain rate must be positive")
        if self.equil < 0:
            err.append("equilibration steps must be >= 0")
        if not 1 <= self.seed <= 900_000_000:
            err.append("seed must be between 1 and 900000000 (LAMMPS velocity seed)")
        if err:
            raise ValueError("; ".join(err))

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        d = json.loads(_read(path))
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_cli(self, script="cg_hea_tpms.py"):
        """Equivalent command line (for batch / HPC runs)."""
        q = shlex.quote
        a = [f"python3 {script}", f"--pair {self.pair}"]
        if self.pair == "meam":
            a.append(f"--library {q(self.library)}")
            if self.params:
                a.append(f"--params {q(self.params)}")
            if self.lib_elements:
                a.append("--lib-elements " + " ".join(self.lib_elements))
        else:
            a.append(f"--potential {q(self.potential)}")
        a.append("--elements " + " ".join(self.elements))
        if self.composition:
            a.append("--composition " + " ".join(f"{c:.6g}" for c in self.fractions()))
        a += [f"--lattice {self.lattice}", f"-n {self.n}"]
        if self.a0:
            a.append(f"--a0 {self.a0:.6g}")
        a += [f"--{k.replace('_', '-')}" for k in ("auto_a0", "verify", "check_pure",
                                                   "compare_phases") if getattr(self, k)]
        a += [f"--surface {self.surface}",
              f"--box-nm {self.box_nm:g}" if self.box_nm else f"--cells {self.cells}",
              f"--periods {self.periods}", f"--phase {self.phase}", f"--t {self.t:g}"]
        if self.density:
            a.append(f"--density {self.density:g}")
        if self.min_coord:
            a.append(f"--min-coord {self.min_coord}")
        a += [f"--temp {self.temp:g}", f"--dt-aa {self.dt_aa:g}", f"--equil {self.equil}"]
        a += [f"--{k.replace('_', '-')} {getattr(self, k):g}" for k in ("dt_cg", "tdamp", "pdamp")
              if getattr(self, k)]
        if self.crush:
            a += ["--crush", f"--crush-strain {self.crush_strain:g}",
                  f"--strain-rate {self.strain_rate:g}", f"--crush-axis {self.crush_axis}"]
        a += [f"--seed {self.seed}", f"--lmp {q(self.lmp)}", f"--outdir {q(self.outdir)}"]
        return " \\\n    ".join(a)


# ───────────────────────── LAMMPS ─────────────────────────
def split_command(cmd):
    if isinstance(cmd, (list, tuple)):
        return list(cmd)
    if os.path.exists(cmd):                       # a plain path, possibly with spaces
        return [cmd]
    return shlex.split(cmd, posix=(os.name != "nt"))


def find_lammps(preferred=None):
    """First LAMMPS executable found on PATH (or None)."""
    for c in ([preferred] if preferred else []) + ["lmp", "lmp_serial", "lmp_mpi", "lmp_omp",
                                                   "lammps", "lmp.exe"]:
        p = shutil.which(c) if c else None
        if p:
            return p
    return None


class Session:
    """Logging, LAMMPS execution and cancellation shared by the CLI and the GUI."""

    def __init__(self, log=print):
        self.log = log
        self.cancelled = False
        self._proc = None
        self._lock = threading.Lock()

    def cancel(self):
        self.cancelled = True
        with self._lock:
            p = self._proc
        if p is not None and p.poll() is None:
            p.terminate()

    def check(self):
        if self.cancelled:
            raise Cancelled("stopped by user")

    def _popen(self, cmd, cwd):
        try:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, text=True, bufsize=1)
        except FileNotFoundError:
            raise LammpsError(f"LAMMPS executable not found: '{cmd[0]}'. Install LAMMPS "
                              f"(with the MANYBODY package; MEAM for meam) or give its path.")
        with self._lock:
            self._proc = p
        return p

    def run_lammps(self, lmp, inp, cwd, name="_tmp"):
        """Run a short LAMMPS job and return its screen output."""
        self.check()
        _write(os.path.join(cwd, f"{name}.in"), inp)
        p = self._popen(split_command(lmp) + ["-in", f"{name}.in", "-log", f"{name}.log",
                                              "-nocite"], cwd)
        out, _ = p.communicate()
        with self._lock:
            self._proc = None
        self.check()
        m = re.search(r"^ERROR.*$", out, re.M)
        if m or p.returncode:
            msg = out[m.start():][:600] if m else out[-600:]
            raise LammpsError(f"LAMMPS failed in {os.path.join(cwd, name + '.in')}:\n{msg}")
        return out

    def stream_lammps(self, cmd, cwd, on_line=None):
        """Run a (long) LAMMPS job, streaming its output; returns the exit code."""
        self.check()
        p = self._popen(split_command(cmd), cwd)
        for line in p.stdout:
            (on_line or self.log)(line.rstrip("\n"))
        rc = p.wait()
        with self._lock:
            self._proc = None
        self.check()
        return rc


def write_data(path, pos, types, L, masses, title, fmt=".8f"):
    with open(path, "w") as f:
        f.write(f"{title}\n\n{len(pos)} atoms\n{len(masses)} atom types\n\n")
        for ax in "xyz":
            f.write(f"0.0 {L:{fmt}} {ax}lo {ax}hi\n")
        f.write("\nMasses\n\n" + "".join(f"{i} {m:.10g}\n" for i, m in enumerate(masses, 1)))
        f.write("\nAtoms # atomic\n\n")
        f.writelines(f"{i} {t} {x:{fmt}} {y:{fmt}} {z:{fmt}}\n"
                     for i, (t, (x, y, z)) in enumerate(zip(types, pos), 1))


_HEADER = "units metal\natom_style atomic\nboundary p p p\n"


def relax_cubic(S, lmp, cwd, name, pair, elements, kind, nc, types, a_guess, masses):
    """Relax a cubic `kind` supercell (types given) with the atomistic potential:
    coarse E(a) scan around a_guess (robust to bad guesses / mislabelled headers),
    then box/relax. Returns (a0, E/atom, scan_hit_boundary)."""
    pts = lattice(kind, 1.0, nc)
    write_data(os.path.join(cwd, f"{name}.data"), pts * a_guess, types, nc * a_guess, masses, name)
    for attempt in range(4):                     # re-centre if the minimum is on the scan edge
        scan = []
        for s in SCAN:
            L = s * nc * a_guess
            scan.append(f"change_box all x final 0 {L:.8f} y final 0 {L:.8f} z final 0 {L:.8f} "
                        f"remap units box\nrun 0\nprint \"SCAN {s} $(pe/atoms:%.10f)\"")
        out = S.run_lammps(lmp, f"""{_HEADER}read_data {name}.data
{pair.style}
{pair.coeff('aa', elements, '../')}
thermo_style custom step pe
thermo 1
{chr(10).join(scan)}
""", cwd, name + "_scan")
        res = [(float(a), float(e)) for a, e in re.findall(r"^SCAN (\S+) (\S+)$", out, re.M)]
        k = int(np.argmin([e for _, e in res]))
        edge = k in (0, len(res) - 1)
        if not edge:
            break
        a_guess *= res[k][0]
        res = [(1.0, res[k][1])]; k = 0
    s_best = res[k][0]
    L = s_best * nc * a_guess
    out = S.run_lammps(lmp, f"""{_HEADER}read_data {name}.data
change_box all x final 0 {L:.8f} y final 0 {L:.8f} z final 0 {L:.8f} remap units box
{pair.style}
{pair.coeff('aa', elements, '../')}
thermo 100
min_style cg
fix r all box/relax iso 0.0 vmax 0.001
minimize 1e-12 1e-14 20000 200000
unfix r
fix r all box/relax iso 0.0 vmax 0.001
minimize 1e-12 1e-14 20000 200000
print "A0RES $(lx/{nc}:%.8f) $(pe/atoms:%.10f) $(press:%.4f)"
""", cwd, name)
    m = re.search(r"A0RES (\S+) (\S+) (\S+)", out)
    return float(m.group(1)), float(m.group(2)), edge


def static_state(S, lmp, cwd, name, pair, which, elements, pos, types, L, masses):
    """Energy, the 6 pressure-tensor components and the per-atom forces of one frozen
    configuration (coordinates written at full precision)."""
    write_data(os.path.join(cwd, name + ".data"), pos, types, L, masses, name, fmt=".17g")
    out = S.run_lammps(lmp, f"""{_HEADER}read_data {name}.data
{pair.style}
{pair.coeff(which, elements, '../')}
thermo_style custom step pe pxx pyy pzz pxy pxz pyz
run 0
print "STATIC $(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pxy:%.17g) $(pxz:%.17g) $(pyz:%.17g)"
write_dump all custom {name}.forces id fx fy fz modify sort id format float %.17g
""", cwd, name)
    vals = np.array(re.search(r"^STATIC (.+)$", out, re.M).group(1).split(), float)
    F = np.loadtxt(os.path.join(cwd, name + ".forces"), skiprows=9, ndmin=2)
    if F.shape != (len(pos), 4) or not np.all(np.isfinite(F)):
        raise LammpsError(f"invalid force dump {name}.forces")
    return vals[0], vals[1:], F[:, 1:]


def compare_states(a, b, k):
    """Relative deviation of state b from a when b is scaled by k (E k^3, P 1, F k^2)."""
    (ea, pa, fa), (eb, pb, fb) = a, b
    dE = abs(eb / k**3 - ea) / max(abs(ea), 1e-12)
    dP = float(np.max(np.abs(pb - pa)) / max(np.max(np.abs(pa)), 1.0))
    dF = float(np.max(np.abs(fb / k**2 - fa)) / max(np.max(np.abs(fa)), 1e-12))
    return dict(energy=dE, stress=dP, force=dF, max=max(dE, dP, dF))


def nve_trace(S, lmp, cwd, name, pair, which, elements, pos, types, L, masses, T, dt):
    """1000 NVE steps from `pos` at temperature T; (step, T, PE) every 200 steps."""
    write_data(os.path.join(cwd, name + ".data"), pos, types, L, masses, name, fmt=".17g")
    out = S.run_lammps(lmp, f"""{_HEADER}read_data {name}.data
{pair.style}
{pair.coeff(which, elements, '../')}
velocity all create {T!r} 777 mom yes rot yes dist gaussian
timestep {dt!r}
fix 1 all nve
thermo 200
fix pr all print 200 "VERIFY $(step) $(temp:%.12g) $(pe:%.12g)"
run 1000
""", cwd, name)
    return [tuple(map(float, r)) for r in re.findall(r"^VERIFY (\S+) (\S+) (\S+)$", out, re.M)]


def dyn_dev(a, b, k):
    d = 0.0
    for (_, ta, pa), (_, tb, pb) in zip(a, b):
        d = max(d, abs(tb/k**3 - ta)/max(ta, 1e-9), abs(pb/k**3 - pa)/abs(pa))
    return d


# ───────────────────────── pipeline ─────────────────────────
def run_pipeline(cfg: Config, session: Session = None):
    """Rescale the potential, relax, verify and build the CG TPMS model. Returns a summary."""
    S = session or Session()
    log = S.log
    cfg.validate()
    n, els, kind = cfg.n, list(cfg.elements), cfg.lattice
    comp = cfg.fractions()
    info = load_potential(cfg.pair, cfg.potential, cfg.library, cfg.params)
    missing = [e for e in els if e not in info.masses]
    if missing:
        raise PotentialError(f"element(s) {missing} not in the potential "
                             f"(available: {' '.join(info.elements)})")
    outdir = os.path.abspath(cfg.outdir)
    checks = os.path.join(outdir, "checks")
    os.makedirs(checks, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    tag = "".join(els)
    s_conf, klass, npr = entropy_class(comp)
    summary = dict(config=asdict(cfg), lattice=kind, elements=els, composition=comp,
                   entropy_R=s_conf, alloy_class=klass)
    log(f"Alloy {tag} ({' '.join(f'{e}{100*c:.4g}' for e, c in zip(els, comp))} at.%), "
        f"{kind.upper()}: dS_conf = {s_conf:.3f} R -> {klass}, {npr} principal element(s)")

    # 1. rescale potential
    pair = rescale_potential(info, n, outdir, els, cfg.lib_elements)
    m_aa = [info.masses[e] for e in els]
    m_cg = [n**3 * m for m in m_aa]
    log(f"[1] Potential rescaled (n={n}, bead = {n**3} atoms): {pair.cg_files[0]}"
        + (f" + {pair.cg_files[1]}" if pair.pair == "meam" else "")
        + f"; cutoff {info.cutoff:.3f} -> {n*info.cutoff:.3f} Å")
    if pair.pair == "meam":
        log(f"     MEAM library-element order: {' '.join(pair.lib_elements)}"
            f"{'' if cfg.lib_elements else '  (default - check it against the potential docs)'}")
    for note in info.notes:
        log(f"     note: {note}")
    if pair.zbl_active:
        log(f"     ZBL was ON in the original potential for {len(pair.zbl_active)} pair(s) "
            f"(explicitly or by LAMMPS default). It cannot be rescaled, so it is OFF in the CG "
            f"model: the CG model is exactly '{pair.aa0_files[1]}' (original, ZBL off).")
    summary["cg_potential"] = pair.cg_files
    summary["zbl_active_in_original"] = pair.zbl_active
    summary["pair_coeff_cg"] = pair.coeff("cg", els)

    a0 = cfg.a0
    if a0 is None:
        a0 = vegard_a0(info.ref, els, comp, kind)
        src = "Vegard estimate from the potential header"
        if a0 is None:
            a0, src = DEFAULT_A0[kind], "default"
        log(f"     a0 guess ({kind.upper()}) = {a0:.4f} Å ({src})")

    # 1b. element-mapping sanity check: each pure element as FCC and BCC
    if cfg.check_pure:
        S.check()
        log("[1b] Pure-element check (AA potential) - compare with the potential's reference "
            "values; the stable lattice should match the element's ground state:")
        summary["pure"] = {}
        for k, el in enumerate(els, 1):
            row = {}
            for lk in ("fcc", "bcc"):
                S.check()
                g = vegard_a0(info.ref, [el], [1.0], lk) or DEFAULT_A0[lk]
                nc = PURE_CELLS[lk]
                row[lk] = relax_cubic(S, cfg.lmp, checks, f"pure_{el}_{lk}", pair, els, lk, nc,
                                      np.full(len(LATTICES[lk]) * nc**3, k), g, m_aa)[:2]
            dE = 1e3 * (row["bcc"][1] - row["fcc"][1])
            lat, ar = info.ref.get(el, ("?", 0))
            log(f"     type {k} = {el:>3s}:  FCC a0 {row['fcc'][0]:.4f} Å E {row['fcc'][1]:.4f}  | "
                f"BCC a0 {row['bcc'][0]:.4f} Å E {row['bcc'][1]:.4f} eV/atom  -> "
                f"{'BCC' if dE < 0 else 'FCC'} lower by {abs(dE):.1f} meV/atom "
                f"(header: {unquote(lat)} {float(ar):.4f} Å)")
            summary["pure"][el] = dict(fcc=row["fcc"], bcc=row["bcc"])

    # 1c. FCC vs BCC energy of the random alloy
    if cfg.compare_phases:
        S.check()
        prng = np.random.default_rng(cfg.seed + 1)
        res = {}
        for lk in ("fcc", "bcc"):
            g = vegard_a0(info.ref, els, comp, lk) or DEFAULT_A0[lk]
            if cfg.a0 and lk == kind:
                g = cfg.a0
            nc = BULK_CELLS[lk]
            t = assign_species(ATOMS_PER_CELL[lk] * nc**3, comp, prng)
            res[lk] = relax_cubic(S, cfg.lmp, checks, f"phase_{lk}", pair, els, lk, nc, t, g, m_aa)
        dE = 1e3 * (res["bcc"][1] - res["fcc"][1])
        log(f"[1c] Random {tag} (AA, 0 K): FCC a0 {res['fcc'][0]:.4f} Å, E {res['fcc'][1]:.4f} | "
            f"BCC a0 {res['bcc'][0]:.4f} Å, E {res['bcc'][1]:.4f} eV/atom")
        log(f"     E_BCC - E_FCC = {dE:+.1f} meV/atom -> this potential favours "
            f"{'BCC' if dE < 0 else 'FCC'} (one random configuration, no entropy/ordering)"
            + ("" if (dE < 0) == (kind == "bcc") else
               f"  ** you selected {kind.upper()}: it may transform during MD **"))
        summary["phase_compare"] = dict(fcc=res["fcc"][:2], bcc=res["bcc"][:2], dE_meV=dE)
        if cfg.a0 is None and not cfg.auto_a0:
            a0 = res[kind][0]

    # 2. lattice constant of the random alloy (atomistic potential, exact for CG by scaling)
    bulk_nc = BULK_CELLS[kind]
    bt = assign_species(ATOMS_PER_CELL[kind] * bulk_nc**3, comp, rng)
    if cfg.auto_a0:
        S.check()
        a0, ecoh, edge = relax_cubic(S, cfg.lmp, checks, "bulk_aa", pair, els, kind, bulk_nc,
                                     bt, a0, m_aa)
        log(f"[2] Random-alloy {kind.upper()} a0 = {a0:.4f} Å, E_coh = {ecoh:.4f} eV/atom "
            f"-> a_CG = {n*a0:.4f} Å, E/bead = {n**3*ecoh:.2f} eV")
        if edge:
            log("     WARNING: E(a) scan minimum at the scan edge - check the a0 guess")
        summary.update(a0_relaxed=a0, E_coh=ecoh)
    else:
        log(f"[2] Using a0 = {a0:.4f} Å (enable auto-a0 to relax it with LAMMPS)")
    a_cg = n * a0
    summary.update(a0=a0, a_cg=a_cg)

    # 3. exact-mapping verification on the bulk random alloy
    if cfg.verify:
        S.check()
        bp = lattice(kind, 1.0, bulk_nc)
        shake = (bp + np.random.default_rng(cfg.seed + 2).normal(0.0, 0.03, bp.shape)) % bulk_nc
        holes = SURFACES["gyroid"](*(TWO_PI * bp / bulk_nc).T) <= 0     # free surfaces
        configs = {"distorted bulk": (shake, bt), "porous (free surfaces)": (shake[holes], bt[holes])}
        variants = [("AA0", "aa0", 1, m_aa), ("CG", "cg", n, m_cg)]
        if pair.zbl_active:                          # also the original potential, ZBL on
            variants.append(("AA", "aa", 1, m_aa))
        st, dyn = {}, {}
        for ci, (cname, (pts, ty)) in enumerate(configs.items()):
            for lab, which, sc, ms in variants:
                st[cname, lab] = static_state(S, cfg.lmp, checks, f"verify_{ci}_{lab}", pair,
                                              which, els, pts*a0*sc, ty, bulk_nc*a0*sc, ms)
        for lab, which, sc, ms in variants:
            T, dt = (n**3*cfg.temp, n*cfg.dt_aa) if lab == "CG" else (cfg.temp, cfg.dt_aa)
            dyn[lab] = nve_trace(S, cfg.lmp, checks, f"verify_md_{lab}", pair, which, els,
                                 bp*a0*sc, bt, bulk_nc*a0*sc, ms, T, dt)
        ref = "AA potential with ZBL off, as in the CG model" if pair.pair == "meam" else "AA potential"
        log(f"[3] Exact-mapping check, {kind.upper()} random alloy (reference: {ref}):")
        sdev, static = 0.0, {}
        for cname in configs:
            c = compare_states(st[cname, "AA0"], st[cname, "CG"], n)
            static[cname] = c
            sdev = max(sdev, c["max"])
            log(f"     static, {cname:23s}: rel. dev  E {c['energy']:.1e}  stress {c['stress']:.1e}"
                f"  forces {c['force']:.1e}  ({'PASS' if c['max'] < 1e-8 else 'FAIL'})")
        s_ok = sdev < 1e-8
        log(f"     dynamics (AA {cfg.temp:.0f} K, dt={cfg.dt_aa} | CG {n**3*cfg.temp:.0f} K, "
            f"dt={n*cfg.dt_aa:g}):")
        for (s, ta, pa), (_, tc, pc) in zip(dyn["AA0"], dyn["CG"]):
            log(f"     step {int(s):5d}  T {ta:9.3f} vs {tc/n**3:9.3f}   "
                f"PE {pa:13.4f} vs {pc/n**3:13.4f}")
        dev = dyn_dev(dyn["AA0"], dyn["CG"], n)
        d_ok = dev < 1e-4 and len(dyn["AA0"]) > 0
        A = dyn["AA0"]
        T_end = A[-1][1] if A else cfg.temp
        dPE = (A[-1][2] - A[0][2]) / len(bp) if A else 0.0
        unstable = T_end > 1.2 * cfg.temp          # NVE from a perfect lattice should drop to ~T/2
        log(f"     max relative deviation = {dev:.2e}  ({'PASS' if d_ok else 'diverged'})")
        if unstable:
            log(f"     ** the random {kind.upper()} lattice is UNSTABLE with this potential: T rose "
                f"{cfg.temp:.0f} -> {T_end:.0f} K and PE changed {dPE:+.3f} eV/atom in 1 ps "
                f"(structure collapsing). Check the lattice choice (compare FCC/BCC) and the "
                f"potential/element mapping. **")
            if s_ok and not d_ok:
                log("     The static tests prove the mapping is exact; the trajectories separate "
                    "only because the instability amplifies round-off differences.")
        zbl = None
        if pair.zbl_active:
            zs = max(compare_states(st[c, "AA0"], st[c, "AA"], 1)["max"] for c in configs)
            zd = dyn_dev(dyn["AA0"], dyn["AA"], 1)
            zbl = dict(static=zs, dynamic=zd)
            log(f"     ZBL switched off (the one approximation of CG-MEAM): vs the original "
                f"potential, displaced configs dev {zs:.1e}, 1 ps MD at {cfg.temp:.0f} K dev {zd:.1e}"
                + (f"  -> no effect at {cfg.temp:.0f} K; ZBL only matters for close contacts "
                   f"(cascades, shocks, very high pressure)" if zd < 1e-6 or unstable and zs < 1e-3
                   else "  -> close contacts reach the ZBL range in this run"))
        ok = s_ok and (d_ok or unstable)
        log(f"     verdict: {'PASS - CG mapping exact' if ok else 'FAIL - mapping not exact'}")
        summary["verify"] = dict(static=static, static_dev=sdev, max_dev=dev, unstable=unstable,
                                 passed=ok, zbl_effect=zbl)
        if not ok:
            with open(os.path.join(outdir, "verification_failed.json"), "w") as f:
                json.dump(summary, f, indent=2, default=float)
            raise VerificationError(
                "the CG potential does not reproduce the atomistic one (see the [3] lines), so "
                "no CG model was written. Please report the potential file: an unsupported "
                "feature of it is probably not rescaled.")

    # 4. CG lattice + TPMS mask + random species
    S.check()
    cells = max(1, math.ceil(cfg.box_nm * 10 / a_cg - 1e-9)) if cfg.box_nm else cfg.cells
    if cfg.box_nm:
        log(f"     box {cfg.box_nm:g} nm requested -> {cells} CG cells per side (rounded up)")
    geo = build_geometry(kind, a_cg, cells, cfg.surface, cfg.periods, cfg.phase, cfg.t,
                         cfg.density, cfg.min_coord)
    if len(geo.pos) == 0:
        raise ValueError("the TPMS mask removed every bead - raise t or the target density")
    types = assign_species(len(geo.pos), comp, rng)
    L = geo.L
    pre = f"{cfg.surface}_{kind}_{tag}_cgn{n}"
    write_data(os.path.join(outdir, pre + ".data"), geo.pos, types, L, m_cg,
               f"CG n={n} {kind.upper()} {cfg.surface} {tag} composition "
               f"{[round(c, 6) for c in comp]}")
    frac = ", ".join(f"{e} {np.mean(types == i+1):.3f}" for i, e in enumerate(els))
    lev = f"|f| <= {geo.level:.4g}" if geo.phase == "sheet" else f"f <= {geo.level:.4g}"
    log(f"[4] {cfg.surface} {geo.phase} ({lev}) on a CG-{kind.upper()} lattice: box {L:.1f} Å "
        f"({L/10:.1f} nm), {len(geo.pos):,} beads (~{len(geo.pos)*n**3:,} atoms), "
        f"fill {100*geo.fill:.1f}%  [{frac}]")
    if geo.pruned:
        log(f"     removed {geo.pruned} bead(s) with < {cfg.min_coord} first neighbours")
    log(f"     surface beads {100*geo.surface_fraction:.0f}%"
        + ("  ** thin walls: raise cells or lower periods/n **" if geo.surface_fraction > 0.6
           else ""))
    log(f"     exact atomistic equivalent: same cell at {L/n/10:.1f} nm edge")
    summary.update(prefix=pre, n_beads=len(geo.pos), cells=cells, box=L, fill=geo.fill,
                   phase=geo.phase, level=geo.level, surface_fraction=geo.surface_fraction,
                   type_to_element={str(i + 1): e for i, e in enumerate(els)},
                   counts={e: int(np.sum(types == i + 1)) for i, e in enumerate(els)},
                   interpretation=f"exact similarity map: this CG cell behaves like an atomistic "
                                  f"cell of edge {L/n/10:.2f} nm with the same particle count; "
                                  f"size effects are those of that smaller cell")

    # 5. run input
    Tcg = n**3 * cfg.temp
    dt = cfg.dt_cg or n * cfg.dt_aa
    tdamp = cfg.tdamp or 100 * n * cfg.dt_aa          # time scales by n: keep the AA damping
    pdamp = cfg.pdamp or 1000 * n * cfg.dt_aa
    crush = ""
    nst = 0
    if cfg.crush:
        ax = cfg.crush_axis
        lat = [x for x in "xyz" if x != ax]
        nst = int(round(cfg.crush_strain / (cfg.strain_rate * dt)))
        if nst < 1:
            raise ValueError("compression too short for this strain rate and timestep")
        crush = f"""
# ── Stage 3: uniaxial compression along {ax} (lateral {lat[0]},{lat[1]} at 0 bar) ──
# CG rate {cfg.strain_rate:g}/ps on this cell == atomistic rate {n*cfg.strain_rate:g}/ps on the L/n cell
variable       l0 equal $(l{ax})
variable       strain equal (v_l0-l{ax})/v_l0
compute        prs all pressure thermo_temp
variable       stress equal c_prs[{'xyz'.index(ax)+1}]*1.0e-4        # GPa; LAMMPS pressure > 0 in compression (scale-invariant)
fix            ss all ave/time 20 10 200 v_strain v_stress file {pre}_stress_strain.txt
unfix          eq
fix            baro all npt temp {Tcg:.1f} {Tcg:.1f} {tdamp:.6g} {lat[0]} 0 0 {pdamp:.6g} {lat[1]} 0 0 {pdamp:.6g}
fix            comp all deform 1 {ax} erate -{cfg.strain_rate:g} remap x units box
thermo_style   custom step v_strain v_stress temp pe l{ax}
run            {nst}
"""
    dt_note = ("" if not cfg.dt_cg else
               f"\n# NOTE: dt overridden ({dt:g} ps); the exact map uses n*dt_aa = {n*cfg.dt_aa:g} ps")
    with open(os.path.join(outdir, pre + "_run.in"), "w") as fh:
        fh.write(f"""# CG n={n} {kind.upper()} {cfg.surface} {tag} (composition {[round(c, 6) for c in comp]})
# Exact CG map: thermostat T_CG = n^3 * {cfg.temp:.0f} K = {Tcg:.0f} K ; dt = n * {cfg.dt_aa} ps{dt_note}
# Atom types: {'  '.join(f'{i}={e}' for i, e in enumerate(els, 1))}
units          metal
atom_style     atomic
boundary       p p p
read_data      {pre}.data
{pair.style}
{pair.coeff('cg', els)}
neighbor       {0.1*a_cg:.3f} bin
neigh_modify   every 1 delay 0 check yes

# ── Stage 1: minimize (relative energy tol. 1e-10, or forces < 1e-6 eV/Å in physical units:
#    CG forces scale as n^2; the cap keeps rugged random-alloy surfaces from stalling) ──
min_style      cg
minimize       1.0e-10 {1e-6*n**2:.6g} 5000 50000
reset_timestep 0
timestep       {dt:g}

# ── Stage 2: NPT equilibration at physical {cfg.temp:.0f} K (periodic TPMS, 0 bar) ──
velocity       all create {Tcg:.1f} {cfg.seed} mom yes rot yes dist gaussian
fix            eq all npt temp {Tcg:.1f} {Tcg:.1f} {tdamp:.6g} iso 0.0 0.0 {pdamp:.6g}
compute        pe_t all pe/atom
variable       Tphys equal temp/{n**3}
thermo         200
thermo_style   custom step temp v_Tphys pe press lx
dump           1 all custom 2000 {pre}.lammpstrj id type x y z c_pe_t
dump_modify    1 sort id
run            {cfg.equil}
write_data     {pre}_equilibrated.data
write_restart  {pre}_equilibrated.restart
{crush}
write_data     {pre}_final.data
""")
    log(f"[5] Wrote {outdir}{os.sep}{pre}.data and {pre}_run.in "
        f"(T_CG={Tcg:.0f} K, dt={dt:g} ps, Tdamp {tdamp:.3g} ps, Pdamp {pdamp:.3g} ps"
        f"{f', crush {cfg.crush_axis} to {100*cfg.crush_strain:.0f}%' if cfg.crush else ''})")
    summary.update(run_input=pre + "_run.in", data_file=pre + ".data", T_cg=Tcg, dt_cg=dt,
                   tdamp=tdamp, pdamp=pdamp, compression_steps=nst,
                   stress_strain=(pre + "_stress_strain.txt") if cfg.crush else None,
                   outdir=outdir)
    cfg.to_json(os.path.join(outdir, pre + "_config.json"))
    with open(os.path.join(outdir, pre + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
