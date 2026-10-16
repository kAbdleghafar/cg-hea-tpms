#!/usr/bin/env python3
"""
cg_hea_gui.py — graphical interface for coarse-grained TPMS models of FCC and BCC
high- and medium-entropy alloys (engine: cg_hea_core.py).

    python3 cg_hea_gui.py          (or: python3 cg_hea_tpms.py   with no arguments)

  Step 1  attach YOUR atomistic interatomic potential (EAM/alloy, EAM/FS or MEAM)
  Step 2  choose FCC or BCC, tick the elements, set the composition
  Step 3  choose the CG level and the TPMS geometry (preview it)
  Step 4  set the MD protocol and LAMMPS, then Generate (and optionally Run)

Needs only the Python standard library + numpy; matplotlib (optional) adds the
geometry preview and the stress-strain plot.
"""
import os, platform, queue, shutil, subprocess, sys, threading, traceback
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cg_hea_core as core

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    HAVE_MPL = True
except Exception:                                   # matplotlib is optional
    HAVE_MPL = False

APP_TITLE = "CG-HEA TPMS Builder"
EAM_TYPES = [("EAM / ADP setfl (eam/alloy, eam/fs, adp)",
              "*.eam.alloy *.eam.fs *.alloy *.fs *.setfl *.adp"),
             ("All files", "*")]
MEAM_TYPES = [("MEAM files", "*.meam *.library *.lib *.params"), ("All files", "*")]
ANY_TYPES = [("Interatomic potentials", "*.eam.alloy *.eam.fs *.alloy *.fs *.setfl *.adp *.meam "
                                         "*.library *.lib *.params"), ("All files", "*")]
LIVE_SITE_LIMIT = 150_000                           # live bead count up to this many sites

GUIDE = """Quick guide
1. Potential - attach the SAME atomistic potential file(s) you would use in LAMMPS: one EAM/alloy, EAM/FS or ADP file, or a MEAM library + parameter file. It is rescaled exactly: E_CG(X) = n^3 E(X/n).
2. Alloy & lattice - choose FCC or BCC, tick the elements of your alloy and set at.%. Keep 'Relax a0' and 'Verify' on (they need LAMMPS and take seconds).
3. CG & TPMS geometry - choose the CG level n, the TPMS surface and its relative density. 'Preview geometry' shows the beads before anything is written.
4. Simulation & output - physical temperature and timestep (the CG scaling is applied for you), optional compression, LAMMPS executable, output folder.
Then press 'Generate CG model'; 'Run simulation' launches LAMMPS."""

ABOUT_CG = """Exact coarse-graining (Chalamet, Rodney, Shibuta, Comput. Mater. Sci. 228 (2023) 112306)

E_CG(X) = n^3 * E_AA(X/n): one bead stands for n^3 atoms on an n-times larger lattice.
  EAM : F and phi scaled by n^3, rho(r/n), cutoff and r-grid by n, mass by n^3
  ADP : as EAM, plus dipole u * n^(1/2) and quadrupole w * n^(-1/2)
  MEAM: Ec, delta, esub, atwt by n^3; re, alat, rc, delr by n (and repuls by n when
        erose_form = 1); ZBL switched off

A CG run at T_CG = n^3 T with timestep n*dt reproduces the atomistic trajectory of the
same structure shrunk by n. Stresses and elastic constants are unchanged, so a CG cell
of edge L behaves exactly like an atomistic cell of edge L/n.

FCC and BCC alloys are treated identically: the bead lattice has 4 (FCC) or 2 (BCC)
beads per cubic cell, and the lattice constant of the random alloy is relaxed with the
atomistic potential, then multiplied by n.  'Verify' checks the mapping with a static
test (random displacements) and 1 ps of MD, and warns if the random alloy is unstable
in the chosen lattice."""


def wrap_label(parent, text="", style=None):
    """Label whose text re-wraps to the width it is given (pack fill='x' / grid sticky='we')."""
    lbl = ttk.Label(parent, text=text, justify="left", **({"style": style} if style else {}))
    lbl.bind("<Configure>", lambda e: lbl.configure(wraplength=max(e.width - 4, 120)))
    return lbl


class Tooltip:
    def __init__(self, widget, text, wrap=380):
        self.w, self.text, self.wrap, self.tip = widget, text, wrap, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _=None):
        if self.tip or not self.text:
            return
        x = self.w.winfo_rootx() + 18
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.w)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", wraplength=self.wrap, background="#fffbe6",
                 foreground="#222", relief="solid", borderwidth=1, padx=7, pady=5).pack()

    def _hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollFrame(ttk.Frame):
    """Vertically scrollable container; the wheel is routed by App._on_wheel."""

    def __init__(self, parent, height=None):
        super().__init__(parent)
        bg = ttk.Style().lookup("TFrame", "background") or None
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=bg,
                                **({"height": height} if height else {}))
        self.canvas._scrollable = True
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"{APP_TITLE} — coarse-grained FCC/BCC high-entropy alloy lattices")
        root.geometry("1320x860")
        root.minsize(1040, 640)
        self._style()
        self.q = queue.Queue()
        self.info = None                 # PotentialInfo of the attached potential
        self.session = None
        self.summary = None              # result of the last generation
        self.busy = False
        self.a0_manual = False
        self.relaxed = None              # (alloy key, a0) relaxed by the last generation
        self.el_rows = {}                # element -> (BooleanVar use, StringVar at.%, Entry)
        self._est_job = None
        self._vars()
        self._menu()
        self._layout()
        root.bind_all("<MouseWheel>", self._on_wheel, add="+")
        root.bind_all("<Button-4>", self._on_wheel, add="+")
        root.bind_all("<Button-5>", self._on_wheel, add="+")
        root.protocol("WM_DELETE_WINDOW", self._quit)
        self._show_pair_frame()
        self._rebuild_elements()
        self._refresh()
        self.root.after(100, self._poll)
        self._log(GUIDE)
        lmp = core.find_lammps()
        if lmp:
            self.lmp.set(lmp)
            self._log(f"\nLAMMPS found: {lmp}")
        else:
            self._log("\nLAMMPS was not found on PATH: set its path in step 4 (needed for a0 "
                      "relaxation, verification and running; files can be generated without it).")
        if not HAVE_MPL:
            self._log("matplotlib not installed: geometry preview and plots are disabled "
                      "(pip install matplotlib).")
        self._status("Step 1: attach your interatomic potential.")

    # ───────────────────────── setup ─────────────────────────
    def _style(self):
        st = ttk.Style()
        if platform.system() == "Linux" and "clam" in st.theme_names():
            st.theme_use("clam")
        base = tkfont.nametofont("TkDefaultFont")
        fam, size = base.actual("family"), base.actual("size")
        self.f_bold = (fam, size, "bold")
        self.f_title = (fam, size + 3, "bold")
        st.configure("Title.TLabel", font=self.f_title)
        st.configure("Hint.TLabel", foreground="#666")
        st.configure("Good.TLabel", foreground="#1b7a2f")
        st.configure("Bad.TLabel", foreground="#b3261e")
        st.configure("Warn.TLabel", foreground="#9a6700")
        st.configure("Key.TLabel", foreground="#555")
        st.configure("Val.TLabel", font=self.f_bold)
        st.configure("Accent.TButton", font=self.f_bold, padding=(10, 6))
        st.configure("TLabelframe.Label", font=self.f_bold)

    def _vars(self):
        V, B, S = tk.StringVar, tk.BooleanVar, tk.StringVar
        self.pair = V(value="eam")
        self.pot_path, self.lib_path, self.par_path, self.lib_order = S(), S(), S(), S()
        self.el_filter = S()
        self.lattice = V(value="fcc")
        self.a0 = S(value="")
        self.auto_a0, self.verify = B(value=True), B(value=True)
        self.check_pure, self.compare = B(value=False), B(value=False)
        self.n = S(value="5")
        self.surface = S(value="gyroid")
        self.cells, self.periods = S(value="20"), S(value="1")
        self.box_mode, self.box_nm = S(value="cells"), S(value="30")
        self.matrix_mode, self.matrix_path, self.matrix_types = S(value="lattice"), S(), S()
        self.matrix_random = B(value=False)
        self.dt_cg, self.tdamp, self.pdamp = S(), S(), S()
        self.phase = S(value="network")
        self.thick_mode = S(value="density")
        self.t = S(value="0.0")
        self.density = S(value="50")
        self.min_coord = S(value="0")
        self.temp, self.dt = S(value="300"), S(value="0.001")
        self.equil = S(value="20000")
        self.crush = B(value=False)
        self.crush_axis, self.crush_strain, self.rate = S(value="z"), S(value="0.5"), S(value="1e-4")
        self.seed = S(value="2024")
        self.lmp, self.mpi = S(value="lmp"), S(value="")
        self.outdir = S(value=os.path.abspath("cg_hea_output"))
        for v in (self.lattice, self.n, self.cells, self.box_mode, self.box_nm, self.dt_cg,
                  self.matrix_mode, self.matrix_path,
                  self.periods, self.phase, self.thick_mode,
                  self.t, self.density, self.min_coord, self.temp, self.dt, self.a0, self.surface):
            v.trace_add("write", lambda *a: self._refresh())
        self.lattice.trace_add("write", lambda *a: self._update_a0())

    def _menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Attach potential…", command=self._attach, accelerator="Ctrl+O")
        f.add_separator()
        f.add_command(label="Load settings…", command=self._load_settings)
        f.add_command(label="Save settings…", command=self._save_settings)
        f.add_command(label="Copy equivalent command line", command=self._copy_cli)
        f.add_separator()
        f.add_command(label="Quit", command=self._quit)
        m.add_cascade(label="File", menu=f)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Quick guide", command=lambda: messagebox.showinfo("Quick guide", GUIDE))
        h.add_command(label="About the CG mapping",
                      command=lambda: messagebox.showinfo("CG mapping", ABOUT_CG))
        m.add_cascade(label="Help", menu=h)
        self.root.config(menu=m)
        self.root.bind_all("<Control-o>", lambda e: self._attach())

    def _layout(self):
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=3)
        pw.add(right, weight=2)
        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)
        tabs = []
        for title in ("① Potential", "② Alloy & lattice", "③ CG & TPMS geometry",
                      "④ Simulation & output"):
            sf = ScrollFrame(self.nb)
            self.nb.add(sf, text=f"  {title}  ")
            tabs.append(sf.inner)
        self._tab_potential(tabs[0])
        self._tab_alloy(tabs[1])
        self._tab_geometry(tabs[2])
        self._tab_simulation(tabs[3])
        self._right(right)
        self.statusbar = ttk.Label(self.root, anchor="w", padding=(10, 4), relief="sunken")
        self.statusbar.pack(fill="x", side="bottom", padx=8, pady=(4, 8))

    # small widget helpers
    def _row(self, parent, r, label, var, width=10, unit="", tip="", col=0):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=r, column=col, sticky="w", padx=(4, 8), pady=3)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=r, column=col + 1, sticky="w", pady=3)
        if unit:
            ttk.Label(parent, text=unit, style="Hint.TLabel").grid(row=r, column=col + 2,
                                                                  sticky="w", padx=4)
        if tip:
            Tooltip(lab, tip)
            Tooltip(e, tip)
        return e

    def _next(self, parent, idx, text):
        f = ttk.Frame(parent)
        f.pack(fill="x", padx=10, pady=(6, 14))
        ttk.Button(f, text=text, command=lambda: self.nb.select(idx)).pack(side="right")

    # ───────────────────────── tab 1: potential ─────────────────────────
    def _tab_potential(self, p):
        ttk.Label(p, text="Step 1 — Attach your atomistic interatomic potential",
                  style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 2))
        wrap_label(p, "Use the same file(s) you would give LAMMPS for an atomistic run. The "
                      "potential is rescaled exactly, E_CG(X) = n³·E(X/n), so any EAM, ADP or MEAM "
                      "parameterisation of your alloy works, FCC or BCC.", "Hint.TLabel").pack(
            fill="x", padx=10, pady=(0, 8))

        top = ttk.Frame(p)
        top.pack(fill="x", padx=10)
        b = ttk.Button(top, text="Attach potential file(s)…", style="Accent.TButton",
                       command=self._attach)
        b.pack(side="left")
        Tooltip(b, "Select one EAM or ADP file (eam/alloy, eam/fs, adp), or the MEAM library AND parameter "
                   "file together (Ctrl/Shift-click). The format is detected automatically.")
        self.detected = ttk.Label(top, text="", style="Hint.TLabel")
        self.detected.pack(side="left", padx=12)

        fmt = ttk.LabelFrame(p, text="Potential format", padding=8)
        fmt.pack(fill="x", padx=10, pady=8)
        for i, (k, lab) in enumerate(core.PAIR_LABELS.items()):
            ttk.Radiobutton(fmt, text=lab, value=k, variable=self.pair,
                            command=self._pair_changed).grid(row=i, column=0, sticky="w", padx=6,
                                                             pady=1)

        self.eam_frame = ttk.LabelFrame(p, text="EAM / ADP potential file", padding=8)
        ttk.Label(self.eam_frame, text="File").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(self.eam_frame, textvariable=self.pot_path, width=64).grid(
            row=0, column=1, sticky="we", padx=4)
        ttk.Button(self.eam_frame, text="Browse…",
                   command=lambda: self._browse(self.pot_path, EAM_TYPES)).grid(row=0, column=2)
        self.eam_frame.columnconfigure(1, weight=1)

        self.meam_frame = ttk.LabelFrame(p, text="MEAM potential files", padding=8)
        mf = self.meam_frame
        ttk.Label(mf, text="Library file").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(mf, textvariable=self.lib_path, width=60).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(mf, text="Browse…", command=lambda: self._browse(self.lib_path, MEAM_TYPES)
                   ).grid(row=0, column=2)
        ttk.Label(mf, text="Parameter file").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(mf, textvariable=self.par_path, width=60).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(mf, text="Browse…", command=lambda: self._browse(self.par_path, MEAM_TYPES)
                   ).grid(row=1, column=2)
        wrap_label(mf, "optional — leave empty if the potential uses only the library",
                   "Hint.TLabel").grid(row=2, column=1, sticky="we", padx=4)
        lab = ttk.Label(mf, text="Library element order")
        lab.grid(row=3, column=0, sticky="w", padx=4, pady=(8, 2))
        e = ttk.Entry(mf, textvariable=self.lib_order, width=60)
        e.grid(row=3, column=1, sticky="we", padx=4, pady=(8, 2))
        tip = ("The element list that follows the library file in the potential's ORIGINAL "
               "pair_coeff line, e.g. 'Co Ni Cr Fe Mn'. Indices (i,j) in the parameter file refer "
               "to this order, so take it from the potential's documentation. Leave empty to use "
               "the library order (dedicated library) or the alloy elements.")
        Tooltip(lab, tip)
        Tooltip(e, tip)
        wrap_label(mf, "must match the potential's documentation — a wrong order is NOT "
                       "detectable by the checks", "Warn.TLabel").grid(row=4, column=1, sticky="we",
                                                                      padx=4)
        mf.columnconfigure(1, weight=1)

        self.pot_status = wrap_label(p, "No potential attached yet.", "Hint.TLabel")
        self.pot_status.pack(fill="x", padx=12, pady=(4, 4))
        btns = ttk.Frame(p)
        btns.pack(fill="x", padx=10)
        ttk.Button(btns, text="Reload / check files", command=self._load_potential).pack(side="left")

        tf = ttk.LabelFrame(p, text="Elements in the attached potential", padding=8)
        tf.pack(fill="both", expand=True, padx=10, pady=8)
        cols = ("el", "mass", "lat", "a0", "vol")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=9)
        for c, h, w in zip(cols, ("Element", "Mass (amu)", "Header ref. structure",
                                  "Header a0 (Å)", "Atomic volume (Å³)"), (90, 110, 160, 110, 140)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        wrap_label(p, "Header lattice data are only used for the first a0 guess; they are "
                      "sometimes mislabelled, which is why 'Relax a0' scans E(a) first.",
                   "Hint.TLabel").pack(fill="x", padx=12)
        self._next(p, 1, "Next: alloy & lattice  →")

    def _show_pair_frame(self):
        meam = self.pair.get() == "meam"
        (self.eam_frame.pack_forget if meam else self.meam_frame.pack_forget)()
        frame = self.meam_frame if meam else self.eam_frame
        frame.pack(fill="x", padx=10, pady=4, before=self.pot_status)

    def _pair_changed(self):
        self._show_pair_frame()
        if self.pot_path.get() or self.lib_path.get():
            self._load_potential()

    def _browse(self, var, types):
        path = filedialog.askopenfilename(title="Select potential file", filetypes=types)
        if path:
            var.set(path)
            if var is self.pot_path:
                kind = core.detect_pair_style(path)
                if kind in core.SETFL_PAIRS:
                    self.pair.set(kind)
                elif kind == "meam":
                    self._route_files([path])
                    return
            self._show_pair_frame()
            self._load_potential()

    def _attach(self):
        paths = filedialog.askopenfilenames(
            title="Attach your interatomic potential (one EAM file, or MEAM library + parameter file)",
            filetypes=ANY_TYPES)
        if paths:
            self._route_files(list(paths))

    def _route_files(self, paths):
        lib = par = setfl = None
        for pth in paths:
            kind = core.detect_pair_style(pth)
            if kind == "meam":
                if core.classify_meam_file(pth) == "params":
                    par = pth
                else:
                    lib = pth
            elif kind in core.SETFL_PAIRS:
                setfl = (pth, kind)
            else:
                messagebox.showerror("Unsupported file",
                                     f"{os.path.basename(pth)} is not an EAM/ADP setfl (eam/alloy, "
                                     f"eam/fs, adp) or MEAM file.")
                return
        if lib or par:
            self.pair.set("meam")
            if lib:
                self.lib_path.set(lib)
            if par:
                self.par_path.set(par)
            msg = "MEAM " + " + ".join(x for x in ("library" if lib else "",
                                                   "parameter file" if par else "") if x)
        else:
            self.pair.set(setfl[1])
            self.pot_path.set(setfl[0])
            msg = core.PAIR_LABELS[setfl[1]].split("  ")[0]
        self.detected.config(text=f"detected: {msg}")
        self._show_pair_frame()
        self._load_potential()

    def _load_potential(self):
        get = lambda v: v.get().strip() or None
        try:
            info = core.load_potential(self.pair.get(), get(self.pot_path), get(self.lib_path),
                                       get(self.par_path))
        except core.PotentialError as e:
            self.info = None
            self.pot_status.config(text=f"✗  {e}", style="Bad.TLabel")
            self.tree.delete(*self.tree.get_children())
            self._rebuild_elements()
            self._status("Step 1: the potential could not be loaded — see the message in tab ①.")
            return False
        self.info = info
        extra = "".join(f"\n   note: {x}" for x in info.notes)
        if info.pair == "meam" and info.param_index_max:
            extra += (f"\n   parameter file refers to element indices up to "
                      f"{info.param_index_max}")
        files = "   +   ".join(os.path.basename(f) for f in info.files)
        self.pot_status.config(text=f"✓  {files}\n   {info.describe()}{extra}",
                               style="Good.TLabel")
        self.tree.delete(*self.tree.get_children())
        for el in info.elements:
            lat, a = info.ref[el]
            vol = core.atomic_volume(lat, a)
            self.tree.insert("", "end", values=(el, f"{info.masses[el]:.4f}", core.unquote(lat),
                                                f"{float(a):.4f}", f"{vol:.3f}" if vol else "—"))
        if info.pair == "meam" and not self.lib_order.get().strip() and \
                info.param_index_max and len(info.elements) == info.param_index_max:
            self.lib_order.set(" ".join(info.elements))
        self._rebuild_elements()
        self._log(f"Potential loaded: {', '.join(info.files)}\n   {info.describe()}")
        self._status("Potential loaded. Step 2: choose FCC/BCC and the alloy composition.")
        return True

    # ───────────────────────── tab 2: alloy ─────────────────────────
    def _tab_alloy(self, p):
        ttk.Label(p, text="Step 2 — Crystal structure and alloy composition",
                  style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 6))
        lf = ttk.LabelFrame(p, text="Crystal structure of the alloy", padding=8)
        lf.pack(fill="x", padx=10, pady=4)
        for r, (k, txt, eg) in enumerate((
                ("fcc", "FCC — face-centred cubic (4 beads per cubic cell)",
                 "e.g. CoCrFeMnNi (Cantor), CoCrNi, FeNiCoCu, Al0.1CoCrFeNi"),
                ("bcc", "BCC — body-centred cubic (2 beads per cubic cell)",
                 "e.g. NbMoTaW, NbMoTaWV, HfNbTaTiZr, TiZrNbHfTa, AlCoCrFeNi (disordered A2)"))):
            ttk.Radiobutton(lf, text=txt, value=k, variable=self.lattice).grid(
                row=2*r, column=0, sticky="w", padx=4, pady=(4, 0))
            wrap_label(lf, eg, "Hint.TLabel").grid(row=2*r+1, column=0, sticky="we", padx=28)
        lf.columnconfigure(0, weight=1)
        wrap_label(lf, "Chemical order is random (solid solution). Not sure which lattice your "
                       "potential prefers? Tick 'Compare FCC vs BCC' below.", "Hint.TLabel").grid(
            row=4, column=0, sticky="we", padx=4, pady=(8, 0))

        cf = ttk.LabelFrame(p, text="Composition — tick the elements of your alloy (at.%)",
                            padding=8)
        cf.pack(fill="x", padx=10, pady=8)
        ff = ttk.Frame(cf)
        ff.pack(fill="x", pady=(0, 4))
        ttk.Label(ff, text="Filter").pack(side="left", padx=(2, 6))
        fe = ttk.Entry(ff, textvariable=self.el_filter, width=16)
        fe.pack(side="left")
        Tooltip(fe, "Show only elements whose name contains this text (ticked elements always "
                    "stay visible) - handy for large MEAM libraries.")
        ttk.Button(ff, text="×", width=2, command=lambda: self.el_filter.set("")).pack(
            side="left", padx=2)
        self.el_filter.trace_add("write", lambda *a: self._filter_elements())
        self.el_scroll = ScrollFrame(cf, height=190)
        self.el_scroll.pack(fill="x")
        bf = ttk.Frame(cf)
        bf.pack(fill="x", pady=(6, 0))
        ttk.Button(bf, text="Equiatomic", command=self._equiatomic).pack(side="left")
        ttk.Button(bf, text="Normalise to 100 %", command=self._normalise).pack(side="left", padx=6)
        ttk.Button(bf, text="Clear", command=self._clear_elements).pack(side="left")
        self.comp_info = wrap_label(cf, "", "Val.TLabel")
        self.comp_info.pack(fill="x", pady=(8, 0))
        self.types_info = wrap_label(cf, "", "Hint.TLabel")
        self.types_info.pack(fill="x")

        af = ttk.LabelFrame(p, text="Lattice constant (atomistic)", padding=8)
        af.pack(fill="x", padx=10, pady=4)
        ttk.Label(af, text="a0 (guess)").grid(row=0, column=0, sticky="w", padx=4)
        e = ttk.Entry(af, textvariable=self.a0, width=10)
        e.grid(row=0, column=1, sticky="w")
        e.bind("<Key>", lambda ev: setattr(self, "a0_manual", True))
        ttk.Label(af, text="Å").grid(row=0, column=2, sticky="w", padx=4)
        b = ttk.Button(af, text="Estimate from potential", command=self._reset_a0)
        b.grid(row=0, column=3, padx=8)
        Tooltip(b, "Vegard's law on the atomic volumes of the reference structures in the "
                   "potential header, converted to the chosen lattice.")
        self.a0_src = ttk.Label(af, text="", style="Hint.TLabel")
        self.a0_src.grid(row=0, column=4, sticky="w")
        c = ttk.Checkbutton(af, text="Relax a0 of the random alloy with LAMMPS (recommended)",
                            variable=self.auto_a0)
        c.grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 0))
        Tooltip(c, "Coarse E(a) scan (robust to a poor guess) followed by box/relax of a "
                   "random-alloy supercell with the ATOMISTIC potential; a_CG = n·a0 exactly.")

        kf = ttk.LabelFrame(p, text="Checks (run with LAMMPS, a few seconds to a minute)",
                            padding=8)
        kf.pack(fill="x", padx=10, pady=8)
        for r, (var, txt, tip) in enumerate((
                (self.verify, "Verify the exact CG mapping (AA vs CG: static test + 1 ps MD)",
                 "Same random alloy at T and at n³T: energies, pressure and forces must map "
                 "exactly. Also flags a random alloy that is unstable in the chosen lattice."),
                (self.compare, "Compare FCC vs BCC energy of the random alloy",
                 "Relaxes the random alloy in both lattices and reports E_BCC − E_FCC "
                 "(0 K, one random configuration — indicative only)."),
                (self.check_pure, "Pure-element check (each element relaxed as FCC and BCC)",
                 "Confirms each atom TYPE is the intended element by comparing with known "
                 "lattice constants / ground states. Cannot detect a wrong MEAM element order."))):
            cb = ttk.Checkbutton(kf, text=txt, variable=var)
            cb.grid(row=r, column=0, sticky="w", pady=2)
            Tooltip(cb, tip)
        self._next(p, 2, "Next: CG & TPMS geometry  →")

    def _rebuild_elements(self, keep=None):
        inner = self.el_scroll.inner
        for w in inner.winfo_children():
            w.destroy()
        self.el_rows = {}
        self._el_widgets = {}
        if not self.info:
            ttk.Label(inner, text="Attach a potential in step ① to list its elements.",
                      style="Hint.TLabel").grid(row=0, column=0, padx=6, pady=6)
            self._comp_changed()
            return
        for c, h in enumerate(("Use", "Element", "at.%", "Mass (amu)", "Header structure")):
            ttk.Label(inner, text=h, style="Key.TLabel").grid(row=0, column=c, padx=8, sticky="w")
        els = self.info.elements
        pre = len(els) <= 5 and keep is None
        for i, el in enumerate(els, 1):
            use = tk.BooleanVar(value=pre or bool(keep and el in keep))
            pct = tk.StringVar(value=(f"{100/len(els):.6g}" if pre else
                                      f"{keep[el]:.6g}" if keep and el in keep else ""))
            ent = ttk.Entry(inner, textvariable=pct, width=8)
            lat, a = self.info.ref[el]
            row = [ttk.Checkbutton(inner, variable=use, command=lambda e=el: self._toggle(e)),
                   ttk.Label(inner, text=el, style="Val.TLabel"), ent,
                   ttk.Label(inner, text=f"{self.info.masses[el]:.3f}"),
                   ttk.Label(inner, text=f"{core.unquote(lat)}  {float(a):.4f} Å",
                             style="Hint.TLabel")]
            for c, w in enumerate(row):
                w.grid(row=i, column=c, padx=8, pady=1, sticky="w" if c in (1, 4) else "")
            pct.trace_add("write", lambda *a: self._comp_changed())
            self.el_rows[el] = (use, pct, ent)
            self._el_widgets[el] = row
            ent.configure(state="normal" if use.get() else "disabled")
        self._filter_elements()
        self._comp_changed()

    def _filter_elements(self):
        f = self.el_filter.get().strip().lower()
        for el, row in getattr(self, "_el_widgets", {}).items():
            show = not f or f in el.lower() or self.el_rows[el][0].get()
            for w in row:
                (w.grid if show else w.grid_remove)()
        self.el_scroll.canvas.yview_moveto(0)

    def _toggle(self, el):
        use, pct, ent = self.el_rows[el]
        ent.configure(state="normal" if use.get() else "disabled")
        others = {self.el_rows[e][1].get().strip() for e in self._checked() if e != el}
        if len(others) <= 1:                       # composition still uniform -> stay equiatomic
            self._equiatomic()
        elif not use.get():
            pct.set("")
        self._comp_changed()

    def _checked(self):
        return [e for e, (u, _, _) in self.el_rows.items() if u.get()]

    def _equiatomic(self):
        sel = self._checked()
        for e, (u, p, _) in self.el_rows.items():
            p.set(f"{100/len(sel):.6g}" if u.get() else "")

    def _normalise(self):
        try:
            els, comp = self._composition()
        except ValueError as e:
            messagebox.showerror("Composition", str(e))
            return
        for e, c in zip(els, comp):
            self.el_rows[e][1].set(f"{100*c:.6g}")

    def _clear_elements(self):
        for e, (u, p, ent) in self.el_rows.items():
            u.set(False)
            p.set("")
            ent.configure(state="disabled")
        self._comp_changed()

    def _composition(self):
        """Selected elements (potential order) and normalised fractions."""
        els, vals = [], []
        for e, (u, p, _) in self.el_rows.items():
            if u.get():
                try:
                    v = float(p.get())
                except ValueError:
                    raise ValueError(f"enter the at.% of {e}")
                if v <= 0:
                    raise ValueError(f"the at.% of {e} must be positive (untick it to remove it)")
                els.append(e)
                vals.append(v)
        if not els:
            raise ValueError("tick at least one element")
        s = sum(vals)
        return els, [v / s for v in vals]

    def _comp_changed(self):
        try:
            els, comp = self._composition()
            raw = sum(float(self.el_rows[e][1].get()) for e in els)
        except ValueError as e:
            self.comp_info.config(text=f"Composition: {e}", style="Warn.TLabel")
            self.types_info.config(text="")
            self._update_a0()
            self._refresh()
            return
        s, label, npr = core.entropy_class(comp)
        name = "".join(f"{e}{100*c:.3g}" if abs(c - 1/len(els)) > 1e-9 else e
                       for e, c in zip(els, comp))
        warn = "" if abs(raw - 100) < 0.05 else f"   (sum {raw:.4g} % → normalised to 100 %)"
        self.comp_info.config(
            text=f"{name}:  ΔS_conf = {s:.3f} R  →  {label}, {npr} principal element(s) "
                 f"(5–35 at.%){warn}", style="Val.TLabel")
        self.types_info.config(text="LAMMPS atom types:  " + "   ".join(
            f"{i} = {e}" for i, e in enumerate(els, 1)))
        self._update_a0()
        self._refresh()

    def _vegard(self):
        if not self.info:
            return None
        try:
            els, comp = self._composition()
        except ValueError:
            return None
        return core.vegard_a0(self.info.ref, els, comp, self.lattice.get())

    def _alloy_key(self):
        try:
            els, comp = self._composition()
        except ValueError:
            return None
        return (self.lattice.get(), tuple(els), tuple(round(c, 6) for c in comp),
                tuple(self.info.files) if self.info else None)

    def _update_a0(self):
        if self.a0_manual:
            self.a0_src.config(text="(your value)")
            return
        if self.relaxed and self.relaxed[0] == self._alloy_key():
            self.a0.set(f"{self.relaxed[1]:.4f}")
            self.a0_src.config(text="(relaxed with LAMMPS)")
            return
        v = self._vegard()
        if v:
            self.a0.set(f"{v:.4f}")
            self.a0_src.config(text="(Vegard estimate)")
        else:
            self.a0.set(f"{core.DEFAULT_A0[self.lattice.get()]:.2f}")
            self.a0_src.config(text="(generic default)")

    def _reset_a0(self):
        self.a0_manual = False
        self._update_a0()

    # ───────────────────────── tab 3: geometry ─────────────────────────
    def _tab_geometry(self, p):
        ttk.Label(p, text="Step 3 — Coarse-graining level and TPMS geometry",
                  style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 6))
        cg = ttk.LabelFrame(p, text="Coarse-graining", padding=8)
        cg.pack(fill="x", padx=10, pady=4)
        ttk.Label(cg, text="CG level n").grid(row=0, column=0, sticky="w", padx=4)
        sp = ttk.Spinbox(cg, from_=1, to=20, textvariable=self.n, width=6)
        sp.grid(row=0, column=1, sticky="w")
        Tooltip(sp, "One bead represents n³ atoms; the lattice constant and all lengths of the "
                    "potential are multiplied by n. n = 1 gives a plain atomistic model.")
        self.n_info = ttk.Label(cg, text="", style="Hint.TLabel")
        self.n_info.grid(row=0, column=2, sticky="w", padx=10)

        mf = ttk.LabelFrame(p, text="Solid to carve the TPMS from", padding=8)
        mf.pack(fill="x", padx=10, pady=4)
        ttk.Radiobutton(mf, text="generated single crystal (FCC/BCC of step ②, random alloy)",
                        value="lattice", variable=self.matrix_mode).grid(row=0, column=0,
                                                                          columnspan=3, sticky="w")
        ttk.Radiobutton(mf, text="existing structure: LAMMPS data file (e.g. your polycrystal)",
                        value="file", variable=self.matrix_mode).grid(row=1, column=0,
                                                                       columnspan=3, sticky="w")
        ttk.Entry(mf, textvariable=self.matrix_path, width=50).grid(row=2, column=0, sticky="we",
                                                                    padx=(22, 4), pady=2)
        ttk.Button(mf, text="Browse…", command=self._browse_matrix).grid(row=2, column=1)
        c = ttk.Checkbutton(mf, text="Re-assign species at random with the composition of step ② "
                                     "(for a pure template, e.g. an Atomsk polycrystal)",
                            variable=self.matrix_random, command=self._refresh)
        c.grid(row=5, column=0, columnspan=2, sticky="w", padx=(22, 4), pady=(4, 0))
        Tooltip(c, "Alloys the whole structure first, then carves it, so the bulk reference and "
                   "every lattice share one species arrangement. The file's own types are ignored.")
        lab = ttk.Label(mf, text="Atom types 1, 2, … in the file are")
        lab.grid(row=3, column=0, sticky="w", padx=(22, 4), pady=2)
        e = ttk.Entry(mf, textvariable=self.matrix_types, width=24)
        e.grid(row=3, column=1, sticky="w", pady=2)
        tip = ("Element of each atom type in your data file, in type order, e.g. 'Ti Zr Nb'. "
               "Must be the elements ticked in step ②. Empty = the order shown in step ②.")
        Tooltip(lab, tip)
        Tooltip(e, tip)
        wrap_label(mf, "atom_style atomic, orthogonal box. The file is scaled by n (n = 1 keeps "
                       "it atomistic) and the TPMS is cut from it, so grains, boundaries and "
                       "composition are exactly those of your bulk reference.",
                   "Hint.TLabel").grid(row=4, column=0, columnspan=2, sticky="we", padx=(22, 4))
        mf.columnconfigure(0, weight=1)

        g = ttk.LabelFrame(p, text="TPMS geometry", padding=8)
        g.pack(fill="x", padx=10, pady=8)
        ttk.Label(g, text="Surface").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        cb = ttk.Combobox(g, textvariable=self.surface, values=sorted(core.SURFACES),
                          state="readonly", width=12)
        cb.grid(row=0, column=1, sticky="w")
        lab = ttk.Label(g, text="Box size")
        lab.grid(row=1, column=0, sticky="w", padx=4, pady=3)
        bx = ttk.Frame(g)
        bx.grid(row=1, column=1, columnspan=3, sticky="w")
        ttk.Radiobutton(bx, value="cells", variable=self.box_mode).pack(side="left")
        ttk.Entry(bx, textvariable=self.cells, width=6).pack(side="left")
        ttk.Label(bx, text="CG cells per side").pack(side="left", padx=(4, 16))
        ttk.Radiobutton(bx, value="nm", variable=self.box_mode).pack(side="left")
        ttk.Entry(bx, textvariable=self.box_nm, width=6).pack(side="left")
        ttk.Label(bx, text="nm box edge (rounded up to whole cells)").pack(side="left", padx=4)
        Tooltip(lab, "Either the number of cubic CG unit cells along each edge (L = cells·n·a0) "
                     "or the box edge in nm; the nm value is rounded up to whole cells after a0 "
                     "has been relaxed.")
        self._row(g, 2, "TPMS periods per box side", self.periods, 8, "",
                  "Number of TPMS unit cells along each edge (the structure is periodic).")
        ttk.Label(g, text="Solid phase").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        ph = ttk.Frame(g)
        ph.grid(row=3, column=1, columnspan=3, sticky="w")
        ttk.Radiobutton(ph, text="network  (f ≤ t)", value="network", variable=self.phase).pack(
            side="left")
        ttk.Radiobutton(ph, text="sheet  (|f| ≤ t)", value="sheet", variable=self.phase).pack(
            side="left", padx=12)
        ttk.Label(g, text="Thickness set by").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        th = ttk.Frame(g)
        th.grid(row=4, column=1, columnspan=3, sticky="w")
        ttk.Radiobutton(th, text="target relative density", value="density",
                        variable=self.thick_mode).pack(side="left")
        ttk.Entry(th, textvariable=self.density, width=7).pack(side="left", padx=4)
        ttk.Label(th, text="%").pack(side="left")
        ttk.Radiobutton(th, text="level-set value t", value="level",
                        variable=self.thick_mode).pack(side="left", padx=(18, 0))
        ttk.Entry(th, textvariable=self.t, width=7).pack(side="left", padx=4)
        self._row(g, 5, "Remove beads with fewer neighbours than", self.min_coord, 6,
                  "(0 = keep all)",
                  "Iteratively deletes dangling beads with fewer occupied first neighbours "
                  "(FCC has 12, BCC 8). 3–4 cleans up single-bead protrusions.")
        bf = ttk.Frame(p)
        bf.pack(fill="x", padx=10, pady=4)
        b = ttk.Button(bf, text="Preview geometry", command=self._preview)
        b.pack(side="left")
        if not HAVE_MPL:
            Tooltip(b, "matplotlib is not installed: the preview reports numbers only.")
        self.geo_info = wrap_label(p, "", "Hint.TLabel")
        self.geo_info.pack(fill="x", padx=12, pady=4)
        self._next(p, 3, "Next: simulation & output  →")

    # ───────────────────────── tab 4: simulation ─────────────────────────
    def _tab_simulation(self, p):
        ttk.Label(p, text="Step 4 — MD protocol, LAMMPS and output",
                  style="Title.TLabel").pack(anchor="w", padx=10, pady=(10, 2))
        wrap_label(p, "Enter PHYSICAL values. The CG thermostat (n³·T) and timestep (n·dt) are "
                      "applied automatically.", "Hint.TLabel").pack(fill="x", padx=10, pady=(0, 6))
        md = ttk.LabelFrame(p, text="MD protocol", padding=8)
        md.pack(fill="x", padx=10, pady=4)
        self._row(md, 0, "Temperature", self.temp, 10, "K (physical)")
        self._row(md, 1, "Atomistic timestep", self.dt, 10, "ps  (CG uses n·dt)")
        self._row(md, 2, "NPT equilibration", self.equil, 10, "steps")
        adv = ttk.LabelFrame(p, text="Advanced (leave empty for the exact-map defaults)",
                             padding=8)
        adv.pack(fill="x", padx=10, pady=4, after=md)
        self._row(adv, 0, "CG timestep", self.dt_cg, 10, "ps  (default n·dt — exact map)",
                  "Override the CG timestep. The exact map uses n·dt; a larger value is faster "
                  "but no longer reproduces the atomistic trajectory.")
        self._row(adv, 1, "Thermostat damping", self.tdamp, 10, "ps  (default 100·n·dt)",
                  "Nose-Hoover temperature damping time of the CG run. Times scale by n, so the "
                  "default equals the usual 100·dt of the atomistic run.")
        self._row(adv, 2, "Barostat damping", self.pdamp, 10, "ps  (default 1000·n·dt)")
        self._row(md, 3, "Random seed", self.seed, 10, "",
                  "Seeds the species assignment and the velocities (reproducible models).")
        c = ttk.Checkbutton(md, text="Add uniaxial compression after equilibration",
                            variable=self.crush, command=self._crush_state)
        c.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 2))
        self.crush_frame = ttk.Frame(md)
        self.crush_frame.grid(row=5, column=0, columnspan=4, sticky="w", padx=20)
        ttk.Label(self.crush_frame, text="axis").grid(row=0, column=0, padx=4)
        ttk.Combobox(self.crush_frame, textvariable=self.crush_axis, values=("x", "y", "z"),
                     width=3, state="readonly").grid(row=0, column=1)
        ttk.Label(self.crush_frame, text="final strain").grid(row=0, column=2, padx=(14, 4))
        ttk.Entry(self.crush_frame, textvariable=self.crush_strain, width=7).grid(row=0, column=3)
        ttk.Label(self.crush_frame, text="strain rate").grid(row=0, column=4, padx=(14, 4))
        ttk.Entry(self.crush_frame, textvariable=self.rate, width=8).grid(row=0, column=5)
        ttk.Label(self.crush_frame, text="1/ps (CG)", style="Hint.TLabel").grid(row=0, column=6,
                                                                                padx=4)
        self._crush_state()

        lf = ttk.LabelFrame(p, text="LAMMPS", padding=8)
        lf.pack(fill="x", padx=10, pady=8)
        ttk.Label(lf, text="Executable").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(lf, textvariable=self.lmp, width=46).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(lf, text="Browse…", command=self._browse_lmp).grid(row=0, column=2)
        ttk.Button(lf, text="Detect", command=self._detect_lmp).grid(row=0, column=3, padx=4)
        lab = ttk.Label(lf, text="MPI prefix for the run")
        lab.grid(row=1, column=0, sticky="w", padx=4, pady=(6, 0))
        e = ttk.Entry(lf, textvariable=self.mpi, width=46)
        e.grid(row=1, column=1, sticky="we", padx=4, pady=(6, 0))
        tip = ("Optional, used by 'Run simulation' only, e.g. 'mpirun -np 8'. The short checks "
               "always run serially.")
        Tooltip(lab, tip)
        Tooltip(e, tip)
        wrap_label(lf, "Needs pair styles eam/alloy, eam/fs, adp (MANYBODY package) or meam (MEAM "
                       "package). 'pip install lammps' provides a suitable 'lmp'.",
                   "Hint.TLabel").grid(row=2, column=0, columnspan=4, sticky="we", padx=4,
                                       pady=(6, 0))
        lf.columnconfigure(1, weight=1)

        of = ttk.LabelFrame(p, text="Output folder", padding=8)
        of.pack(fill="x", padx=10, pady=4)
        ttk.Entry(of, textvariable=self.outdir, width=60).grid(row=0, column=0, sticky="we", padx=4)
        ttk.Button(of, text="Browse…", command=self._browse_outdir).grid(row=0, column=1)
        of.columnconfigure(0, weight=1)
        wrap_label(p, "Written there: the rescaled potential, a copy of yours, the CG data "
                      "file, the LAMMPS run input, settings (.json) and a summary (.json). "
                      "Check runs go to the 'checks' sub-folder.", "Hint.TLabel").pack(
            fill="x", padx=12, pady=6)

    def _crush_state(self):
        state = "normal" if self.crush.get() else "disabled"
        for w in self.crush_frame.winfo_children():
            if isinstance(w, ttk.Combobox):
                w.configure(state="readonly" if self.crush.get() else "disabled")
            elif isinstance(w, ttk.Entry):
                w.configure(state=state)

    def _browse_lmp(self):
        path = filedialog.askopenfilename(title="LAMMPS executable")
        if path:
            self.lmp.set(path)

    def _detect_lmp(self):
        p = core.find_lammps()
        if p:
            self.lmp.set(p)
            self._log(f"LAMMPS found: {p}")
        else:
            messagebox.showwarning("LAMMPS", "No LAMMPS executable (lmp, lmp_serial, lmp_mpi…) "
                                             "found on PATH. Use Browse… to select it.")

    def _browse_matrix(self):
        path = filedialog.askopenfilename(title="Structure to carve from (LAMMPS data file)",
                                          filetypes=[("LAMMPS data", "*.data *.lmp *.dat"),
                                                     ("All files", "*")])
        if path:
            self.matrix_path.set(path)
            self.matrix_mode.set("file")
            try:
                m = self._matrix()
                self._log(f"Matrix loaded: {m['name']}, {len(m['pos']):,} atoms, {m['ntypes']} "
                          f"atom types, box {' x '.join(f'{x:.2f}' for x in m['box'])} Å")
            except Exception as e:
                messagebox.showerror("Matrix file", str(e))

    def _matrix(self):
        """Parsed matrix file (cached), or None when carving from the generated lattice."""
        path = self.matrix_path.get().strip()
        if self.matrix_mode.get() != "file" or not path:
            return None
        key = (path, os.path.getmtime(path))
        if getattr(self, "_mat_key", None) != key:
            self._mat = core.read_lammps_data(path)
            self._mat_key = key
        return self._mat

    def _browse_outdir(self):
        path = filedialog.askdirectory(title="Output folder", initialdir=self.outdir.get())
        if path:
            self.outdir.set(path)

    # ───────────────────────── right panel ─────────────────────────
    def _right(self, p):
        sf = ttk.LabelFrame(p, text="Model summary (live)", padding=8)
        sf.pack(fill="x", padx=(6, 0))
        self.est = {}
        for r, (k, lab) in enumerate((("pot", "Potential"), ("alloy", "Alloy"),
                                      ("lat", "Lattice"), ("box", "Box"),
                                      ("beads", "Beads"), ("md", "CG MD"))):
            ttk.Label(sf, text=lab, style="Key.TLabel").grid(row=r, column=0, sticky="nw",
                                                             padx=(2, 10), pady=1)
            self.est[k] = wrap_label(sf, "—")
            self.est[k].grid(row=r, column=1, sticky="we", pady=1)
        sf.columnconfigure(1, weight=1)

        bf = ttk.Frame(p)
        bf.pack(fill="x", padx=(6, 0), pady=(8, 2))
        self.b_gen = ttk.Button(bf, text="▶  Generate CG model", style="Accent.TButton",
                                command=self._generate)
        self.b_gen.pack(side="left")
        self.b_run = ttk.Button(bf, text="⚙  Run simulation", command=self._run_sim,
                                state="disabled")
        self.b_run.pack(side="left", padx=6)
        self.b_stop = ttk.Button(bf, text="■  Stop", command=self._stop, state="disabled")
        self.b_stop.pack(side="left")
        bf2 = ttk.Frame(p)
        bf2.pack(fill="x", padx=(6, 0), pady=2)
        ttk.Button(bf2, text="Open output folder", command=self._open_outdir).pack(side="left")
        ttk.Button(bf2, text="Plot stress–strain", command=self._plot_ss).pack(side="left",
                                                                                padx=6)
        ttk.Button(bf2, text="Copy command line", command=self._copy_cli).pack(side="left")
        self.progress = ttk.Progressbar(p, mode="determinate", value=0)
        self.progress.pack(fill="x", padx=(6, 0), pady=(6, 2))
        lf = ttk.LabelFrame(p, text="Log", padding=4)
        lf.pack(fill="both", expand=True, padx=(6, 0), pady=(2, 0))
        self.logw = ScrolledText(lf, wrap="word", height=20, font="TkFixedFont", state="disabled")
        self.logw.pack(fill="both", expand=True)
        self.logw.tag_configure("err", foreground="#b3261e")
        self.logw.tag_configure("ok", foreground="#1b7a2f")
        self.logw.tag_configure("warn", foreground="#9a6700")
        ttk.Button(lf, text="Clear log", command=self._clear_log).pack(anchor="e", pady=(4, 0))

    # ───────────────────────── live estimates ─────────────────────────
    def _refresh(self):
        if self._est_job:
            self.root.after_cancel(self._est_job)
        self._est_job = self.root.after(250, self._estimates)

    def _f(self, var, typ=float):
        try:
            return typ(var.get())
        except (ValueError, tk.TclError):
            return None

    def _estimates(self):
        self._est_job = None
        info, kind = self.info, self.lattice.get()
        n, a0 = self._f(self.n, int), self._f(self.a0)
        T, dt = self._f(self.temp), self._f(self.dt)
        self.est["pot"].config(text=(f"{core.PAIR_STYLES[info.pair]}, cutoff {info.cutoff:.2f} Å"
                                     + (f" → {n*info.cutoff:.2f} Å (CG)" if n else ""))
                               if info else "not attached (step ①)")
        try:
            els, comp = self._composition()
            s, label, _ = core.entropy_class(comp)
            self.est["alloy"].config(text=f"{' '.join(f'{e} {100*c:.4g}' for e, c in zip(els, comp))}"
                                          f" at.% — {label}")
        except ValueError:
            els = None
            self.est["alloy"].config(text="select elements (step ②)")
        if n and n >= 1:
            self.n_info.config(text=f"1 bead = {n**3:,} atoms   •   bead mass = {n**3} × atomic mass"
                                    f"   •   T_CG = {n**3:,} × T")
        if T and dt and n:
            dtc = self._f(self.dt_cg) or n * dt
            self.est["md"].config(text=f"T_CG = {n**3*T:,.0f} K  (physical {T:g} K), "
                                       f"dt_CG = {dtc:g} ps"
                                       + ("" if abs(dtc - n*dt) < 1e-12 else "  (override)"))
        if self.matrix_mode.get() == "file":
            try:
                mat = self._matrix()
            except Exception as e:
                mat, err = None, str(e)
            else:
                err = "choose the data file (step ③)"
            if mat is None or not n:
                for k in ("lat", "beads"):
                    self.est[k].config(text="—")
                self.est["box"].config(text=f"matrix: {err}")
                return
            pk, _ = core.tpms_periods(mat["box"], self._f(self.periods, int) or 1)
            self.est["lat"].config(text=f"carved from {mat['name']} ({len(mat['pos']):,} atoms"
                                        + (", species re-assigned at random" if self.matrix_random.get()
                                           else f", {mat['ntypes']} types")
                                        + (f", scaled ×{n}" if n > 1 else "")
                                        + f"); TPMS cells {pk[0]}×{pk[1]}×{pk[2]}")
            self.est["box"].config(text=" × ".join(f"{x*n/10:.1f}" for x in mat["box"]) + " nm"
                                   + (f"  (exact atomistic equivalent: the file's own "
                                      f"{' × '.join(f'{x/10:.1f}' for x in mat['box'])} nm)"
                                      if n > 1 else ""))
            try:
                g = self._geometry(a0)
                txt = (f"{len(g.pos):,} beads ≈ {len(g.pos)*n**3:,} atoms  •  fill "
                       f"{100*g.fill:.1f} %")
                self.geo_info.config(text=f"{self.surface.get()} {g.phase} carved from "
                                          f"{mat['name']}: level t = {g.level:.4g}, {txt}")
            except Exception as e:
                txt = f"({e})"
            self.est["beads"].config(text=txt)
            return
        cells = self._cells(n, a0)
        if not (n and cells and a0 and n >= 1 and cells >= 1 and a0 > 0):
            for k in ("lat", "box", "beads"):
                self.est[k].config(text="—")
            return
        a_cg = n * a0
        L = cells * a_cg
        sites = core.ATOMS_PER_CELL[kind] * cells**3
        self.est["lat"].config(text=f"{kind.upper()}, a0 = {a0:.4f} Å"
                                    f"{' (to be relaxed)' if self.auto_a0.get() else ''} → "
                                    f"a_CG = {a_cg:.3f} Å")
        self.est["box"].config(text=f"{L:.1f} Å = {L/10:.1f} nm  (exact atomistic equivalent: "
                                    f"{L/n/10:.2f} nm cell)")
        txt = f"{sites:,} lattice sites"
        if sites <= LIVE_SITE_LIMIT:
            try:
                g = self._geometry(a0)
                txt = (f"{len(g.pos):,} beads ≈ {len(g.pos)*n**3:,} atoms  •  fill "
                       f"{100*g.fill:.1f} %  •  surface beads {100*g.surface_fraction:.0f} %")
                self.geo_info.config(text=f"{self.surface.get()} {g.phase}: level t = {g.level:.4g}, "
                                          f"{txt}" + ("  — walls are very thin; increase cells "
                                                      "or reduce n/periods"
                                                      if g.surface_fraction > 0.6 else ""))
            except Exception as e:
                txt += f"  ({e})"
        else:
            dens = self._f(self.density)
            if self.thick_mode.get() == "density" and dens:
                txt = f"≈ {int(sites*dens/100):,} beads ≈ {int(sites*dens/100)*n**3:,} atoms"
            txt += "  (press Preview for exact numbers)"
        self.est["beads"].config(text=txt)

    def _cells(self, n=None, a0=None):
        """CG cells per side, from the cell count or from the box edge in nm."""
        if self.box_mode.get() == "nm":
            nm = self._f(self.box_nm)
            if not (nm and n and a0 and nm > 0 and a0 > 0):
                return None
            return max(1, int(np.ceil(nm * 10 / (n * a0) - 1e-9)))
        return self._f(self.cells, int)

    def _geometry(self, a0=None):
        a0 = a0 or self._f(self.a0)
        n, periods = int(self.n.get()), int(self.periods.get())
        dens, t = None, 0.0
        if self.thick_mode.get() == "density":
            dens = float(self.density.get()) / 100.0
        else:
            t = float(self.t.get())
        mat = self._matrix()
        if mat is not None:
            keep, phase, level, pk = core.carve_matrix(mat, self.surface.get(), periods,
                                                       self.phase.get(), t, dens)
            g = core.Geometry(mat["pos"][keep] * n, keep, float(mat["box"].max() * n), phase,
                              level, float(keep.mean()), None, 0, len(keep))
            if self.matrix_random.get():
                try:
                    _, comp = self._composition()
                except ValueError:
                    comp = [1.0]
                g.types = core.assign_species(len(keep), comp, np.random.default_rng(0))[keep]
            else:
                g.types = mat["types"][keep]
            g.periods = pk
            return g
        cells = self._cells(n, a0)
        if not cells:
            raise ValueError("set the box size")
        return core.build_geometry(self.lattice.get(), n * a0, cells, self.surface.get(),
                                   periods, self.phase.get(), t, dens,
                                   int(self.min_coord.get() or 0))

    # ───────────────────────── preview / plots ─────────────────────────
    def _preview(self):
        try:
            a0 = self._f(self.a0)
            if not a0:
                raise ValueError("set a0 (step ②)")
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            g = self._geometry(a0)
        except Exception as e:
            messagebox.showerror("Preview", str(e))
            return
        finally:
            self.root.config(cursor="")
        n = int(self.n.get())
        where = ("carved from " + self._matrix()["name"] if hasattr(g, "types") else
                 f"on a CG-{self.lattice.get().upper()} lattice")
        msg = (f"{self.surface.get()} {g.phase} {where}: "
               f"{len(g.pos):,} beads (≈ {len(g.pos)*n**3:,} atoms), fill {100*g.fill:.1f} %, "
               + (f"surface beads {100*g.surface_fraction:.0f} %, "
                  if g.surface_fraction is not None else "")
               + f"level t = {g.level:.4g}"
               + (f", {g.pruned} dangling beads removed" if g.pruned else ""))
        self.geo_info.config(text=msg)
        self._log("Preview: " + msg)
        if not HAVE_MPL or len(g.pos) == 0:
            return
        try:
            els, comp = self._composition()
        except ValueError:
            els, comp = ["bead"], [1.0]
        types = getattr(g, "types", None)
        if types is None:
            types = core.assign_species(len(g.pos), comp, np.random.default_rng(0))
        elif self.matrix_types.get().split():
            els = self.matrix_types.get().split()
        win = tk.Toplevel(self.root)
        win.title(f"Geometry preview — {self.surface.get()} {self.lattice.get().upper()}")
        fig = Figure(figsize=(11, 5.4), dpi=100)
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        idx = np.random.default_rng(1).permutation(len(g.pos))[:15000]
        cmap = matplotlib.colormaps["tab10"]
        P, Tt = g.pos[idx], types[idx]
        for k, el in enumerate(els):
            m = Tt == k + 1
            ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=3, color=cmap(k % 10), label=el,
                       depthshade=True)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        ax.set_title(f"{len(g.pos):,} beads" + (" (15,000 shown)" if len(g.pos) > 15000 else ""))
        ax.legend(loc="upper left", fontsize=8, markerscale=3)
        ax2 = fig.add_subplot(1, 2, 2)
        a_cg = n * self._f(self.a0)
        zmid = g.L / 2
        m = np.abs(g.pos[:, 2] - zmid) < a_cg / 2
        for k, el in enumerate(els):
            mm = m & (types == k + 1)
            ax2.scatter(g.pos[mm, 0], g.pos[mm, 1], s=6, color=cmap(k % 10), label=el)
        ax2.set_xlim(0, g.L)
        ax2.set_ylim(0, g.L)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x (Å)")
        ax2.set_ylabel("y (Å)")
        ax2.set_title(f"mid-plane slice (z ≈ {zmid:.0f} Å, one CG cell thick)")
        fig.suptitle(f"{self.surface.get()} {g.phase} on CG-{self.lattice.get().upper()}  •  "
                     f"fill {100*g.fill:.1f} %  •  box {g.L/10:.1f} nm", fontsize=11)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        NavigationToolbar2Tk(canvas, win).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_ss(self):
        path = None
        if self.summary and self.summary.get("stress_strain"):
            path = os.path.join(self.summary["outdir"], self.summary["stress_strain"])
        if not path or not os.path.isfile(path):
            path = filedialog.askopenfilename(
                title="Stress-strain file (*_stress_strain.txt)",
                initialdir=self.outdir.get(), filetypes=[("stress-strain", "*_stress_strain.txt"),
                                                         ("All files", "*")])
            if not path:
                return
        try:
            d = np.loadtxt(path, comments="#", ndmin=2)
        except Exception as e:
            messagebox.showerror("Stress-strain", f"cannot read {path}: {e}")
            return
        if not HAVE_MPL:
            messagebox.showinfo("Stress-strain", f"{len(d)} points; max stress "
                                                 f"{d[:, 2].max():.3f} GPa (matplotlib needed "
                                                 f"for the plot)")
            return
        win = tk.Toplevel(self.root)
        win.title(f"Stress–strain — {os.path.basename(path)}")
        fig = Figure(figsize=(7, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(100 * d[:, 1], d[:, 2], lw=1.6, color="#1f5fa6")
        ax.set_xlabel("engineering strain (%)")
        ax.set_ylabel("stress (GPa, compression positive)")
        ax.grid(alpha=0.3)
        ax.set_title(os.path.basename(path).replace("_stress_strain.txt", ""))
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        NavigationToolbar2Tk(canvas, win).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ───────────────────────── configuration ─────────────────────────
    def _config(self):
        if not self.info:
            raise ValueError("Step ①: attach an interatomic potential first.")
        els, comp = self._composition()

        def num(var, name, typ=float, lo=None):
            try:
                v = typ(var.get())
            except ValueError:
                raise ValueError(f"{name}: '{var.get()}' is not a valid number")
            if lo is not None and v < lo:
                raise ValueError(f"{name} must be ≥ {lo}")
            return v

        def opt(var, name):
            return num(var, name, float, 1e-9) if var.get().strip() else None

        pair = self.pair.get()
        matrix, m_els = None, None
        if self.matrix_mode.get() == "file":
            matrix = self.matrix_path.get().strip()
            if not matrix or not os.path.isfile(matrix):
                raise ValueError("Step ③: choose the LAMMPS data file to carve from")
        if matrix and self.matrix_random.get():
            pass                                   # species from step ②, file types ignored
        elif matrix:
            m_els = self.matrix_types.get().split() or list(els)
            if sorted(m_els) != sorted(els):
                raise ValueError(f"Step ③: the atom types of the file ({' '.join(m_els)}) must "
                                 f"be exactly the elements ticked in step ② ({' '.join(els)})")
            ntypes = self._matrix()["ntypes"]
            if ntypes != len(m_els):
                raise ValueError(f"Step ③: the data file has {ntypes} atom types but "
                                 f"{len(m_els)} elements are listed")
        dens = num(self.density, "target density", float, 0) / 100 \
            if self.thick_mode.get() == "density" else None
        t = 0.0 if dens else num(self.t, "level-set value t")
        lib_order = self.lib_order.get().split() or None
        cfg = core.Config(
            pair=pair,
            potential=self.pot_path.get().strip() if pair != "meam" else "",
            library=self.lib_path.get().strip() if pair == "meam" else "",
            params=self.par_path.get().strip() if pair == "meam" else "",
            lib_elements=lib_order if pair == "meam" else None,
            elements=m_els or els, composition=None if m_els else comp,
            matrix=matrix, matrix_species="random" if matrix and self.matrix_random.get() else "file",
            lattice=self.lattice.get(),
            n=num(self.n, "CG level n", int, 1),
            a0=num(self.a0, "a0", float, 0.5),
            auto_a0=self.auto_a0.get(), verify=self.verify.get(),
            check_pure=self.check_pure.get(), compare_phases=self.compare.get(),
            surface=self.surface.get(),
            cells=num(self.cells, "cells", int, 1) if self.box_mode.get() == "cells" else 20,
            box_nm=num(self.box_nm, "box edge", float, 0.1) if self.box_mode.get() == "nm" else None,
            dt_cg=opt(self.dt_cg, "CG timestep"), tdamp=opt(self.tdamp, "thermostat damping"),
            pdamp=opt(self.pdamp, "barostat damping"),
            periods=num(self.periods, "periods", int, 1), phase=self.phase.get(), t=t,
            density=dens, min_coord=num(self.min_coord, "neighbour threshold", int, 0),
            temp=num(self.temp, "temperature", float, 0), dt_aa=num(self.dt, "timestep", float, 0),
            equil=num(self.equil, "equilibration steps", int, 0), crush=self.crush.get(),
            crush_strain=num(self.crush_strain, "final strain", float, 0),
            strain_rate=num(self.rate, "strain rate", float, 0), crush_axis=self.crush_axis.get(),
            seed=num(self.seed, "seed", int, 0), lmp=self.lmp.get().strip() or "lmp",
            outdir=self.outdir.get().strip() or "cg_hea_output")
        cfg.validate()
        return cfg

    def _apply_config(self, cfg):
        self.pair.set(cfg.pair)
        self.pot_path.set(cfg.potential or "")
        self.lib_path.set(cfg.library or "")
        self.par_path.set(cfg.params or "")
        self.lib_order.set(" ".join(cfg.lib_elements or []))
        self._show_pair_frame()
        self._load_potential()
        if self.info:
            self._rebuild_elements(keep={e: 100 * c for e, c in zip(cfg.elements,
                                                                    cfg.fractions())})
        self.lattice.set(cfg.lattice)
        if cfg.a0:
            self.a0_manual = True
            self.a0.set(f"{cfg.a0:.6g}")
        else:
            self._reset_a0()
        for var, val in ((self.auto_a0, cfg.auto_a0), (self.verify, cfg.verify),
                         (self.check_pure, cfg.check_pure), (self.compare, cfg.compare_phases),
                         (self.crush, cfg.crush)):
            var.set(bool(val))
        for var, val in ((self.n, cfg.n), (self.surface, cfg.surface), (self.cells, cfg.cells),
                         (self.periods, cfg.periods), (self.min_coord, cfg.min_coord),
                         (self.temp, cfg.temp), (self.dt, cfg.dt_aa), (self.equil, cfg.equil),
                         (self.crush_axis, cfg.crush_axis), (self.crush_strain, cfg.crush_strain),
                         (self.rate, cfg.strain_rate), (self.seed, cfg.seed), (self.lmp, cfg.lmp),
                         (self.outdir, cfg.outdir)):
            var.set(str(val))
        self.matrix_mode.set("file" if cfg.matrix else "lattice")
        self.matrix_path.set(cfg.matrix or "")
        self.matrix_types.set(" ".join(cfg.elements) if cfg.matrix else "")
        self.matrix_random.set(cfg.matrix_species == "random")
        self.box_mode.set("nm" if cfg.box_nm else "cells")
        if cfg.box_nm:
            self.box_nm.set(f"{cfg.box_nm:g}")
        for var, val in ((self.dt_cg, cfg.dt_cg), (self.tdamp, cfg.tdamp), (self.pdamp, cfg.pdamp)):
            var.set(f"{val:g}" if val else "")
        phase, t = core.resolve_phase(cfg.phase, cfg.t)
        self.phase.set(phase)
        if cfg.density:
            self.thick_mode.set("density")
            self.density.set(f"{100*cfg.density:.6g}")
        else:
            self.thick_mode.set("level")
            self.t.set(str(t))
        self._crush_state()
        self._comp_changed()

    def _save_settings(self):
        try:
            cfg = self._config()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        path = filedialog.asksaveasfilename(title="Save settings", defaultextension=".json",
                                            filetypes=[("JSON", "*.json")],
                                            initialfile="cg_hea_settings.json")
        if path:
            cfg.to_json(path)
            self._log(f"Settings saved to {path}")

    def _load_settings(self):
        path = filedialog.askopenfilename(title="Load settings",
                                          filetypes=[("JSON", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            self._apply_config(core.Config.from_json(path))
        except Exception as e:
            messagebox.showerror("Settings", f"cannot load {path}: {e}")
            return
        self._log(f"Settings loaded from {path}")

    def _copy_cli(self):
        try:
            cmd = self._config().to_cli()
        except ValueError as e:
            messagebox.showerror("Command line", str(e))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self._log("Equivalent command line (copied to the clipboard):\n" + cmd)

    # ───────────────────────── running ─────────────────────────
    def _lammps_ok(self, lmp):
        first = core.split_command(lmp)[0] if lmp.strip() else ""
        return bool(first) and (os.path.isfile(first) or bool(shutil.which(first)))

    def _generate(self):
        if self.busy:
            return
        try:
            cfg = self._config()
        except ValueError as e:
            messagebox.showerror("Please check the input", str(e))
            return
        needs = cfg.auto_a0 or cfg.verify or cfg.check_pure or cfg.compare_phases
        if needs and not self._lammps_ok(cfg.lmp):
            if not messagebox.askyesno(
                    "LAMMPS not found",
                    f"'{cfg.lmp}' was not found, so a0 relaxation and the checks cannot run.\n\n"
                    f"Generate the CG model without them (using a0 = {cfg.a0:.4f} Å)?"):
                return
            cfg.auto_a0 = cfg.verify = cfg.check_pure = cfg.compare_phases = False
        if os.path.isdir(cfg.outdir) and any(f.endswith("_run.in") for f in os.listdir(cfg.outdir)):
            self._log(f"Note: {cfg.outdir} already holds a model; files with the same name are "
                      f"overwritten.")
        self.summary = None
        self._start("Generating the CG model…")
        self._log("\n" + "─" * 70 + f"\nGenerating: {cfg.lattice.upper()} "
                  f"{''.join(cfg.elements)}, n = {cfg.n}, {cfg.surface}")
        self.session = core.Session(log=lambda m: self.q.put(("log", m)))
        threading.Thread(target=self._worker, args=(cfg,), daemon=True).start()

    def _worker(self, cfg):
        try:
            res = core.run_pipeline(cfg, self.session)
            self.q.put(("done", res))
        except core.Cancelled:
            self.q.put(("stopped", None))
        except (core.PotentialError, core.LammpsError, core.VerificationError, ValueError) as e:
            self.q.put(("error", str(e)))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def _run_sim(self):
        if self.busy or not self.summary:
            return
        s = self.summary
        lmp = self.lmp.get().strip() or "lmp"
        if not self._lammps_ok(lmp):
            messagebox.showerror("LAMMPS", f"'{lmp}' was not found (step ④).")
            return
        cmd = (core.split_command(self.mpi.get()) if self.mpi.get().strip() else []) + \
            core.split_command(lmp) + ["-in", s["run_input"], "-log", "log.lammps"]
        steps = int(self.equil.get() or 0)
        if not messagebox.askyesno(
                "Run simulation",
                f"Run in {s['outdir']}:\n\n{' '.join(cmd)}\n\n{s['n_beads']:,} beads, "
                f"{steps:,} equilibration steps{' + compression' if self.crush.get() else ''}. "
                f"Large models can take hours; you can stop at any time.\n\nStart now?"):
            return
        self._start("Running LAMMPS…")
        self._log("\n" + "─" * 70 + "\nRunning: " + " ".join(cmd))
        self.session = core.Session(log=lambda m: self.q.put(("log", m)))
        threading.Thread(target=self._sim_worker, args=(cmd, s["outdir"]), daemon=True).start()

    def _sim_worker(self, cmd, cwd):
        try:
            rc = self.session.stream_lammps(cmd, cwd, lambda l: self.q.put(("log", l)))
            self.q.put(("simdone", rc))
        except core.Cancelled:
            self.q.put(("stopped", None))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _start(self, msg):
        self.busy = True
        self.b_gen.configure(state="disabled")
        self.b_run.configure(state="disabled")
        self.b_stop.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._status(msg)

    def _finish(self, msg):
        self.busy = False
        self.b_gen.configure(state="normal")
        self.b_run.configure(state="normal" if self.summary else "disabled")
        self.b_stop.configure(state="disabled")
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._status(msg)

    def _stop(self):
        if self.session:
            self.session.cancel()
            self._log("Stopping…", "warn")

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    tag = ("err" if "ERROR" in val or "FAIL" in val else
                           "warn" if ("WARNING" in val or "**" in val or "CHECK" in val) else
                           "ok" if "PASS" in val else None)
                    self._log(val, tag)
                elif kind == "done":
                    self.summary = val
                    if val.get("a0_relaxed"):
                        self.relaxed = (self._alloy_key(), val["a0_relaxed"])
                        self.a0_manual = False
                        self._update_a0()
                    self._log(f"✓ CG model written to {val['outdir']}", "ok")
                    self._log(f"  data: {val['data_file']}   input: {val['run_input']}\n"
                              f"  run:  cd \"{val['outdir']}\" && lmp -in {val['run_input']}", "ok")
                    v = val.get("verify")
                    if v and not v["passed"]:
                        self._log("  ** the mapping check did not pass — see above **", "err")
                    self._finish("Done. Press 'Run simulation' to start LAMMPS, or open the "
                                 "output folder.")
                elif kind == "simdone":
                    self._log(f"LAMMPS finished (exit code {val}).", "ok" if val == 0 else "err")
                    self._finish("Simulation finished." if val == 0 else
                                 "LAMMPS stopped with an error — see the log.")
                elif kind == "stopped":
                    self._log("Stopped by user.", "warn")
                    self._finish("Stopped.")
                elif kind == "error":
                    self._log("ERROR: " + val, "err")
                    self._finish("Failed — see the log.")
                    messagebox.showerror("Error", val[-1500:])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ───────────────────────── misc ─────────────────────────
    def _log(self, msg, tag=None):
        w = self.logw
        w.configure(state="normal")
        w.insert("end", msg + "\n", (tag,) if tag else ())
        lines = int(w.index("end-1c").split(".")[0])
        if lines > 6000:
            w.delete("1.0", f"{lines - 5000}.0")
        w.see("end")
        w.configure(state="disabled")

    def _clear_log(self):
        self.logw.configure(state="normal")
        self.logw.delete("1.0", "end")
        self.logw.configure(state="disabled")

    def _status(self, msg):
        self.statusbar.config(text=msg)

    def _open_outdir(self):
        d = (self.summary or {}).get("outdir") or self.outdir.get()
        if not os.path.isdir(d):
            messagebox.showinfo("Output folder", f"{d} does not exist yet — generate a model first.")
            return
        try:
            if platform.system() == "Windows":
                os.startfile(d)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            messagebox.showinfo("Output folder", f"{d}\n({e})")

    def _on_wheel(self, e):
        w = self.root.winfo_containing(e.x_root, e.y_root)
        if w is None or isinstance(w, (tk.Text, ttk.Treeview, tk.Listbox, ttk.Combobox,
                                       ttk.Spinbox)):
            return
        up = getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0
        while w is not None:
            if getattr(w, "_scrollable", False):
                top, bottom = w.yview()
                if top > 0 or bottom < 1:
                    w.yview_scroll(-1 if up else 1, "units")
                    return
            w = w.master

    def _quit(self):
        if self.busy and not messagebox.askyesno("Quit", "A job is running. Stop it and quit?"):
            return
        if self.session:
            self.session.cancel()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
