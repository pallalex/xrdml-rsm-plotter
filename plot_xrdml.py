import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import re

# ============================================================
# Universal XRDML RSM / 2Theta-Omega plotter
#
# Change only FILENAME.
#
# Works with:
# 1) RSM maps: many Omega rows, 2Theta scan in each row
# 2) 1D 2Theta-Omega scans
# 3) 2Theta as listPositions or start/end
#
# Qx sign:
# - asymmetric reflections: normal sign, QX_SIGN = +1
# - symmetric reflections: can be automatically mirrored to match X'Pert
# ============================================================

# -------------------------
# USER SETTINGS
# -------------------------
FILENAME = "V071826A_Si(111)_RSM_135.xrdml"

LOG_EPS = 1.0
CMAP = "jet"

# Auto mode:
# asymmetric RSM -> QX_SIGN = +1
# symmetric RSM -> QX_SIGN = -1, to look like X'Pert if needed
AUTO_QX_SIGN = True

# If AUTO_QX_SIGN = False, this value is used manually.
# For asymmetric usually use +1.
# For symmetric, if mirrored compared with X'Pert, use -1.
MANUAL_QX_SIGN = 1.0

# Symmetric reflection condition:
# if |omega_peak - theta_peak| is smaller than this value,
# script treats the scan as symmetric-like.
SYMMETRIC_LIMIT_DEG = 2.0

# Plot options
SHOW_CONTOURS = False
USE_XPERT_SCALE = False       # True -> Q * 10000
CENTER_QX_AT_PEAK = False   # True -> peak shifted to Qx = 0

# Figure size
FIGSIZE_RSM = (7, 5)
FIGSIZE_1D = (8, 4)

# -------------------------
# FILE PATH
# -------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

XRDML_FILE = os.path.join(SCRIPT_DIR, FILENAME)


# ============================================================
# FUNCTIONS
# ============================================================

def get_namespace(root):
    m = re.match(r"\{(.*)\}", root.tag)
    if m:
        return {"x": m.group(1)}
    return {}


def get_text(parent, path, ns):
    node = parent.find(path, ns)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def read_wavelength(root, ns):
    for tag in ["kAlpha1", "kAlphaAverage"]:
        txt = get_text(root, f".//x:usedWavelength/x:{tag}", ns)
        if txt is not None:
            return float(txt)

    raise RuntimeError("Cannot find wavelength in XRDML file.")


def read_intensity(scan, ns):
    for tag in ["intensities", "counts"]:
        txt = get_text(scan, f".//x:{tag}", ns)
        if txt is not None:
            arr = np.fromstring(txt, sep=" ")
            if arr.size > 0:
                return arr

    return None


def read_axis(scan, ns, axis_name, n_points):
    pos = scan.find(f".//x:positions[@axis='{axis_name}']", ns)

    if pos is None:
        return None

    # Case 1: full list of positions
    txt = get_text(pos, "x:listPositions", ns)
    if txt is not None:
        arr = np.fromstring(txt, sep=" ")

        if arr.size != n_points:
            raise RuntimeError(
                f"{axis_name}: listPositions has {arr.size} points, "
                f"but intensity has {n_points} points."
            )

        return arr

    # Case 2: one fixed position
    txt = get_text(pos, "x:commonPosition", ns)
    if txt is not None:
        return np.full(n_points, float(txt))

    # Case 3: start/end scan
    start_txt = get_text(pos, "x:startPosition", ns)
    end_txt = get_text(pos, "x:endPosition", ns)

    if start_txt is not None and end_txt is not None:
        start = float(start_txt)
        end = float(end_txt)
        return np.linspace(start, end, n_points)

    return None


def load_xrdml(filename):
    root = ET.parse(filename).getroot()
    ns = get_namespace(root)

    wl = read_wavelength(root, ns)

    scans = root.findall(".//x:xrdMeasurement/x:scan", ns)
    if not scans:
        scans = root.findall(".//x:scan", ns)

    if not scans:
        raise RuntimeError("No scans found in XRDML file.")

    tt_rows = []
    om_rows = []
    int_rows = []

    scan_axes = []

    phi_values = []
    psi_values = []

    for scan in scans:
        I = read_intensity(scan, ns)

        if I is None or I.size == 0:
            continue

        TT = read_axis(scan, ns, "2Theta", I.size)
        OM = read_axis(scan, ns, "Omega", I.size)

        # Some files may use Theta instead of Omega
        if OM is None:
            OM = read_axis(scan, ns, "Theta", I.size)

        if TT is None or OM is None:
            continue

        tt_rows.append(TT)
        om_rows.append(OM)
        int_rows.append(I)

        scan_axes.append(scan.attrib.get("scanAxis", "unknown"))

        PHI = read_axis(scan, ns, "Phi", I.size)
        PSI = read_axis(scan, ns, "Psi", I.size)

        if PHI is not None:
            phi_values.append(np.mean(PHI))

        if PSI is not None:
            psi_values.append(np.mean(PSI))

    if len(int_rows) == 0:
        raise RuntimeError("No usable Omega / 2Theta / intensity data found.")

    lengths = [row.size for row in int_rows]

    if len(set(lengths)) != 1:
        raise RuntimeError(f"Different row lengths found: {sorted(set(lengths))}")

    TT = np.vstack(tt_rows)
    OM = np.vstack(om_rows)
    INT = np.vstack(int_rows)

    info = {
        "filename": os.path.basename(filename),
        "wavelength": wl,
        "scan_axes": sorted(set(scan_axes)),
        "phi": np.unique(np.round(phi_values, 6)),
        "psi": np.unique(np.round(psi_values, 6)),
    }

    return TT, OM, INT, info


def calculate_q(TT_deg, OM_deg, wavelength, qx_sign):
    k = 2.0 * np.pi / wavelength

    # Coplanar geometry:
    # alpha = incident angle relative to sample surface
    # beta  = exit angle relative to sample surface
    alpha = np.deg2rad(OM_deg)
    beta = np.deg2rad(TT_deg - OM_deg)

    Qx = qx_sign * k * (np.cos(beta) - np.cos(alpha))
    Qz =           k * (np.sin(beta) + np.sin(alpha))

    return Qx, Qz


def detect_peak_and_symmetry(TT, OM, INT):
    idx_max = np.unravel_index(np.argmax(INT), INT.shape)

    omega_peak = OM[idx_max]
    twotheta_peak = TT[idx_max]
    theta_peak = twotheta_peak / 2.0

    delta = omega_peak - theta_peak

    is_symmetric = abs(delta) < SYMMETRIC_LIMIT_DEG

    return idx_max, omega_peak, twotheta_peak, theta_peak, delta, is_symmetric


# ============================================================
# MAIN
# ============================================================

TT, OM, INT, info = load_xrdml(XRDML_FILE)

idx_max, omega_peak, twotheta_peak, theta_peak, delta_peak, is_symmetric = detect_peak_and_symmetry(TT, OM, INT)

if AUTO_QX_SIGN:
    if is_symmetric:
        QX_SIGN = -1.0
    else:
        QX_SIGN = 1.0
else:
    QX_SIGN = MANUAL_QX_SIGN

Qx, Qz = calculate_q(TT, OM, info["wavelength"], QX_SIGN)

Z = np.log10(INT + LOG_EPS)

qx_peak = Qx[idx_max]
qz_peak = Qz[idx_max]

print("Loaded:", info["filename"])
print("  wavelength:", info["wavelength"], "Angstrom")
print("  scan axes:", info["scan_axes"])
print("  data shape:", INT.shape)
print("  2theta range:", np.min(TT), "to", np.max(TT), "deg")
print("  omega range:", np.min(OM), "to", np.max(OM), "deg")
print("  phi:", info["phi"])
print("  psi:", info["psi"])

print("\nStrongest peak:")
print("  omega:", omega_peak, "deg")
print("  2theta:", twotheta_peak, "deg")
print("  theta:", theta_peak, "deg")
print("  omega - theta:", delta_peak, "deg")
print("  Qx:", qx_peak, "1/A")
print("  Qz:", qz_peak, "1/A")
print("  intensity:", INT[idx_max])

print("\nScan type:")
if is_symmetric:
    print("  symmetric-like reflection")
else:
    print("  asymmetric reflection")

print("  QX_SIGN used:", QX_SIGN)


# ============================================================
# PLOT
# ============================================================

# Case 1: 1D scan
if INT.shape[0] == 1:
    plt.figure(figsize=FIGSIZE_1D)

    plt.plot(TT[0], Z[0], linewidth=1.0)

    plt.xlabel(r"$2\theta$ (deg)")
    plt.ylabel(r"$\log_{10}(I + 1)$")
    plt.title(info["filename"])

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Case 2: 2D RSM map
else:
    if CENTER_QX_AT_PEAK:
        X = Qx - qx_peak
        xlabel = r"$\Delta Q_x$ ($\AA^{-1}$)"
    else:
        X = Qx
        xlabel = r"$Q_x$ ($\AA^{-1}$)"

    Y = Qz
    ylabel = r"$Q_z$ ($\AA^{-1}$)"

    if USE_XPERT_SCALE:
        X = X * 10000.0
        Y = Y * 10000.0

        if CENTER_QX_AT_PEAK:
            xlabel = r"$\Delta Q_x \cdot 10000$"
        else:
            xlabel = r"$Q_x \cdot 10000$"

        ylabel = r"$Q_z \cdot 10000$"

    plt.figure(figsize=FIGSIZE_RSM)

    mesh = plt.pcolormesh(
        X,
        Y,
        Z,
        shading="auto",
        cmap=CMAP
    )

    if SHOW_CONTOURS:
        z_min = np.percentile(Z, 65)
        z_max = np.max(Z)

        if z_max > z_min:
            levels = np.linspace(z_min, z_max, 12)
            plt.contour(
                X,
                Y,
                Z,
                levels=levels,
                colors="black",
                linewidths=0.5
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(mesh, label=r"$\log_{10}(I + 1)$")

    plt.tight_layout()
    plt.show()