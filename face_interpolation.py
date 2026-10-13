"""
Lab 7 - Numerical Interpolation (Advanced Applications)
Task: Reconstruct a symmetric face outline from half-outline data.

Methods implemented FROM SCRATCH (no library interpolation used for results):
    1. Lagrange interpolating polynomial (degree 5)
    2. Newton divided-difference polynomial (degree 5)  -> must match Lagrange
    3. Natural cubic spline (piecewise cubic, C2 continuous)
SciPy's CubicSpline is used ONLY as an independent check of our spline.

Run:  python face_interpolation.py
Outputs: figures/*.png  and  results.txt
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. DATA  (y = position along face height, x = distance from midline)
# ----------------------------------------------------------------------
y_data = np.array([0, 2, 4, 6, 8, 10], dtype=float)
x_data = np.array([0.0, 1.1, 1.8, 2.4, 2.0, 1.3])
n = len(y_data)


# ----------------------------------------------------------------------
# 2. LAGRANGE INTERPOLATION
#    P(y) = sum_i  x_i * L_i(y),   L_i(y) = prod_{j!=i} (y - y_j)/(y_i - y_j)
# ----------------------------------------------------------------------
def lagrange(y_nodes, x_nodes, y_eval):
    y_eval = np.atleast_1d(np.asarray(y_eval, dtype=float))
    result = np.zeros_like(y_eval)
    m = len(y_nodes)
    for i in range(m):
        L = np.ones_like(y_eval)
        for j in range(m):
            if j != i:
                L *= (y_eval - y_nodes[j]) / (y_nodes[i] - y_nodes[j])
        result += x_nodes[i] * L
    return result


# ----------------------------------------------------------------------
# 3. NEWTON DIVIDED DIFFERENCES
#    P(y) = f[y0] + f[y0,y1](y-y0) + f[y0,y1,y2](y-y0)(y-y1) + ...
# ----------------------------------------------------------------------
def divided_difference_table(y_nodes, x_nodes):
    m = len(y_nodes)
    table = np.zeros((m, m))
    table[:, 0] = x_nodes
    for j in range(1, m):
        for i in range(m - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (
                y_nodes[i + j] - y_nodes[i]
            )
    return table


def newton(y_nodes, table, y_eval):
    y_eval = np.atleast_1d(np.asarray(y_eval, dtype=float))
    coeffs = table[0, :]
    result = np.full_like(y_eval, coeffs[-1])
    for k in range(len(coeffs) - 2, -1, -1):          # Horner-like nesting
        result = result * (y_eval - y_nodes[k]) + coeffs[k]
    return result


# ----------------------------------------------------------------------
# 4. NATURAL CUBIC SPLINE  (S''(y0) = S''(yn) = 0)
#    On [y_i, y_{i+1}] with h_i = y_{i+1}-y_i:
#    S_i(y) = a_i + b_i (y-y_i) + c_i (y-y_i)^2 + d_i (y-y_i)^3
#    Solve tridiagonal system for c_i.
# ----------------------------------------------------------------------
def natural_cubic_spline_coeffs(y_nodes, x_nodes):
    m = len(y_nodes)
    h = np.diff(y_nodes)
    A = np.zeros((m, m))
    rhs = np.zeros(m)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, m - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3.0 * ((x_nodes[i + 1] - x_nodes[i]) / h[i]
                        - (x_nodes[i] - x_nodes[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, rhs)
    a = x_nodes[:-1].copy()
    b = np.zeros(m - 1)
    d = np.zeros(m - 1)
    for i in range(m - 1):
        b[i] = (x_nodes[i + 1] - x_nodes[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    return a, b, c[:-1], d, A, rhs


def spline_eval(y_nodes, coeffs, y_eval):
    a, b, c, d = coeffs
    y_eval = np.atleast_1d(np.asarray(y_eval, dtype=float))
    idx = np.clip(np.searchsorted(y_nodes, y_eval, side="right") - 1, 0, len(a) - 1)
    t = y_eval - y_nodes[idx]
    return a[idx] + b[idx] * t + c[idx] * t**2 + d[idx] * t**3


# ----------------------------------------------------------------------
# 5. BUILD THE INTERPOLANTS
# ----------------------------------------------------------------------
y_fine = np.linspace(0, 10, 501)

table = divided_difference_table(y_data, x_data)
x_lag = lagrange(y_data, x_data, y_fine)
x_new = newton(y_data, table, y_fine)

a, b, c, d, A_mat, rhs_vec = natural_cubic_spline_coeffs(y_data, x_data)
x_spl = spline_eval(y_data, (a, b, c, d), y_fine)

# Independent check of our spline with SciPy
from scipy.interpolate import CubicSpline
x_spl_scipy = CubicSpline(y_data, x_data, bc_type="natural")(y_fine)

# ----------------------------------------------------------------------
# 6. REPORT
# ----------------------------------------------------------------------
lines = []
P = lines.append
np.set_printoptions(precision=5, suppress=True, linewidth=120)

P("=" * 70)
P("LAB 7 - FACE OUTLINE RECONSTRUCTION BY INTERPOLATION")
P("=" * 70)
P("\nInput data (left half):")
P("   y : " + str(y_data))
P("   x : " + str(x_data))

P("\n--- Newton divided-difference table ---")
P(f"{'y':>5} {'f[.]':>9} {'f[.,.]':>9} {'f[3]':>9} {'f[4]':>9} {'f[5]':>9} {'f[6]':>9}")
for i in range(n):
    row = f"{y_data[i]:5.1f}"
    for j in range(n - i):
        row += f" {table[i, j]:9.5f}"
    P(row)
P("\nNewton coefficients (top row of table):")
P("   " + "  ".join(f"c{k}={v:.5f}" for k, v in enumerate(table[0])))

P("\n--- Natural cubic spline coefficients  S_i(y)=a+b t+c t^2+d t^3, t=y-y_i ---")
P(f"{'interval':>10} {'a':>10} {'b':>10} {'c':>10} {'d':>10}")
for i in range(n - 1):
    P(f"[{y_data[i]:.0f},{y_data[i+1]:.0f}]".rjust(10)
      + f" {a[i]:10.5f} {b[i]:10.5f} {c[i]:10.5f} {d[i]:10.5f}")

P("\n--- Interpolated values at intermediate y ---")
y_q = np.array([1, 3, 5, 7, 9], dtype=float)
xl, xn = lagrange(y_data, x_data, y_q), newton(y_data, table, y_q)
xs = spline_eval(y_data, (a, b, c, d), y_q)
P(f"{'y':>4} {'Lagrange':>10} {'Newton':>10} {'Spline':>10}")
for i in range(len(y_q)):
    P(f"{y_q[i]:4.1f} {xl[i]:10.5f} {xn[i]:10.5f} {xs[i]:10.5f}")

P("\n--- Verification ---")
P(f"max |Lagrange - Newton|          = {np.max(np.abs(x_lag - x_new)):.3e}")
P(f"max |our spline - SciPy spline|  = {np.max(np.abs(x_spl - x_spl_scipy)):.3e}")
P(f"Lagrange reproduces data?  {np.allclose(lagrange(y_data, x_data, y_data), x_data)}")
P(f"Spline   reproduces data?  {np.allclose(spline_eval(y_data, (a, b, c, d), y_data), x_data)}")
P(f"Lagrange min x on [0,10] = {x_lag.min():.4f}  (never negative: outline never crosses the midline)")
P(f"Spline   min x on [0,10] = {x_spl.min():.4f}")
P(f"Lagrange max x = {x_lag.max():.4f} at y = {y_fine[x_lag.argmax()]:.2f}")
P(f"Spline   max x = {x_spl.max():.4f} at y = {y_fine[x_spl.argmax()]:.2f}  (widest point of face)")

# Leave-one-out cross-validation (interior points only)
P("\n--- Leave-one-out cross-validation (interior nodes) ---")
P(f"{'left out y':>10} {'true x':>8} {'Lagrange err':>13} {'Spline err':>11} {'Linear err':>11}")
loo = {"lag": [], "spl": [], "lin": []}
for k in range(1, n - 1):
    mask = np.arange(n) != k
    yk, xk = y_data[mask], x_data[mask]
    e_lag = abs(lagrange(yk, xk, y_data[k])[0] - x_data[k])
    cf = natural_cubic_spline_coeffs(yk, xk)[:4]
    e_spl = abs(spline_eval(yk, cf, y_data[k])[0] - x_data[k])
    e_lin = abs(np.interp(y_data[k], yk, xk) - x_data[k])
    loo["lag"].append(e_lag); loo["spl"].append(e_spl); loo["lin"].append(e_lin)
    P(f"{y_data[k]:10.0f} {x_data[k]:8.2f} {e_lag:13.4f} {e_spl:11.4f} {e_lin:11.4f}")
P(f"{'RMS':>10} {'':>8} {np.sqrt(np.mean(np.square(loo['lag']))):13.4f} "
  f"{np.sqrt(np.mean(np.square(loo['spl']))):11.4f} {np.sqrt(np.mean(np.square(loo['lin']))):11.4f}")

report = "\n".join(lines)
print(report)
with open("results.txt", "w") as f:
    f.write(report + "\n")

# ----------------------------------------------------------------------
# 7. PLOTS
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
BLUE, RED, GREEN, GREY = "#1f5fa8", "#c0392b", "#2a8a4a", "#555555"

# Fig 1 - raw data
fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(x_data, y_data, "o--", color=GREY, ms=8, label="Given data")
for xi, yi in zip(x_data, y_data):
    ax.annotate(f"({xi}, {yi:.0f})", (xi, yi), textcoords="offset points",
                xytext=(8, 4), fontsize=9)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("x (distance from midline)"); ax.set_ylabel("y (height; 0 = chin)")
ax.set_title("Given left half-outline data"); ax.set_aspect("equal")
ax.legend(); fig.tight_layout(); fig.savefig("figures/fig1_data.png", dpi=170); plt.close()

# Fig 2 - Lagrange vs spline vs data (x vs y)
fig, ax = plt.subplots(figsize=(6, 6.5))
ax.plot(x_lag, y_fine, color=RED, lw=2, label="Lagrange (degree 5)")
ax.plot(x_spl, y_fine, color=BLUE, lw=2, ls="--", label="Natural cubic spline")
ax.plot(x_data, y_data, "ko", ms=8, label="Data nodes")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
ax.set_title("Interpolated left half-outline"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig2_left_half.png", dpi=170); plt.close()

# Fig 3 - difference between methods
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(y_fine, x_lag - x_spl, color=GREEN, lw=2)
ax.plot(y_data, np.zeros(n), "ko")
ax.set_xlabel("y"); ax.set_ylabel("Lagrange - Spline")
ax.set_title("Difference between the two interpolants (zero at nodes)")
fig.tight_layout(); fig.savefig("figures/fig3_difference.png", dpi=170); plt.close()

# Fig 4 - complete face (spline) with mirrored right half
def face(ax, xcurve, title, color):
    ax.plot(-xcurve, y_fine, color=color, lw=2.5, label="Left half (interpolated)")
    ax.plot(+xcurve, y_fine, color=color, lw=2.5, ls="-", alpha=0.55,
            label="Right half (mirror)")
    ax.plot(-x_data, y_data, "ko", ms=6, label="Given data")
    ax.plot(+x_data, y_data, "o", color="white", mec="k", ms=6, label="Mirrored data")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title); ax.legend(loc="lower center", fontsize=8)

fig, ax = plt.subplots(figsize=(6, 6.5))
face(ax, x_spl, "Complete face outline (cubic spline)", BLUE)
fig.tight_layout(); fig.savefig("figures/fig4_face_spline.png", dpi=170); plt.close()

# Fig 5 - complete face (Lagrange)
fig, ax = plt.subplots(figsize=(6, 6.5))
face(ax, x_lag, "Complete face outline (Lagrange)", RED)
fig.tight_layout(); fig.savefig("figures/fig5_face_lagrange.png", dpi=170); plt.close()

# Fig 6 - final face, filled, clean, with closed top
y_closed = np.concatenate([y_fine, y_fine[::-1]])
x_closed = np.concatenate([-x_spl, x_spl[::-1]])
fig, ax = plt.subplots(figsize=(6, 6.5))
ax.fill(x_closed, y_closed, color="#f2d3b3", alpha=0.8)
ax.plot(-x_spl, y_fine, color="#6b3e1b", lw=2.5)
ax.plot(x_spl, y_fine, color="#6b3e1b", lw=2.5)
ax.plot([-x_spl[-1], x_spl[-1]], [10, 10], color="#6b3e1b", lw=2.5, ls="--")
ax.plot(-x_data, y_data, "ko", ms=5); ax.plot(x_data, y_data, "ko", ms=5)
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Final reconstructed face outline\n(dashed top edge = closing segment, not data)")
fig.tight_layout(); fig.savefig("figures/fig6_face_final.png", dpi=170); plt.close()

# Fig 7 - comparison panel incl. linear interpolation
fig, axs = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
x_lin = np.interp(y_fine, y_data, x_data)
for ax, xc, t, col in zip(axs, [x_lin, x_lag, x_spl],
                          ["Linear", "Lagrange (deg 5)", "Cubic spline"],
                          [GREY, RED, BLUE]):
    ax.plot(-xc, y_fine, color=col, lw=2.2); ax.plot(xc, y_fine, color=col, lw=2.2)
    ax.plot(-x_data, y_data, "ko", ms=5); ax.plot(x_data, y_data, "ko", ms=5)
    ax.axvline(0, color="k", lw=0.8, ls=":"); ax.set_aspect("equal"); ax.set_title(t)
    ax.set_xlabel("x")
axs[0].set_ylabel("y")
fig.suptitle("Comparison of interpolation methods on the full face")
fig.tight_layout(); fig.savefig("figures/fig7_comparison.png", dpi=170); plt.close()

# Fig 8 - console screenshots (output split into two images)
flat = report.split("\n")
split = next(i for i, l in enumerate(flat) if l.startswith("--- Interpolated values"))
for name, chunk in (("fig8a_console_output", flat[:split]), ("fig8b_console_output", flat[split:])):
    fig = plt.figure(figsize=(7.2, 0.158 * len(chunk) + 0.25))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.015, 0.985, "\n".join(chunk), family="monospace", fontsize=9, color="#d4d4d4", va="top")
    fig.savefig(f"figures/{name}.png", dpi=170, facecolor="#1e1e1e"); plt.close()

print("\nFigures saved to ./figures/")
