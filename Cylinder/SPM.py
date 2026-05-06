import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ___ Parameters _____________________
r      = 0.1        # cylinder radius
n      = 8          # number of panels
U_inf  = 1.5        # freestream speed
alpha  = 0.0        # angle of attack (degrees)
alpha  = alpha * np.pi / 180.0
U      = U_inf * np.cos(alpha)
V      = U_inf * np.sin(alpha)

# ── Panel geometry (CW ordering: π → -π) ───────
theta_panels = np.linspace(np.pi, -np.pi, n + 1)           # angles of panels' endpoints subtended at the cylinder center
xp, yp = r * np.cos(theta_panels), r * np.sin(theta_panels)# N+1 points defining the panels (x, y)
xc   = (xp[:-1] + xp[1:]) / 2       # control points xc at panel midpoints
yc   = (yp[:-1] + yp[1:]) / 2       # control points yc at panel midpoints
leng = np.sqrt(np.diff(xp)**2 + np.diff(yp)**2)       # panel lengths
phi  = np.arctan2(np.diff(yp), np.diff(xp))           # panel angles
phi[phi < 0] += 2*np.pi              # ensure in [0, 2π)
delta = phi + np.pi/2               # angle of outward normal
delta[delta >= 2*np.pi] -= 2*np.pi  # ensure in [0, 2π)
beta  = delta - alpha               # angle between normal and freestream

tx = np.cos(phi);  ty = np.sin(phi)   # unit tangent
nx = -ty;          ny =  tx           # outward normal for CW body

# __ Log and Tan term generator _______________________
def _ct_cl(eps1, eta, sj):
    """tangential (Ct) and log (Cn) terms."""
    eps2  = eps1 - sj
    B = np.maximum(eps1**2 + eta**2, 1e-20)
    Num = np.maximum(eps2**2 + eta**2, 1e-20)
    Cl = (1/(2*np.pi)) * 0.5 * np.log(Num / B)
    Ct = (1/(2*np.pi)) * (np.arctan2(eta, eps2) - np.arctan2(eta, eps1))
    return Cl, Ct

# ── Build influence matrices ────
I = np.zeros((n, n))
J = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            I[i, j] = 0.5
        else:
            dxj = xc[i] - xp[j];  dyj = yc[i] - yp[j]
            eps1 = dxj*np.cos(phi[j]) + dyj*np.sin(phi[j])
            eta = -dxj*np.sin(phi[j]) + dyj*np.cos(phi[j])
            Cl, Ct = _ct_cl(eps1, eta, leng[j])
            I[i, j] = Cl*(np.sin(phi[i] - phi[j])) + Ct*(np.cos(phi[i] - phi[j]))
            J[i, j] = -Cl*(np.cos(phi[i] - phi[j])) + Ct*(np.sin(phi[i] - phi[j]))

# ── Solve for source strengths ─────────────────
S = np.linalg.solve(I, -U_inf*np.cos(beta))

# ── Surface pressure coefficient ──────────────────────────────────
Vt = U_inf*np.sin(beta) + np.dot(J, S)
Cp = 1 - (Vt / U_inf)**2

# ── Velocity field (vectorised over panels) ─────────
Ng     = 400
x_grid = np.linspace(-3*r, 5*r, Ng)
y_grid = np.linspace(-4*r, 4*r, Ng)
X, Y   = np.meshgrid(x_grid, y_grid)
R2     = X**2 + Y**2
mask   = R2 > 0         

ug = np.full_like(X, np.nan)
vg = np.full_like(Y, np.nan)

px, py  = X[mask], Y[mask]
u_f = np.full(px.shape, U)
v_f = np.full(py.shape, V)

for j in range(n):
    dxj   = px - xp[j];  dyj = py - yp[j]
    xi  = dxj*np.cos(phi[j]) + dyj*np.sin(phi[j])
    eta = -dxj*np.sin(phi[j]) + dyj*np.cos(phi[j])
    Cl, Ct = _ct_cl(xi, eta, leng[j])
    u_f += S[j] * (-Cl*np.cos(phi[j]) - Ct*np.sin(phi[j]))
    v_f += S[j] * (-Cl*np.sin(phi[j]) + Ct*np.cos(phi[j]))

ug[mask] = u_f
vg[mask] = v_f

# __ Export CSVs _______________________________________________
pd.DataFrame(ug).to_csv(f'a2p1/n{n}/u_{n}.csv', index=False, header=False)
pd.DataFrame(vg).to_csv(f'a2p1/n{n}/v_{n}.csv', index=False, header=False)

# __ Export required data for grading __________________________
df_geom = pd.DataFrame({'SNo': np.arange(1, n+2), 'Panel Endpoints_x':np.round(xp, 4), 'Panel Endpoints_y':np.round(yp, 4)})
df_spce = pd.DataFrame({'':[""]*3})
df_panels = pd.DataFrame({'PanelID':np.arange(1, n+1), 
                         'Control Ponts_x': np.round(xc, 4), 'Control Points_y': np.round(yc, 4),
                         'Panel Angle (deg)': np.round(phi*180/np.pi, 2),
                         'Normal Angle (deg)': np.round(delta*180/np.pi, 2),
                         'Beta (deg)': np.round(beta*180/np.pi, 2),
                         'Panel Length': np.round(leng, 4),})
df_panel = pd.concat([df_geom, df_spce, df_panels], axis=1)
headrow_I = pd.DataFrame([["Normal Velocities = 0"] + [""]*(I.shape[1] - 1)])
headrow_I_mat = pd.DataFrame([["Normal Influence Matrix (I_ij)"] + [""]*(I.shape[1] - 1)])
df_I_mat = pd.DataFrame(2*np.pi*I)  
headrow_rhs_I = pd.DataFrame([[""]*I.shape[1], ["RHS Values_Normal"] + [""]*(I.shape[1] - 1)])
df_I_rhs = pd.DataFrame([-2*np.pi*U_inf*np.cos(beta)])
headrow_S = pd.DataFrame([[""]*I.shape[1], ["Source Strengths"] + [""]*(I.shape[1] - 1)])
df_S = pd.DataFrame([S])
headrow_J = pd.DataFrame([[""]*J.shape[1], [""]*J.shape[1], ["Computing Tangential Velocities & C_p"] + [""]*(I.shape[1] - 1)])
headrow_J_mat = pd.DataFrame([["Tangential Influence Matrix (J_ij)"] + [""]*(J.shape[1] - 1)])
df_J = pd.DataFrame(2*np.pi*J)
headrow_rhs_J = pd.DataFrame([[""]*J.shape[1], ["Freestream_Tangential"] + [""]*(J.shape[1] - 1)])
df_J_rhs = pd.DataFrame([2*np.pi*U_inf*np.sin(beta)])
headrow_rhs_Vt = pd.DataFrame([[""]*J.shape[1], ["Tangential Velocities"] + [""]*(J.shape[1] - 1)])
df_Vt = pd.DataFrame([Vt])
headrow_rhs_Cp = pd.DataFrame([[""]*J.shape[1], ["Pressure Coefficients"] + [""]*(J.shape[1] - 1)])
df_Cp = pd.DataFrame([Cp])
df_mat = pd.concat([headrow_I, headrow_I_mat, df_I_mat, headrow_rhs_I, df_I_rhs,
                    headrow_S, df_S,
                    headrow_J, headrow_J_mat, df_J, headrow_rhs_J, df_J_rhs,
                    headrow_rhs_Vt, df_Vt, headrow_rhs_Cp, df_Cp], ignore_index=True, axis = 0)


with pd.ExcelWriter(f'a2p1/n{n}/panel_geometry_{n}.xlsx', mode='w', engine='openpyxl') as writer:
    df_panel.to_excel(writer, sheet_name='Panel_Data', index=False, header = True)
    df_mat.to_excel(writer, sheet_name='Influence_Matrices', index=False, header=False)


# __ Velocity vs Theta plot ____________________________________
theta = np.linspace(0, 2*np.pi, 100)
plots = {delta[i]: (Vt[i], Cp[i]) for i in range(n)}
sortedplots = sorted(plots.items())
theta_plot, vals = zip(*sortedplots)
theta_plot = np.array(theta_plot)
vt_plot, cp_plot = zip(*vals)
vt_ana = 2*U_inf*np.sin(theta)  # Analytical Vt distribution for non-lifting cylinder
vt_plot = np.array(vt_plot)
cp_ana = 1 - (vt_ana / U_inf)**2
cp_plot = np.array(cp_plot)
plt.figure()
plt.plot(theta_plot*180/np.pi, vt_plot, 'o-', label='Tangential Velocity (Vt)')
plt.plot(theta*180/np.pi, vt_ana, 'r--', label='Analytical Vt')
plt.xlabel('Theta (degrees)')
plt.ylabel('Velocity (m/s)')
plt.title('Tangential Velocity Distribution')
plt.grid(True)
plt.savefig(f"a2p1/n{n}/Vt_vs_theta_{n}.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(theta_plot*180/np.pi, cp_plot, 's-', label='Pressure Coefficient (Cp)')
plt.plot(theta*180/np.pi, cp_ana, 'm--', label='Analytical Cp')
plt.xlabel('Theta (degrees)')
plt.ylabel('$C_p$')
plt.title('Pressure Coefficient Distribution')
plt.grid(True)
plt.savefig(f"a2p1/n{n}/Cp_vs_theta_{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# __ Streamline plot ___________________________________________
n_seeds  = 50
y_seeds  = np.linspace(-3.9*r, 3.9*r, n_seeds)
start_pts = np.column_stack([np.full(n_seeds, -2.9*r), y_seeds])

theta_cyl = np.linspace(0, 2*np.pi, n+1)
cx, cy    = r*np.cos(theta_cyl), r*np.sin(theta_cyl)

fig, ax = plt.subplots(figsize=(9, 9))
ax.streamplot(X, Y, ug, vg,
              start_points=start_pts,
              linewidth=1.0,
              color='steelblue',
              arrowsize=1.0,
             integration_direction='forward',
              broken_streamlines=False,
              maxlength=100.0,
              minlength=0.0)
ax.fill(cx, cy, color='lightgray', zorder=5)
ax.plot(cx, cy, 'k-', linewidth=1.5, zorder=6)
ax.set_aspect('equal')
ax.set_xlim(-3*r, 5*r);  ax.set_ylim(-4*r, 4*r)
ax.set_title('Flow over Non-Lifting Cylinder — Source Panel Method', fontsize=13)
ax.set_xlabel('x');  ax.set_ylabel('y')
plt.tight_layout()
plt.savefig(f"a2p1/n{n}/streamlines_{n}.png", dpi=300, bbox_inches='tight')
plt.show()

