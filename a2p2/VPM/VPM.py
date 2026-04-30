import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

U_inf = 1.5
alpha = 0.0
U      = U_inf * np.cos((np.pi / 180) * alpha)
V      = U_inf * np.sin((np.pi / 180) * alpha)
# 'skiprows=1' ignores the airfoil name/header on the first line
# 'unpack=True' allows you to assign columns directly to x and y
xp, yp = np.loadtxt('a2p2/MYFOIL.dat', skiprows=1, unpack=True)
xp = xp[::-2]
yp = yp[::-2]
xc   = (xp[:-1] + xp[1:]) / 2       # control points xc at panel midpoints
yc   = (yp[:-1] + yp[1:]) / 2 
n = len(xc)      # number of panels
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
    B = np.maximum(eps1**2 + eta**2, 1e-15)
    Num = np.maximum(eps2**2 + eta**2, 1e-15)
    Cl = (1/(2*np.pi)) * 0.5 * np.log(Num / B)
    Ct = (1/(2*np.pi)) * (np.arctan2(eta, eps2) - np.arctan2(eta, eps1))
    return Cl, Ct

# ── Build influence matrices ────
K = np.zeros((len(xc), len(xc)))
L = np.zeros((len(xc), len(xc)))
for i in range(len(xc)):
    for j in range(len(xc)):
        if i == j:
            K[i, j] = 0
            L[i, j] = 0.5
        else:
            dxj = xc[i] - xp[j];  dyj = yc[i] - yp[j]
            eps1 = dxj*np.cos(phi[j]) + dyj*np.sin(phi[j])
            eta = -dxj*np.sin(phi[j]) + dyj*np.cos(phi[j])
            Cl, Ct = _ct_cl(eps1, eta, leng[j])
            L[i, j] = -Cl*(np.sin(phi[i] - phi[j])) - Ct*(np.cos(phi[i] - phi[j]))
            K[i, j] = -Cl*(np.cos(phi[i] - phi[j])) + Ct*(np.sin(phi[i] - phi[j]))

# ── Solve for vorticity ─────────────────
K[-1] = np.zeros(len(xc))  
K[-1][0] = 1/(2*np.pi)
K[-1][-1] = 1/(2*np.pi)
RHS = -U_inf*np.cos(beta)
RHS[-1] = 0
G = np.linalg.solve(K, RHS)

# ── Surface pressure coefficient ──────────────────────────────────
Vt = U_inf*np.sin(beta) + np.dot(L, G)
Vt_up = Vt[yc >= 0]; Cp_up = 1 - (Vt_up / U_inf)**2
Vt_lo = Vt[yc < 0]; Cp_lo = 1 - (Vt_lo / U_inf)**2

# ── Velocity field (vectorised over panels) ─────────
Ng     = 40
x_grid = np.linspace(-0.4, 2, Ng)
y_grid = np.linspace(-0.4, 0.4, Ng)
X, Y   = np.meshgrid(x_grid, y_grid)
R2     = X**2 + Y**2
mask   = R2 > 0          

ug = np.full_like(X, np.nan)
vg = np.full_like(Y, np.nan)

px, py  = X[mask], Y[mask]
u_f = np.full(px.shape, U)
v_f = np.full(py.shape, V)

for j in range(len(xc)):
    dxj   = px - xp[j];  dyj = py - yp[j]
    xi  = dxj*np.cos(phi[j]) + dyj*np.sin(phi[j])
    eta = -dxj*np.sin(phi[j]) + dyj*np.cos(phi[j])
    Cl, Ct = _ct_cl(xi, eta, leng[j])
    u_f += G[j] * (Cl*np.sin(phi[j]) - Ct*np.cos(phi[j]))
    v_f += G[j] * (-Cl*np.cos(phi[j]) - Ct*np.sin(phi[j]))

#print(xc, xc.shape, len(xc))
ug[mask] = u_f
vg[mask] = v_f

x_up = xc[yc >= 0]
x_lo = xc[yc < 0]
idx_u = np.argsort(x_up)
idx_l = np.argsort(x_lo)
x_upper = x_up[idx_u]
cp_upper = Cp_up[idx_u]
x_lower = x_lo[idx_l]
cp_lower = Cp_lo[idx_l]


# __ Cp distribution plot ___________________________________________
plt.figure()
plt.plot(x_upper[1:-1], cp_upper[1:-1], 'o-', label='Cp at control points (Upper Surface)')
plt.plot(x_lower, cp_lower, 's-', label='Cp at control points (Lower Surface)')
plt.gca().invert_yaxis()
plt.xlabel('x');  plt.ylabel('Cp')
plt.title('Pressure Coefficient Distribution over Non-Lifting Cylinder — Source Panel Method')
plt.savefig(f'a2p2/VPM/Cp_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(xc[1:-1], Vt[1:-1], 'o-', label='Tangential Velocity at control points')
plt.xlabel('x');  plt.ylabel('Tangential Velocity')
plt.title('Tangential Velocity Distribution over Non-Lifting Cylinder — Source Panel Method')
plt.savefig(f'a2p2/VPM/tangential_velocity.png', dpi=300, bbox_inches='tight')
plt.show()


# __ Export CSVs _______________________________________________
pd.DataFrame(ug).to_csv(f'a2p2/VPM/u.csv', index=False, header=False)
pd.DataFrame(vg).to_csv(f'a2p2/VPM/v.csv', index=False, header=False)

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
headrow_K = pd.DataFrame([["Normal Velocities = 0"] + [""]*(K.shape[1] - 1)])
headrow_K_mat = pd.DataFrame([["Normal Influence Matrix (I_ij)"] + [""]*(K.shape[1] - 1)])
df_K_mat = pd.DataFrame(2*np.pi*K)  
headrow_rhs_K = pd.DataFrame([[""]*K.shape[1], ["RHS Values_Normal"] + [""]*(K.shape[1] - 1)])
df_K_rhs = pd.DataFrame([-2*np.pi*RHS])
headrow_G = pd.DataFrame([[""]*K.shape[1], ["Source Strengths"] + [""]*(K.shape[1] - 1)])
df_G = pd.DataFrame([G])
headrow_L = pd.DataFrame([[""]*L.shape[1], [""]*L.shape[1], ["Computing Tangential Velocities & C_p"] + [""]*(K.shape[1] - 1)])
headrow_L_mat = pd.DataFrame([["Tangential Influence Matrix (L_ij)"] + [""]*(L.shape[1] - 1)])
df_L = pd.DataFrame(2*np.pi*L)
headrow_rhs_L = pd.DataFrame([[""]*L.shape[1], ["Freestream_Tangential"] + [""]*(L.shape[1] - 1)])
df_L_rhs = pd.DataFrame([2*np.pi*U_inf*np.sin(beta)])
headrow_rhs_Vt = pd.DataFrame([[""]*L.shape[1], ["Tangential Velocities"] + [""]*(L.shape[1] - 1)])
df_Vt = pd.DataFrame([Vt])
headrow_rhs_Cp = pd.DataFrame([[""]*L.shape[1], ["Pressure Coefficients"] + [""]*(L.shape[1] - 1)])
df_Cp = pd.DataFrame([Cp_up]+[Cp_lo])
df_mat = pd.concat([headrow_K, headrow_K_mat, df_K_mat, headrow_rhs_K, df_K_rhs,
                    headrow_G, df_G,
                    headrow_L, headrow_L_mat, df_L, headrow_rhs_L, df_L_rhs,
                    headrow_rhs_Vt, df_Vt, headrow_rhs_Cp, df_Cp], ignore_index=True, axis = 0)


with pd.ExcelWriter('a2p2/VPM/vortex_panel_method.xlsx', mode='w', engine='openpyxl') as writer:
    df_panel.to_excel(writer, sheet_name='Panel_Data', index=False, header = True)
    df_mat.to_excel(writer, sheet_name='Influence_Matrices', index=False, header=False)


# __ Streamline plot ___________________________________________
n_seeds  = 20
y_seeds  = np.linspace(-0.39, 0.39, n_seeds)
start_pts = np.column_stack([np.full(n_seeds, -0.39), y_seeds])

fig, ax = plt.subplots(figsize=(6, 4))
ax.streamplot(X, Y, ug, vg,
              start_points=start_pts,
              density=0.5,
              linewidth=1.0,
              color='steelblue',
              arrowsize=1.0,
              integration_direction='forward',
              broken_streamlines=False,
              maxlength=100.0,
              minlength=0.0)
ax.fill(xp, yp, color='lightgray', zorder=5)
ax.plot(xp, yp, 'k-', linewidth=1.5, zorder=6)
ax.set_aspect('equal')
ax.set_xlim(-0.4, 2);  ax.set_ylim(-0.4, 0.4)
ax.set_title('Flow over Non-Lifting Cylinder — Source Panel Method', fontsize=13)
ax.set_xlabel('x');  ax.set_ylabel('y')
plt.tight_layout()
plt.savefig(f"a2p2/VPM/streamlines.png", dpi=300, bbox_inches='tight')
plt.show()
