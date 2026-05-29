import numpy as np
from shapely.geometry import  LineString
from astropy import units as u
from astropy import constants as const 
import pandas as pd
from scipy import stats
from numba import njit, prange
import pickle

#######################
#tables# 
#######################
f_k_i = pd.read_csv('f(xi).csv',names=("k_i","fk_i"))
Cross_Section_pg = pd.read_csv('cross_section.csv',names=('Ph_En','C_S'))
kp_pg = pd.read_csv('kp_pg.txt',names=('e','k'),sep=" ")

### Tabulated functions for photomeson interactions ###
with open('interpolated_data_f_2g.pkl', 'rb') as int_2g:
    f_2g, x_l, y_l, Z_2g = pickle.load(int_2g)  
    
with open('interpolated_data_f_e_p.pkl', 'rb') as int_e_p:
    f_e_p, x_l, y_l, Z_e_p = pickle.load(int_e_p)  
    
with open('interpolated_data_f_e_m.pkl', 'rb') as int_e_m:
    f_e_m, x_l, y_l, Z_e_m = pickle.load(int_e_m)  
    
with open('interpolated_data_f_b_nu_mu.pkl', 'rb') as int_b_nu_mu:
    f_b_nu_mu, x_l, y_l, Z_b_nu_mu = pickle.load(int_b_nu_mu)  
    
with open('interpolated_data_f_b_nu_e.pkl', 'rb') as int_b_nu_e:
    f_b_nu_e, x_l, y_l, Z_b_nu_e = pickle.load(int_b_nu_e)  
    
with open('interpolated_data_f_nu_mu.pkl', 'rb') as int_nu_mu:
    f_nu_mu, x_l, y_l, Z_nu_mu = pickle.load(int_nu_mu)  
    
with open('interpolated_data_f_nu_e.pkl', 'rb') as int_nu_e:
    f_nu_e, x_l, y_l, Z_nu_e = pickle.load(int_nu_e) 
    
#######################
#constants# 
#######################
G = (const.G).cgs.value       
c = (const.c).cgs.value     
Ro = (const.R_sun).cgs.value            
Mo = (const.M_sun).cgs.value       
yr = (u.yr).to(u.s)                
kpc = (u.kpc).to(u.cm)             
pc = (u.pc).to(u.cm)              
m_pr = (u.M_p).to(u.g)         
m_el = (u.M_e).to(u.g)         
kb = (const.k_B).cgs.value
h = (const.h).cgs.value 
q = (const.e.gauss).value                
sigmaT = (const.sigma_T).cgs.value               
eV = (u.eV).to(u.erg)   
B_cr = 2*np.pi*m_el**2*c**3/(h*q)
h_0 = 0.313
r = 0.1458
E_th_pi = 1.22*10**(-3.) #in TeV

#Unints of rest mass in TeV
x_m_pr = 938.272046*10**(-6.)
x_m_el = 0.511*10**(-6.)
x_m_pi = 139.57*10**(-6.) #charged pion 
x_m_pi0 = 134.9766*10**(-6.) #neutral pion 
x_m_mu = 105.66*10**(-6.)
K_pi = 0.17

E_nu_space = np.logspace(10.,22.,50)*eV
Cross_Section_pg_Ph_En = Cross_Section_pg.Ph_En[1:].values * 1e9 * eV
Cross_Section_pg_C_S = Cross_Section_pg.C_S[1:].values
kp_pg_e = kp_pg.e.values * 1e9 * eV
kp_pg_k = kp_pg.k.values
#Read parameters file
fileName = "Parameters.txt"
fileObj = open(fileName)
params = {}
for line in fileObj:
    line=line.strip()
    key_value = line.split("=")
    params[key_value[0].strip()] = float(key_value[1].strip())
    
time_init = float(params['time_init'])     # initial time in code units ~ R0/c
time_end = float(params['time_end'])       # final time in code units ~ R0/c
step_alg = float(params['step_alg'])       # step size for the PDE solver ~ R0/c
PL_inj = float(params['PL_inj'])           # flag controlling injection profile PL or exp cut_off

g_min_el = float(params['g_min_el'])       # log10(min electron Lorentz factor)
g_max_el = float(params['g_max_el'])       # log10(max electron Lorentz factor)
g_el_PL_min = float(params['g_el_PL_min']) # log10(min electron PL for injection)
g_el_br = float(params['g_el_br'])         # log10(break Lorentz factor, e-) IF 0 -> normal power law from g_el_PL_min to g_el_PL_max
g_el_PL_max = float(params['g_el_PL_max']) # log10(max electron PL for injection)
grid_g_el = float(params['grid_g_el'])     # number of electron grid points

g_min_pr = float(params['g_min_pr'])       # log10(min proton Lorentz factor)
g_max_pr = float(params['g_max_pr'])       # log10(max proton Lorentz factor)
g_pr_PL_min = float(params['g_pr_PL_min']) # log10(min proton PL for injection)
g_pr_br = float(params['g_pr_br'])         # log10(break Lorentz factor, p) IF 0 -> normal power law from g_pr_PL_min to g_pr_PL_max
g_pr_PL_max = float(params['g_pr_PL_max']) # log10(max proton PL for injection)
grid_g_pr = float(params['grid_g_pr'])     # number of proton grid points
grid_nu = float(params['grid_nu'])         # number of frequency grid points

p_el1 = float(params['p_el_1'])            # 1st electron power-law index
p_el2 = float(params['p_el_2'])            # 2nd electron power-law index
L_el = float(params['L_el'])               # log10 of electron luminosity

p_pr1 = float(params['p_pr_1'])            # 1st proton power-law index
p_pr2 = float(params['p_pr_2'])            # 2nd proton power-law index
L_pr = float(params['L_pr'])               # log10 of proton luminosity

Vexp = float(params['Vexp'])*c             # expansion velocity of the blob in units of c
R0 = 10**float(params['R0'])               # log10 of radius in cm
B0 = float(params['B0'])                   # initial magnetic field in Gauss
m = float(params['m'])                     # exponent controlling B(R)
delta = float(params["delta"])             # Doppler factor (if relevant)
inj_flag = float(params["inj_flag"])       # controls whether injection is on/off

# Flags for energy-loss processes:
Ad_l_flag = float(params['Ad_l_flag'])     # adiabatic losses on/off
Syn_l_flag = float(params['Syn_l_flag'])   # synchrotron losses on/off
Syn_emis_flag = float(params['Syn_emis_flag'])  # synchrotron emission on/off
IC_l_flag = float(params['IC_l_flag'])     # inverse Compton losses on/off
IC_emis_flag = float(params['IC_emis_flag'])# inverse Compton emission on/off
SSA_l_flag = float(params['SSA_l_flag'])   # synchrotron self-absorption on/off
gg_flag = float(params['gg_flag'])         # gamma-gamma annihilation on/off

# Photohadronic channels:
pg_pi_l_flag = float(params['pg_pi_l_flag'])      # p-gamma pion losses on/off
pg_pi_emis_flag = float(params['pg_pi_emis_flag'])# p-gamma pion emission on/off
pg_BH_l_flag = float(params['pg_BH_l_flag'])      # Bethe-Heitler losses on/off
pg_BH_emis_flag = float(params['pg_BH_emis_flag'])# Bethe-Heitler emission on/off

# Proton-proton interactions:
n_H = float(params['n_H'])                # ambient cold proton density
pp_l_flag = float(params['pp_l_flag'])    # p-p interaction losses on/off
pp_ee_emis_flag = float(params['pp_ee_emis_flag']) # e+ e- from p-p on/off
pp_g_emis_flag = float(params['pp_g_emis_flag'])   # gamma from p-p on/off
pp_nu_emis_flag = float(params['pp_nu_emis_flag']) # neutrinos from p-p on/off

neutrino_flag = float(params['neutrino_flag'])     # controls p-gamma neutrino emission
esc_flag_el = float(params['esc_flag_el'])         # electron escape on/off
esc_flag_pr = float(params['esc_flag_pr'])         # proton escape on/off

# Blackbody or greybody photon field from external source:
BB_flag = float(params['BB_flag'])                # blackbody on/off
temperature = 10**float(params['temperature'])     # blackbody temperature in K (log10)
GB_ext = float(params['GB_ext'])                   # factor for external BB normalization

# External power-law photon field:
PL_flag = float(params['PL_flag'])      # external power-law on/off
dE_dV_ph = float(params['dE_dV_ph'])    # power-law photon energy density factor
nu_min_ph = float(params['nu_min_ph'])  # power-law photon min freq exponent
nu_max_ph = float(params['nu_max_ph'])  # power-law photon max freq exponent
s_ph = float(params['s_ph'])           # power-law photon spectral index

# External user-supplied photon field:
User_ph = float(params['User_ph'])     # user-defined field on/off 

grid_size_pr = grid_g_pr
g_pr = np.logspace(g_min_pr,g_max_pr,int(grid_size_pr))
g_pr_mp = np.array([(g_pr[im+1]+g_pr[im-1])/2. for im in range(0,len(g_pr)-1)])
dg_pr = np.array([((g_pr[im+1])-(g_pr[im-1]))/2. for im in range(1,len(g_pr)-1)])
dg_l_pr = np.log(g_pr[1])-np.log(g_pr[0])

#######################
#functions# 
#######################

# expanding source radius 
@njit(cache=True, fastmath=True)
def R(R0,t,t_i,Vexp):
    return R0+Vexp*(t-t_i)

# magnetic field strength
@njit(cache=True, fastmath=True)
def B(B0,R0,R,m):
    return B0*(R0/R)**m

# volume of spherical source
@njit(cache=True, fastmath=True)
def Volume(R):
    return (4.0 / 3.0) * np.pi * R ** 3

@njit(cache=True, fastmath=True)
def build_log_grid(log10_min: float, log10_max: float, n: int):
    x = np.logspace(log10_min, log10_max, n)
    # midpoint array for adjacent bins
    x_mid = np.array([(x[im+1]+x[im-1])/2. for im in range(0,len(x)-1)])
    # central-difference widths with safe edges
    dx = np.array([((x[im+1])-(x[im-1]))/2. for im in range(1,len(x)-1)])
    dlog = np.log(x[1]) - np.log(x[0])
    return x, x_mid, dx, dlog

@njit(cache=True, fastmath=True)
def find_injection_bounds(grid: np.ndarray, log10_min: float, log10_max: float) -> tuple[int, int]:
    lo_val = 10 ** log10_min
    hi_val = 10 ** log10_max
    lo = max(1, int(np.searchsorted(grid, lo_val, side="left")))
    hi = min(len(grid) - 1, int(np.searchsorted(grid, hi_val, side="right")))
    return lo, hi

@njit(cache=True, fastmath=True)
def inj_spectrum(V_R0, g, index_min, index_max, p1, L_log10, mass, PL_inj, g_br_log10, p2):
    """
    inj : 1D array
        Injection array on the same grid as g.
    """
    inj = np.zeros_like(g)
    L = 10.0**L_log10
    g_cut = g[index_max]
    if g_br_log10 != 0:
        g_br = 10**g_br_log10 
        index_br = np.abs(g - g_br).argmin()
        gb = g[index_br]   
        
        I1 = np.trapz(g[:index_br+1]**(1.0 - p1), g[:index_br+1])
        if PL_inj == 1:
            I2 = gb**(p2 - p1) * np.trapz(g[index_br+1:index_max]**(1.0 - p2), g[index_br+1:index_max])
            A = L / (mass * c**2 * (I1 + I2))
            inj[index_min:index_br+1] = (A * g[index_min:index_br+1]**(-p1) / V_R0)
            inj[index_br+1:index_max] = (A * gb ** (p2 - p1) * g[index_br+1:index_max]**(-p2) / V_R0)
        else: 
            I2 = gb**(p2 - p1) * np.trapz(g[index_br+1:]**(1.0 - p2) * np.exp(-g[index_br+1:] / g[index_max]), g[index_br+1:])
            A = L / (mass * c**2 * (I1 + I2))
            inj[index_min:index_br+1] = (A * g[index_min:index_br+1]**(-p1) / V_R0)
            inj[index_br+1:] = A * gb ** (p2 - p1) * g[index_br+1:]**(-p2) * np.exp(-g[index_br+1:] / g[index_max])/V_R0
    else:
        if PL_inj == 1:
            inj[index_min:index_max] = L/np.trapz(mass*c**2.*g[index_min:index_max]**(-p1+2.),np.log(g[index_min:index_max]))*g[index_min:index_max]**(-p1)/V_R0
        else:
            inj[index_min:] = L/np.trapz(mass*c**2.*g[index_min:]**(-p1+2.)*np.exp(-g[index_min:]/g_cut),np.log(g[index_min:]))*g[index_min:]**(-p1) * np.exp(-g[index_min:]/g[index_max])/V_R0
    return inj
    
@njit(cache=True, fastmath=True)
def interp_log_numba(x, xp, fp):
    result = np.empty_like(x)
    for i in range(len(x)):
        xi = x[i]
        idx = np.searchsorted(xp, xi) - 1
        if idx < 0:
            idx = 0
        elif idx >= len(xp) - 1:
            idx = len(xp) - 2
        x0 = xp[idx]
        x1 = xp[idx + 1]
        y0 = fp[idx]
        y1 = fp[idx + 1]
        result[i] = y0 + (xi - x0) * (y1 - y0) / (x1 - x0)
    return result

@njit(cache=True, fastmath=True)
def trapz_log_numba(y, x):
    s = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        s += 0.5 * (y[i + 1] + y[i]) * dx
    return s

#Synchrotron critical frequency (Radiative Processes in Astrophysics, by George B. Rybicki, Alan P. Lightman, Wiley-VCH , June 1986.)
@njit(cache=True, fastmath=True)
def nu_c_numba(gamma, B):
    return (3.*q*B*gamma**2)/(4.*np.pi*m_el*c)

#Synchrotron emissivity dN/dVdνdtdΩ (Relativistic Jets from Active Galactic Nuclei, by M. Boettcher, D.E. Harris, ahd H. Krawczynski, Berlin: Wiley, 2012)
@njit(cache=True, fastmath=True)
def Q_syn_space_single(Np, B, nu, a_cr, C_syn, ln_g, g13, g2):
    integrand = Np * g13 * np.exp(-nu / (a_cr * g2))
    return C_syn * B**(2./3.) * nu**(-2./3.) * trapz_log_numba(integrand, ln_g)

@njit(parallel=True, cache=True, fastmath=True)
def Q_syn_space(Np, B, nu_arr, a_cr, C_syn, ln_g, g13, g2):
    out = np.empty(len(nu_arr) - 2)
    for k in prange(1, len(nu_arr) - 1):
        out[k - 1] = Q_syn_space_single(Np, B, nu_arr[k], a_cr, C_syn, ln_g, g13, g2)
    return out
    
@njit(cache=True, fastmath=True)
def Q_IC_single(Np, g_el, nu_ic_temp, photons, nu_targ, ln_nu_targ, ln_g_el):
    E_ic = h * nu_ic_temp
    temp_int_in_en = np.zeros(len(g_el))
    for i in range(len(g_el)):
        gamma = g_el[i]
        E_e = gamma * m_el * c**2
        denom = E_e - E_ic
        if denom <= 0.0:
            continue
        epsilon = E_ic / denom
        eps2 = epsilon * epsilon
        q_denom = 4.0 * nu_targ * gamma * (gamma - E_ic / (m_el * c**2))
        q_IC = nu_ic_temp / q_denom
        valid = q_IC > 0.0
        if not np.any(valid):
            continue
        qv = q_IC[valid]
        fn = 2.0*qv*np.log(qv) + (1.0 + 2.0*qv)*(1.0 - qv) + (eps2/(1.0 + epsilon))*(1.0 - qv)/2.0
        fn = np.maximum(fn, 0.0)
        temp_int_in_en[i] = trapz_log_numba(photons[valid] * fn, ln_nu_targ[valid])
    integrand_final = Np * temp_int_in_en / g_el
    integrand_final = np.nan_to_num(integrand_final, nan=0.0)
    return 0.75 * sigmaT * c * trapz_log_numba(integrand_final, ln_g_el)

@njit(parallel=True, cache=True, fastmath=True)
def Q_IC(Np, g_el, nu_ic, photons, nu_targ):
    out = np.empty(len(nu_ic) - 2)
    ln_nu_targ = np.log(nu_targ)
    ln_g_el = np.log(g_el)
    for k in prange(1, len(nu_ic) - 1):
        out[k - 1] = Q_IC_single(Np, g_el, nu_ic[k], photons, nu_targ, ln_nu_targ, ln_g_el)
    return out

@njit(parallel=True, cache=True, fastmath=True)
def aSSA(N_el, B, nu_arr, g, dg_l_el):
    out = np.empty(len(nu_arr) - 2)
    nu_c_g = nu_c_numba(g[1:-1], B)
    ln_g_mid = np.log(g[1:-1])
    diff_N = (N_el[2:] - N_el[1:-1]) / dg_l_el

    for k in prange(1, len(nu_arr) - 1):
        nu = nu_arr[k]
        pref = q**3 * B * nu**(-5./3.) / (2.0 * m_el**2 * c**2 * 0.8975)
        factor = (2.0 * nu_c_g)**(-1./3.) * np.exp(-nu / nu_c_g)
        out[k - 1] = -abs(pref * trapz_log_numba(factor * (diff_N - 2.0 * N_el[1:-1]), ln_g_mid))
    return out

#Synchrotron self absorption coefficient (High Energy Radiation from Black Holes: Gamma Rays, Cosmic Rays, and Neutrinos by Charles D. Dermer and Govind Menon. Princeton Univerisity Press, 2009) delta approximation
def aSSA_delta_approx(N_el,B,nu,g,dg_l_el):
    gamma = np.sqrt(nu*np.pi*m_el*c/(q*B))
    return np.pi*c*q**2./(36.*m_el*c**2.*nu*gamma)*interp_log_numba(np.log10(gamma),np.log10(g[1:-1]),np.diff(N_el[1:])/dg_l_el-N_el[1:-1])

#Synchrotron self absorption coefficient (Eq. 40 in Mastichiadis A., Kirk J. G., 1995, A\&A, 295, 613)
def aSSA_delta_approx_M_K95(N_el,B,nu,g,dg_l_el):
    b = B/B_cr
    x = h*nu/(m_el*c**2.)
    gamma = np.sqrt(x/b)
    return 137.*np.pi/(6.)*(x*b)**(-1./2.)*gamma**(-3.)*interp_log_numba(np.log10(gamma),np.log10(g[1:-1]),np.diff(N_el[1:])/dg_l_el-2.*N_el[1:-1])*sigmaT

#Synchrotron self absorption frequency (determined by solving tau_ssa = 1)
def SSA_frequency(index_SSA,nu,aSSA_space,R):
    if index_SSA>0:
        line_1 = LineString(np.column_stack((np.log10(nu[index_SSA-2:index_SSA+1]),np.log10(np.multiply(aSSA_space,-R)[index_SSA-2:index_SSA+1]))))
        line_2 = LineString(np.column_stack((np.log10(nu[index_SSA-2:index_SSA+1]),np.zeros([3]))))
        int_pt = line_1.intersection(line_2)
        return (10**int_pt.x) 

#gamma-gamma absorption coefficient  ( Coppi P. S., Blandford R. D., 1990, MNRAS, 245, 453. doi:10.1093/mnras/245.3.453) 
@njit(cache=True, fastmath=True)
def a_gg(nu_ic, nu_target, photons_target):
    t_gg_m = np.zeros(len(nu_ic))
    x = h * nu_target / (m_el * c**2)
    log10_x = np.log10(x)
    log10_photons = np.log10(photons_target /h*m_el*c**2)
    max_x = np.max(x)
    for i in prange(len(nu_ic)):
        nu_ic_el = nu_ic[i]
        x_IC = h*nu_ic_el/(m_el*c**2)
        start = np.log10(1.3/x_IC)
        stop = np.log10(max_x)
        if start < stop:
            x_space = np.logspace(start, stop, 100)
            photons_gg_log10 = interp_log_numba(x_space, x, log10_photons)
            photons_gg = 10**photons_gg_log10
            photons_gg = np.nan_to_num(photons_gg, nan=0.)
            xICxspace = x_space * x_IC
            integrand = 0.652*sigmaT*((xICxspace**2-1.)/xICxspace**3.)*np.log(xICxspace)*x_space*photons_gg
            log_x_space = np.log(x_space)
            t_gg_m[i] = trapz_log_numba(integrand, log_x_space)
        else: t_gg_m[i] = 0.0
        t_gg_m = np.nan_to_num(t_gg_m,nan=0.)
    return t_gg_m
            
#Pair creation from gamma gamma absorption dN/dVdtdg (Eq. 57 in Mastichiadis A., Kirk J. G., 1995, A\&A, 295, 613)
@njit(cache=True, fastmath=True)
def Q_ee_f(nu_target, photons_target, nu_ic, photons_IC, g, R0):
    Q_ee_temp = np.empty(len(g))
    log_nu_target = np.log10(h * nu_target / (m_el * c ** 2))
    log_photons_target_modified = np.log10(photons_target * m_el * c ** 2 / h)
    log_nu_ic = np.log10(nu_ic)
    log_photons_IC = np.log10(photons_IC)
    for i in range(len(g)):
        g_e = g[i]
        if 2.*g_e > 1.0:
            x_prime_min = (2.*g_e)**(-1.)
            x_prime_max = h * nu_target[-1] / (m_el * c ** 2)
            x_prime = np.logspace(np.log10(x_prime_min), np.log10(x_prime_max), 100)
            log_x_prime = np.log10(x_prime)
            # Interpolate n_ph_prime
            n_ph_prime = 10**interp_log_numba(log_x_prime, log_nu_target, log_photons_target_modified)
            n_ph_prime = np.nan_to_num(n_ph_prime, nan=0.0)
            n_d_u = 2.*g_e*x_prime
            # Interpolate n_g
            x_n_g = np.array([np.log10(2.*g_e * m_el * c**2 / h)])
            n_g_array = 10**interp_log_numba(x_n_g, log_nu_ic, log_photons_IC)
            n_g_array = np.nan_to_num(n_g_array, nan=0.0)
            n_g = n_g_array[0]
            # Compute R_gg
            numerator = n_d_u**2-1.
            denominator = n_d_u**3
            log_n_d_u = np.log(n_d_u)
            R_gg = 2.61*numerator/denominator*log_n_d_u
            integrand = n_ph_prime * R_gg * x_prime
            Q_ee_temp[i] = n_g * trapz_log_numba(integrand, log_x_prime) * np.log(10.)
        else:
            Q_ee_temp[i] = 0.0
    factor = c*sigmaT*m_el*c**2/h
    return Q_ee_temp * factor

#computes energy density of target photons for ICS scattering      
@njit(cache=True, fastmath=True)
def U_ph_KN(g, nu_target, photons_target):
    m_el_c2 = m_el*c**2
    U_ph_tot = np.empty(len(g))
    for i in range(len(g)):
        integrand = photons_target*(h*nu_target)*fKN_exact(4*g[i]*h*nu_target/m_el_c2)
        U_ph_tot[i] = trapz_log_numba(integrand*nu_target, np.log(nu_target))    
    return U_ph_tot

    
@njit(cache=True, fastmath=True)
def U_ph_f(g, nu_target, photons_target, R):
    U_ph_temp = np.empty(len(g))
    m_el_c2 = m_el*c**2
    U_ph_tot = m_el_c2/(sigmaT*R)*trapz_log_numba(nu_target[:-1]*photons_target[:-1]*(sigmaT*R*m_el_c2/h),nu_target[:-1])
    for i in range(len(g)):
        l_f = g[i]
        nu_T = 3.*m_el_c2/(4.*h*l_f)
        if nu_T < nu_target[-1]:
            nu_temp = np.logspace(np.log10(nu_target[0]), np.log10(nu_T), 50)
            # Interpolate photons_temp
            photons_temp = 10**interp_log_numba(np.log10(nu_temp),np.log10(nu_target),np.log10(photons_target))
            photons_temp = np.nan_to_num(photons_temp, nan=0.0)
            n_d_targ = h*nu_temp/m_el_c2
            integrand = n_d_targ*photons_temp*(sigmaT*R*m_el_c2/h)
            U_ph_temp[i] = m_el_c2/(sigmaT*R)*trapz_log_numba(integrand, n_d_targ)
        else:
            U_ph_temp[i] = U_ph_tot
    return U_ph_temp

@njit(cache=True, fastmath=True)
def li2_series_pos(x, n=200):
    s = 0.0
    p = x
    for k in range(1, n + 1):
        if k > 1:
            p *= x
        s += p / (k * k)
    return s

@njit(cache=True, fastmath=True)
def li2_series_minus_z(z, n=200):
    s = 0.0
    p = -z
    for k in range(1, n + 1):
        if k > 1:
            p *= -z
        s += p / (k * k)
    return s


@njit(cache=True, fastmath=True)
def Li2_minus_z(z):
    if z < 1.0:
        return li2_series_minus_z(z)
    elif z < 10.0:
        x = z / (1.0 + z)
        return -0.5 * np.log1p(z)**2 - li2_series_pos(x)
    else:
        invz = 1.0 / z
        return (-np.pi**2 / 6.0  - 0.5 * np.log(z)**2 - li2_series_minus_z(invz))


@njit(cache=True, fastmath=True)
def g_KN(b):
    return ((0.5 * b + 6.0 + 6.0 / b) * np.log1p(b)- ((11.0 / 12.0) * b**3. + 6.0 * b**2 + 9.0 * b + 4.0 ) / (1.0 + b)**2 - 2.0 + 2.0 * Li2_minus_z(b))

@njit(cache=True, fastmath=True)
def fKN_exact(b):
    out = np.empty_like(b)
    for i in range(len(b)):
        bi = b[i]
        if bi < 1e-3:
            out[i] = 1.0 - 63.0 * bi / 40.0 + 441.0 * bi**2 / 200.0
        else:
            out[i] = 9.0 * g_KN(bi) / bi**3
            if out[i] < 0.0:
                out[i] = 0.0
            elif out[i] > 1.0:
                out[i] = 1.0
    return out

@njit(cache=True, fastmath=True)
def U_ph_KN(g, nu_target, photons_target):
    m_el_c2 = m_el * c**2
    U_ph_tot = np.empty(len(g))
    for i in range(len(g)):
        b = 4.0 * g[i] * h * nu_target / m_el_c2
        integrand = (photons_target * h * nu_target * fKN_exact(b))
        U_ph_tot[i] = trapz_log_numba(integrand * nu_target, np.log(nu_target))
    return U_ph_tot

@njit(cache=True, fastmath=True)
def f_kn_moderski(b):
    return (1.0 + b)**(-1.5)

@njit(cache=True, fastmath=True)
def U_ph_KN_fast(g, nu_target, photons_target):
    m_el_c2 = m_el * c**2
    eps0 = h * nu_target / m_el_c2
    ueps = photons_target * h * nu_target   # energy density per frequency bin, as in your convention

    out = np.empty(len(g))
    lognu = np.log(nu_target)

    for i in range(len(g)):
        b = 4.0 * g[i] * eps0
        integrand = ueps * f_kn_moderski(b)
        out[i] = trapz_log_numba(integrand * nu_target, lognu)
    return out
    
@njit(cache=True, fastmath=True)
def dg_dt_BH(g_pr, nu, photons, ln_k_i, ln_fk_i):
    C_BH = 3.*sigmaT*c*m_el/(8.*np.pi*137.*m_pr)
    dg_dt_pg = np.zeros(len(g_pr))

    # Precompute logarithms
    ln_nu_scaled = np.log10(h*nu/(m_el*c**2))
    ln_photons_log10 = np.log10(photons)

    for idx in range(len(g_pr)):
        g_p = g_pr[idx]
        if g_p*h*nu[-1] > (2.1)*m_el*c**2.:
            kappa_int = np.logspace(np.log10(2.), np.log10(g_p*h*nu[-1]/(m_el*c**2.)), 50)
            # Interpolate photons_BH
            x_interp = np.log10(kappa_int/(2.*g_p))
            ln_photons_BH_log10 = interp_log_numba(x_interp, ln_nu_scaled, ln_photons_log10)
            photons_BH = (m_el*c**2/h)*10**ln_photons_BH_log10
            photons_BH = np.nan_to_num(photons_BH , nan=0.0)
            # Interpolate f_k
            f_k_log10 = interp_log_numba(np.log10(kappa_int), ln_k_i, ln_fk_i)
            f_k = 10 ** f_k_log10
            integrand = f_k * photons_BH * kappa_int
            integral = trapz_log_numba(integrand, np.log10(kappa_int))
            dg_dt_pg[idx] = integral
        else:
            dg_dt_pg[idx] = 0.0
    return dg_dt_pg * C_BH

@njit(cache=True, fastmath=True)
def dg_dt_pg_approx(g_pr, nu, photons, E_th=145e6 * eV):
#---- Κ*σ constant = 70 microbarns from Atoyan and Dermer 2003, ApJ--------
    sigma_eff = 7e-29  # Effective cross-section
    h_nu = h*nu # Photon energies
    h_nu_max = h_nu[-1]
    dg_dt_pg = np.zeros(len(g_pr))
    for idx in range(len(g_pr)):
        g_p = g_pr[idx]
        if g_p*h_nu_max > E_th:
            epsilon_prime_space = np.logspace(np.log10(E_th/(2.*g_p)),np.log10(h_nu_max/2.),100)
            dN_dVd_epsilon_prime = interp_log_numba(np.log10(epsilon_prime_space),np.log10(h_nu),(photons/h))
            dN_dVd_epsilon_prime = np.nan_to_num(dN_dVd_epsilon_prime, nan=0.0)
            dg_dt_pg[idx] = sigma_eff*c/g_p*trapz_log_numba(dN_dVd_epsilon_prime*(g_p**2.-(E_th**2./(2.*epsilon_prime_space**2.)))*epsilon_prime_space,np.log(epsilon_prime_space))
        else:
            dg_dt_pg[idx] = 0.0
    return dg_dt_pg

@njit(cache=True, fastmath=True)
def dg_dt_pg(g_pr, nu, photons,E_th=145.*10**6.*eV):
    C_pion = 5.*10**(-31.)*c
    h_nu = h*nu
    h_nu_last = h_nu[-1]
    dg_dt_pg = np.zeros_like(g_pr)
    # Precompute logs for interpolation
    for idx_g in prange(len(g_pr)):
        g_p = g_pr[idx_g]
        if g_p*h_nu_last > E_th:
            ε_bar_min = E_th
            ε_bar_max = 2.*g_p*h_nu_last
            ε_bar_space = np.logspace(np.log10(ε_bar_min), np.log10(ε_bar_max), 100)
            log_ε_bar_space = np.log10(ε_bar_space)
            Cross_Section_pg_int = interp_log_numba(log_ε_bar_space, np.log10(Cross_Section_pg_Ph_En), Cross_Section_pg_C_S)
            kp_pg_int = interp_log_numba(log_ε_bar_space, np.log10(kp_pg_e), kp_pg_k)
            Cross_Section_pg_int = np.nan_to_num(Cross_Section_pg_int, nan=0.0)
            kp_pg_int = np.nan_to_num(kp_pg_int, nan=0.0)
            int_pg_losses = np.zeros_like(ε_bar_space)
            for idx in range(len(ε_bar_space)):
                if ε_bar_space[idx]/(2.*g_p) < h_nu_last/2.:
                    ε_prime_space = np.logspace(np.log10(ε_bar_space[idx]/(2.*g_p)),np.log10(h*nu[-1]),30)
                    dN_dVdε_prime = 10**interp_log_numba(np.log10(ε_prime_space),np.log10(h_nu),np.log10(photons/h))
                    dN_dVdε_prime = np.nan_to_num(dN_dVdε_prime, nan=0.0)
                    int_pg_losses[idx] = trapz_log_numba(dN_dVdε_prime/ε_prime_space,np.log(ε_prime_space))
                else:
                    int_pg_losses[idx] = 0.
            dg_dt_pg[idx_g] = (1./(g_p)*trapz_log_numba(Cross_Section_pg_int*kp_pg_int*int_pg_losses*ε_bar_space**2.,np.log(ε_bar_space)))
    return dg_dt_pg*C_pion

def Phi_g_mod(eta_l, x, flag_product):
    x_log10 = np.array([np.log10(x)])  # Shape (1,)
    eta_log10 = np.log10(eta_l)        # Shape (N,)
    if flag_product == "2_g":
        Z = f_2g(x_log10, eta_log10)   # Z has shape (N, 1)
    elif flag_product == "e-":
        Z = f_e_m(x_log10, eta_log10)
    elif flag_product == "e+":
        Z = f_e_p(x_log10, eta_log10)
    elif flag_product == "\bar_nu_e":
        Z = f_b_nu_e(x_log10, eta_log10)
    elif flag_product == "\bar_nu_mu":
        Z = f_b_nu_mu(x_log10, eta_log10)
    elif flag_product == "nu_mu":
        Z = f_nu_mu(x_log10, eta_log10)
    elif flag_product == "nu_e":
        Z = f_nu_e(x_log10, eta_log10)        
    # Include other conditions as needed
    else:
        Z = np.zeros((len(eta_log10), 1))
    Z_interpolated = Z[:, 0]           # Extract the interpolated values
    return Z_interpolated


# Emissivities of the secondary particles from Eq. (30) Kelner S.R. and Aharonian F. A. 2009  
def Qp_g_mod(g_el,nu_ic,N_pr,g_pr,photons_targ,nu_target,flag_product,h_0=0.313,r=0.1458):
    p_re = m_pr*c**2
    e_re = m_el*c**2.
    g_pr_min_int = h_0*p_re/(4.*h*nu_target[-1])
    g_pr_temp = np.logspace(np.log10(max(g_pr_min_int,g_pr[0]))+0.01,np.log10(g_pr[-1]),50)
    N_pr_temp = 10**interp_log_numba(np.log10(g_pr_temp),np.log10(g_pr),np.log10(N_pr))
    c_temp_l = p_re/(4.*g_pr_temp)
    eta_0_max_l = 4.*h*nu_target[-1]*g_pr_temp/p_re
    N_pr_temp = np.nan_to_num(N_pr_temp, nan=0.0)
    ln_g_pr_temp = np.log(g_pr_temp)
    if flag_product == "2_g": 
        sum_Qp_g = []
        for nu_g in nu_ic:
            Qp_g_temp = []
            for g_ind, (c_temp, eta_0_max) in enumerate(zip(c_temp_l, eta_0_max_l)):
                x = h*nu_g/(g_pr_temp[g_ind]*p_re)
                epsilon_0 = h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**interp_log_numba(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    photons_targ_temp = np.nan_to_num(photons_targ_temp, nan=0.0)
                    Qp_g_temp.append(trapz_log_numba(photons_targ_temp*10**Phi_g_mod(eta_space,x,flag_product)*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)
            sum_Qp_g.append(trapz_log_numba(h*N_pr_temp*Qp_g_temp/p_re,ln_g_pr_temp))
    
    elif flag_product == "e-" :
        sum_Qp_g = []
        for g in g_el:
            Qp_g_temp = []
            for g_ind, (c_temp, eta_0_max) in enumerate(zip(c_temp_l, eta_0_max_l)):
                x = g*e_re/(g_pr_temp[g_ind]*p_re)
                epsilon_0 = 2.14*h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(2.14*h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**interp_log_numba(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    Qp_g_temp.append(trapz_log_numba(photons_targ_temp*10**Phi_g_mod(eta_space,x,flag_product)*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)
            sum_Qp_g.append(trapz_log_numba(e_re*N_pr_temp*Qp_g_temp/p_re,ln_g_pr_temp))
                
    elif flag_product == "e+":     
        sum_Qp_g = []
        for g in g_el:
            Qp_g_temp = []
            for g_ind, (c_temp, eta_0_max) in enumerate(zip(c_temp_l, eta_0_max_l)):
                x = g*e_re/(g_pr_temp[g_ind]*p_re)
                epsilon_0 = h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**interp_log_numba(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    Qp_g_temp.append(trapz_log_numba(photons_targ_temp*10**Phi_g_mod(eta_space,x,flag_product)*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)
            sum_Qp_g.append(trapz_log_numba(e_re*N_pr_temp*Qp_g_temp/p_re,ln_g_pr_temp))
                    
    elif flag_product == "\bar_nu_e":
        sum_Qp_g = []
        g_pr_min_int = 2.14*h_0*p_re/(4.*E_nu_space[-1])
        g_pr_temp = np.logspace(np.log10(max(g_pr_min_int,g_pr[0])),np.log10(g_pr[-1]),50)
        N_pr_temp = 10**interp_log_numba(np.log10(g_pr_temp),np.log10(g_pr),np.log10(N_pr))
        N_pr_temp = np.nan_to_num(N_pr_temp, nan=0.0)
        for E_nu_ind in range(0,len(E_nu_space)):
            Qp_g_temp = []
            for g_ind, (c_temp, eta_0_max) in enumerate(zip(c_temp_l, eta_0_max_l)):
                x = E_nu_space[E_nu_ind]/(g_pr_temp[g_ind]*p_re)
                epsilon_0 = 2.14*h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(2.14*h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**interp_log_numba(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    photons_targ_temp = np.nan_to_num(photons_targ_temp, nan=0.0)
                    Qp_g_temp.append(trapz_log_numba(photons_targ_temp*10**Phi_g_mod(eta_space,x,flag_product)*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)                
            sum_Qp_g.append(trapz_log_numba(h*N_pr_temp*Qp_g_temp/p_re,ln_g_pr_temp))
    else:
        sum_Qp_g = []
        for E_nu_ind in range(0,len(E_nu_space)):
            Qp_g_temp = []
            for g_ind, (c_temp, eta_0_max) in enumerate(zip(c_temp_l, eta_0_max_l)):
                x = E_nu_space[E_nu_ind]/(g_pr_temp[g_ind]*p_re)
                epsilon_0 = h_0*c_temp
                if np.log10(epsilon_0/h) < np.log10(nu_target[-1]):
                    eta_space = np.logspace(np.log10(h_0),np.log10(eta_0_max),25)
                    nu_target_temp = eta_space*c_temp/h
                    photons_targ_temp = 10**interp_log_numba(np.log10(nu_target_temp),np.log10(nu_target),np.log10(photons_targ))
                    photons_targ_temp = np.nan_to_num(photons_targ_temp, nan=0.0)
                    Qp_g_temp.append(trapz_log_numba(photons_targ_temp*10**Phi_g_mod(eta_space,x,flag_product)*nu_target_temp,np.log(h*nu_target_temp)))
                else:
                    Qp_g_temp.append(0.)                
            sum_Qp_g.append(trapz_log_numba(h*N_pr_temp*Qp_g_temp/p_re,ln_g_pr_temp))
    return(np.array(sum_Qp_g))

@njit(cache=True, fastmath=True)
def q_BH_numba(g, x, g_p, R0):
    # g: scalar (logarithm)
    # x: array (logarithm)
    # g_p: scalar (logarithm)
    # R0: scalar
    
    E = 10.0 ** x       # E is an array
    gamma = 10.0 ** g   # gamma is a scalar
    gamma_p = 10.0 ** g_p  # gamma_p is a scalar
    
    # Constants
    x0_const = 0.6586
    ss = 0.65
    ss2 = -0.06073489219556636
    a_norm = 0.9893670856189293
    p_s = 0.94
    
    const = 2.0 - ss2 * x0_const ** p_s - a_norm / np.exp(1.0) * x0_const ** (-ss)
    log_gamma_p_E = np.log10(gamma_p * E)
    slope = (a_norm * log_gamma_p_E ** (-ss) * np.exp(-log_gamma_p_E / x0_const) +
             ss2 * log_gamma_p_E ** p_s + const)
    
    # Additional Constants
    A = -2.12
    B = 0.8975274693141311
    C = -0.051033221749056536
    D = 0.999057
    E_const = -111.9
    F = 1.22
    G = 0.1288184966128324
    p_e = 1.23
    A_1 = 17.5
    p_1 = 1.8
    c_1 = -3.68493
    
    tr = 4.0 * np.pi * c * R0 / (3.0 * sigmaT)
    
    A_norm = (A * np.exp(-B * (np.log10(gamma_p * E)) ** 2.0) +
              C * (np.log10(gamma_p * E)) ** D +
              E_const + p_e * x +
              G * (g_p) ** F +
              np.log10((1e15 / R0) ** 4 * tr) +
              (A_1 * (np.log10(gamma_p * E)) ** p_1) * np.exp(c_1 * np.log10(gamma_p * E)))
    
    # Parameters
    int_thres = 2.0
    x0 = (1.23 * E) ** (-1)
    a1 = 0.47
    a2 = 0.468
    a2_b = 0.43
    a3 = 0.465
    cor1a = 0.95
    cor1b = 1.0
    cor2a = 0.1
    cor2b = 0.0
    cor2b_thres = 0.14
    cor2c = 0.008
    x_c = gamma_p * 15.0
    p_lim = 1.75
    AM_flag = 1.0
    
    res = np.zeros(len(E))
    for i in range(len(x)):
        A_val = 10.0 ** A_norm[i]
        denom = 2.0 * a2 ** 2 - 2.0 * a3 ** 2
        if denom != 0.0:
            delta = 0.5 * (1.0 / a2 ** 2 - 1.0 / a3 ** 2)
            exponent = np.log10(x_c / x0[i]) ** slope[i] * delta
            cont_const = np.exp(-exponent) / np.exp(x_c * cor2b / x0[i])
        else:
            cont_const = 1.0
        if gamma_p * E[i] >= int_thres and gamma_p * E[i] * AM_flag < 1e4:
            if gamma < gamma_p * m_pr / m_el:
                power = min(2.0, slope[i])
                log_gamma_over_x0 = np.log10(gamma / x0[i])
                if gamma / x0[i] <= 1.0:
                    power = 2.0
                    cor = cor1a
                    a_slope = a1
                    cor2 = cor2a
                    res[i] = (A_val * np.exp(-log_gamma_over_x0 ** power / (2.0 * a_slope ** 2)) *
                              np.exp(-((x0[i] - gamma) * cor / gamma) ** 2) *
                              np.exp(-gamma * cor2 / x0[i]))
                elif gamma < x_c or power > p_lim:
                    if power > p_lim:
                        cor = cor1b
                        cor2 = max(cor2b_thres - 0.0665 * (gamma_p * E[i] - 2.2), 0.007)
                        a_slope = a2
                    else:
                        cor = cor1b
                        cor2 = cor2b
                        a_slope = a2
                    res[i] = (A_val * np.exp(-log_gamma_over_x0 ** power / (2.0 * a_slope ** 2)) *
                              np.exp(-((x0[i] - gamma) * cor / gamma) ** 2) *
                              np.exp(-cor2 * (gamma / x0[i] - 1.0)))
                else:
                    cor = cor1b
                    a_slope = a3
                    cor2 = cor2c
                    res[i] = (A_val * np.exp(-log_gamma_over_x0 ** power / (2.0 * a_slope ** 2)) *
                              np.exp(-((x0[i] - gamma) * cor / gamma) ** 2) *
                              np.exp(-cor2 * (gamma / x_c - 1.0)) * cont_const)
    return res


@njit(cache=True, fastmath=True)
def Q_BH_sol(g_el, g_pr, N_pr, nu_trgt, photons_trgt):
    dnde = np.zeros(len(g_el))
    ln_nu_trgt = np.log(nu_trgt)
    ln_g_pr = np.log(g_pr)
    log_nu_trgt_h = np.log10(nu_trgt*h/(m_el*c**2))
    

    for i in prange(len(g_el)):
        g_pr_int = np.zeros(len(g_pr))
        log_g_el_i = np.log10(g_el[i])
        for j in range(len(g_pr)):
            q_bh_val = q_BH_numba(log_g_el_i, log_nu_trgt_h, np.log10(g_pr[j]), R0)
            integrand = photons_trgt*nu_trgt*q_bh_val
            trapz_result = trapz_log_numba(integrand, ln_nu_trgt)
            g_pr_int[j] = N_pr[j]*Volume(R0)*trapz_result
        dnde[i] = trapz_log_numba(g_pr_int*g_pr, ln_g_pr)
    return dnde                               

#computes correction factor in electron injection luminosity by checking the energy balance in a fast synchrotron cooling scenario
def cor_factor_syn_el(g_el_space,R0,B0,p_el,Lum_e_injected):
    #Constants
    time_init = 0.
    time_end = 20.
    step_alg = 1.

    g_el = g_el_space
    g_el_mp = np.array([(g_el[im+1]+g_el[im-1])/2. for im in range(0,len(g_el)-1)])
    const_el = 4./3.*sigmaT/(8.*np.pi*m_el*c)
    dg_el = np.array([((g_el[im+1])-(g_el[im-1]))/2. for im in range(1,len(g_el)-1)])

    #gamma-nu space
    nu_syn = np.logspace(7.5,np.log10(7.*nu_c_numba(g_el[-1],B0))+1.4,100)
    day_counter=0.

    #Initialize values for particles and photons
    N_el = np.zeros(len(g_el))
    el_inj = np.zeros(len(g_el))
    photons_syn = np.ones(len(nu_syn))*10**(-260.)
    N_el[0] = N_el[-1] = 10**(-160.)
    
    El_lum = []
    Ph_lum = []
    t = []
    t_plot = []
    time_real = time_init
    el_inj = Q_el_Lum(Lum_e_injected,p_el,g_el[1],g_el[-2])*g_el**(-p_el)
    C_syn_el = sigmaT*c/(h*24.*np.pi**2.*0.8975)*(4.*np.pi*m_el*c/(3.*q))**(4./3.)
    while time_real <  time_end*R0/c:
        dt = 0.1001*R0/c
        time_real += dt    
        t.append(time_real)
        Radius = R0
        B = B0
        a_cr_el = 3.*q*B/(4.*np.pi*m_el*c)
        
        b_syn_el = const_el*B**2.
        dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2.),dg_el)
        dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2.),dg_el)

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius+dgdt_Syn_el_m)
        V3 = -dt*(dgdt_Syn_el_p) 
        S_ij = N_el[1:-1]+el_inj[1:-1].copy()*dt
        N_el[1:-1] = thomas_numba(V1, V2, V3, S_ij)

        Q_Syn_el = [Q_syn_space(N_el/Volume(Radius),B,nu_syn[nu_ind],a_cr_el,C_syn_el,g_el) for nu_ind in range(len(nu_syn)-1)] 

        V1 = np.zeros(len(nu_syn)-2)
        V2 = 1.+dt*(c/R0*np.ones(len(nu_syn)-2))
        V3 = np.zeros(len(nu_syn)-2)
        S_ij = photons_syn[1:-1]+4.*np.pi*np.multiply(Q_Syn_el,dt)[1:]*Volume(Radius)
        photons_syn[1:-1] = thomas_numba(V1, V2, V3, S_ij )  
        
        if day_counter < time_real:
            day_counter=day_counter+step_alg*R0/c
            t_plot.append(time_real)
            Syn_temp_plot = np.multiply(photons_syn/Volume(Radius),h*nu_syn**2.)*4.*np.pi/3.*Radius**2.*c
            El_lum.append(np.trapz(el_inj*g_el**2.*m_el*c**2.,np.log(g_el)))
            Ph_lum.append(np.trapz(Syn_temp_plot,np.log(nu_syn)))
    return np.divide(Ph_lum,El_lum)[-1]

p_pr_list = [2.,2.5,3.]
eta_list = [1.1 , 0.86, 0.91]

def n_cor(tau,Radius,J_e):
    return tau*np.sqrt(3.)/(J_e*sigmaT*Radius)

def cs_pp_inel(E_p):
    mask = np.array(E_p) > E_th_pi
    L = np.log(np.array(E_p))
    cs_pp_temp = np.zeros(len(E_p))
    cs_pp_temp[mask] = 10**(-27.)*(34.3+1.88*L[mask]+0.25*np.array(L[mask])**2.)*(1.-(np.divide(E_th_pi,E_p[mask]))**4.)**2 
    return cs_pp_temp

def q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr):
    if species == "g":
        if p_pr == 2.:
            eta = 1.1
        elif p_pr == 2.5:
            eta = 0.86
        elif p_pr == 3.:
            eta = 0.91
        else:
            eta = interp_log_numba(p_pr,eta_list,p_pr_list)
    else:
        if p_pr == 2.:
            eta = 0.77
        elif p_pr == 2.5:
            eta = 0.62
        elif p_pr == 3.:
            eta = 0.67        
        else:
            eta = interp_log_numba(p_pr,eta_list,p_pr_list)
            
    return eta*c*n_H/K_pi*cs_pp_inel(x_m_pr+E_pi/K_pi)*10.**(interp_log_numba(np.log10(x_m_pr+E_pi/K_pi),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV)))


def g_nu_mu(x,r):
    return (3.-2.*r)/(9.*(1.-r)**2.)*(9.*x**2.-6.*np.log(x)-4.*x**3.-5.)

def h_nu_mu_1(x,r):
    return (3.-2.*r)/(9.*(1.-r)**2.)*(9.*r**2.-6.*np.log(r)-4.*r**3.-5.)

def h_nu_mu_2(x,r):
    return (1.+2.*r)*(r-x)/(9.*r**2.)*(9.*(r+x)-4.*(r**2.+r*x+x**2.))

def g_nu_e(x,r):
    return 2.*(1.-x)/(3.*(1.-r)**2.)*(6.*(1.-x)**2.+r*(5.+5.*x-4.*x**2.)+6.*r*np.log(x))

def h_nu_e_1(x,r):
    return 2./(3.*(1.-r)**2.)*((1.-r)*(6.-7.*r+11.*r**2.-4.*r**3.)+6.*r*np.log(r))

def h_nu_e_2(x,r):
    return 2.*(r-x)/(3.*r**2.)*(7.*r**2.-4.*r**3.+7.*x*r**2.-2.*x**2.-4.*x**2.*r)

def Q_pp_sub2(x_space,species,E_p,E,p_pr,N_pr_TeV,n_H,g_pr):
    L = np.log(E_p)
    
    if species == "g":
        B_g = 1.3+0.14*L+0.011*L**2.
        beta_g = 1./(1.79+0.11*L+0.008*L**2.)
        k_g = 1./(0.801+0.049*L+0.014*L**2.)
        F_g = B_g*np.log(x_space)/x_space*((1.-x_space**beta_g)/(1.+k_g*x_space**beta_g*(1.-x_space**beta_g)))**4.*(1./np.log(x_space)-4.*beta_g*x_space**beta_g/(1.-x_space**beta_g)-
                (4.*k_g*beta_g*x_space**beta_g*(1.-2.*x_space**beta_g))/(1.+k_g*x_space**beta_g*(1.-x_space**beta_g)))
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_g,np.log(x_space))
    
    elif species == "nu_mu_1":
        y_space = np.divide(x_space,0.427)
        B_g = 1.75+0.204*L+0.01*L**2.
        beta_g = 1./(1.67+0.111*L+0.0038*L**2.)
        k_g = 1.07-0.086*L+0.002*L**2.

        B_e = 1./(69.5+2.65*L+0.3*L**2.)
        beta_e = 1./(0.201+0.062*L+0.00042*L**2.)**(1./4.)
        k_e = (0.279+0.141*L+0.0172*L**2.)/(0.3+(2.3+L)**2.)
        F_nu_mu_1 = B_g*np.log(y_space)/y_space*((1.-y_space**beta_g)/(1.+k_g*y_space**beta_g*(1.-y_space**beta_g)))**4.*(1./np.log(y_space)-4.*beta_g*y_space**beta_g/(1-y_space**beta_g)-
                (4*k_g*beta_g*y_space**beta_g*(1.-2.*y_space**beta_g))/(1.+k_g*y_space**beta_g*(1.-y_space**beta_g)))
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_nu_mu_1,np.log(x_space))

    elif species == "nu_mu_2":
        y_space = np.divide(x_space,0.427)
        B_g = 1.75+0.204*L+0.01*L**2.
        beta_g = 1./(1.67+0.111*L+0.0038*L**2.)
        k_g = 1.07-0.086*L+0.002*L**2.

        B_e = 1./(69.5+2.65*L+0.3*L**2.)
        beta_e = 1./(0.201+0.062*L+0.00042*L**2.)**(1./4.)
        k_e = (0.279+0.141*L+0.0172*L**2.)/(0.3+(2.3+L)**2.)
        F_nu_mu_2 = B_e*(1.+k_e*(np.log(x_space))**2.)**3./(x_space*(1.+0.3/x_space**beta_e))*(-np.log(x_space))**5.
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_nu_mu_2,np.log(x_space))


    elif species == "e" or species == "nu_e" :
        B_e = 1./(69.5+2.65*L+0.3*L**2.)
        beta_e = 1./(0.201+0.062*L+0.00042*L**2.)**(1./4.)
        k_e = (0.279+0.141*L+0.0172*L**2.)/(0.3+(2.3+L)**2.)
        F_e =  B_e*(1.+k_e*(np.log(x_space))**2.)**3./(x_space*(1.+0.3/x_space**beta_e))*(-np.log(x_space))**5.
        return np.trapz(c*n_H*cs_pp_inel(E/x_space)*10.**np.interp(np.log10((E/x_space)),np.log10(g_pr*x_m_pr),np.log10(N_pr_TeV))*F_e,np.log(x_space))


    else:
        raise ValueError("Unkown particles species")
        
def Q_pp_sub1(x,species,E_p,E,p_pr,N_pr_TeV,n_H,g_pr):
    if species == "g": 
        E_g = np.multiply(x,E_p)
        E_min  = E_g+x_m_pi0**2./(4.*E_g)
        E_pi = np.logspace(np.log10(min(E_min)),3.,40)
        return 2.*np.trapz(q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi0**2.)*E_pi,np.log(E_pi))
    
    elif species == "nu_mu_1":
        E_nu_mu_1 = np.multiply(x,E_p)
        E_max = max(x)*(max(E_p)-x_m_pr)
        E_min_1 = E_nu_mu_1/(1.-(x_m_mu/x_m_pi)**2.)
        E_min_2 = E_nu_mu_1+x_m_pi**2./(4.*E_nu_mu_1) 
        E_min = max(max(E_min_1),max(E_min_2))
        E_pi = np.logspace(np.log10(E_min),np.log10(E_max),40)
        return 2./0.427*np.trapz(q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi**2.)*E_pi,np.log(E_pi))
    

    elif species == "nu_mu_2":
        f_nu_mu_2 = []
        r = (x_m_mu/x_m_pi)**2.
        E_nu_mu_2 = np.multiply(x,E_p)
        E_min = E_nu_mu_2+x_m_mu**2./(4.*E_nu_mu_2)
        E_pi = np.logspace(np.log10(max(E_min)),np.log10(1000.*max(E_min)),40)
        x_new = sorted(max(E_nu_mu_2)/E_pi)
        for x_element in x_new:
            if x_element > r:
                f_nu_mu_2.append(g_nu_mu(x_element,r))
            else:
                f_nu_mu_2.append(h_nu_mu_1(x_element,r)+h_nu_mu_2(x_element,r))
        f_nu_mu_2_norm = (1./np.trapz(f_nu_mu_2,x_new))
        return 2.*np.trapz(np.multiply(f_nu_mu_2_norm,f_nu_mu_2)*q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi**2.)*E_pi,np.log(E_pi))
    
    elif species == "e":
        f_nu_mu_2 = []
        r = (x_m_mu/x_m_pi)**2.
        E_e = np.multiply(x,E_p)
        E_min = E_e+x_m_el**2./(4.*E_e)
        E_pi = np.logspace(np.log10(max(E_min)),np.log10(1000.*max(E_min)),40)
        x_new = sorted(min(np.unique(E_e))/E_pi)
        for x_element in x_new:
            if x_element > r:
                if g_nu_mu(x_element,r)<0. :            
                    f_nu_mu_2.append(0.)
                else :   
                    f_nu_mu_2.append(g_nu_mu(x_element,r))
            else:
                if h_nu_mu_1(x_element,r)<0.:
                    flag_h_nu_mu_1 = 0.
                else:
                    flag_h_nu_mu_1  = 1.
                if h_nu_mu_2(x_element,r)<0.:
                    flag_h_nu_mu_1 = 0.
                else:
                    flag_h_nu_mu_2 = 1.
                f_nu_mu_2.append((flag_h_nu_mu_1*h_nu_mu_1(x_element,r)+flag_h_nu_mu_2*h_nu_mu_2(x_element,r)))
        f_nu_mu_2_norm = (1./np.trapz(f_nu_mu_2,x_new))
        return 2.*np.trapz(np.multiply(f_nu_mu_2_norm,f_nu_mu_2)*q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0.*x_m_pi**2.)*E_pi,np.log(E_pi))
    
    elif species == "nu_e":
        f_nu_e = []
        r = (x_m_mu/x_m_pi)**2.
        E_nu_e = np.multiply(x,E_p)
        E_min = E_nu_e+x_m_el**2./(4.*E_nu_e)
        E_pi = np.logspace(np.log10(max(E_min)),np.log10(100.*max(E_min)),40)
        x_new = sorted(max(E_nu_e)/E_pi)
        for x_element in x_new:
            if x_element > r:
                f_nu_e.append(g_nu_e(x_element,r))
            else:
                f_nu_e.append(h_nu_e_1(x_element,r)+h_nu_e_2(x_element,r))
        f_nu_e_norm = (1./np.trapz(f_nu_e,x_new))
        
        return 2.*np.trapz(np.multiply(f_nu_e_norm,f_nu_e)*q_pi(E_pi,p_pr,species,N_pr_TeV,n_H,g_pr)/np.sqrt(E_pi**2.-0*x_m_pi**2.)*E_pi,np.log(E_pi))
    
    else:
        raise ValueError("Unkown particles species")   
        
def Q_e_pp(g_el,g_pr,N_pr_TeV,p_pr,n_H):
    Q_pp_e_list = []
    norm_ind = 0
    E_species = g_el*m_el*c**2.*0.624151 #Electrons energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_e = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element<0.427:  
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element<1.: 
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_e += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_e += Q_pp_sub2(x_sub2,"e",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_e += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_e += Q_pp_sub1(x_sub1,"e",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_e += 0.
        Q_pp_e_list.append(Q_pp_temp_e)
    slope_e, intercept_e, r_value_e, p_value_e, std_err_e = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_e_list[norm_ind:norm_ind+2]))
    y_fit_e = 10**(slope_e*np.log10(E_species[norm_ind-1])+intercept_e)
    norm_e = y_fit_e/Q_pp_e_list[norm_ind-1]
    Q_pp_e_list[:norm_ind] = np.multiply(norm_e,Q_pp_e_list[:norm_ind])
    return Q_pp_e_list

def Q_g_pp(nu_ic,g_pr,N_pr_TeV,p_pr,n_H):
    norm_ind = 0
    E_species = h*nu_ic*0.624151 #Photons energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV   
    Q_pp_g_list = [] 
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_g = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element < 0.427  :
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element < 1.:
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_g += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_g += Q_pp_sub2(x_sub2,"g",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_g += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_g += Q_pp_sub1(x_sub1,"g",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_g += 0.
        Q_pp_g_list.append(Q_pp_temp_g)
    slope_g, intercept_g, r_value_g, p_value_g, std_err_g = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_g_list[norm_ind:norm_ind+2]))
    y_fit_g = 10**(slope_g*np.log10(E_species[norm_ind-1])+intercept_g)
    norm_g = y_fit_g/Q_pp_g_list[norm_ind-1]
    Q_pp_g_list[:norm_ind] = np.multiply(norm_g,Q_pp_g_list[:norm_ind])
    return Q_pp_g_list

def Q_nu_mu_pp(nu_nu,g_pr,N_pr_TeV,p_pr,n_H):
    norm_ind = 0
    E_species = h*nu_nu*0.624151 #Neutrinos energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV   
    Q_pp_nu_mu_list = []
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_nu_mu = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element < 0.427 and x_element > 10**(-3.) :  #Neutrinos from the deay of pions continue up to 0.427Epi
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element < 10**(-3.):   #delta function approximation
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_nu_mu += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_nu_mu += Q_pp_sub2(x_sub2,"nu_mu_1",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)+Q_pp_sub2(x_sub2,"nu_mu_2",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr)  
        else:
            Q_pp_temp_nu_mu += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_nu_mu += Q_pp_sub1(x_sub1,"nu_mu_1",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)+Q_pp_sub1(x_sub1,"nu_mu_2",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_nu_mu += 0.
        Q_pp_nu_mu_list.append(Q_pp_temp_nu_mu)
    slope_nu_mu, intercept_nu_mu, r_value_nu_mu, p_value_nu_mu, std_err_nu_mu = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_nu_mu_list[norm_ind:norm_ind+2]))
    y_fit_nu_mu = 10**(slope_nu_mu*np.log10(E_species[norm_ind-1])+intercept_nu_mu)
    norm_nu_mu = y_fit_nu_mu/Q_pp_nu_mu_list[norm_ind-1]
    Q_pp_nu_mu_list[:norm_ind] = np.multiply(norm_nu_mu,Q_pp_nu_mu_list[:norm_ind])
    return Q_pp_nu_mu_list

def Q_nu_e_pp(nu_nu,g_pr,N_pr_TeV,p_pr,n_H):
    norm_ind = 0
    E_species = h*nu_nu*0.624151 #Photons energies in TeV
    E_p_space = g_pr*m_pr*c**2.*0.624151 #Protons energies in TeV   
    Q_pp_nu_e_list = []
    for E in E_species:
        x_sub1 = []
        x_sub2 = []
        x_sub3 = []
        Q_pp_temp_nu_e = 0.
        for i in range(0,len(E_p_space)):
            x_element = E/E_p_space[i]
            if   E > 0.1 and x_element < 0.427  and x_element > 10**(-3.):
                x_sub2.append(x_element)
            elif  E < 0.1 and x_element < 10**(-3.):
                x_sub1.append(x_element)
            else:
                x_sub3.append(0.)
                Q_pp_temp_nu_e += 0.

        if len(x_sub2) > 0.:
            x_sub2 = sorted(x_sub2)
            Q_pp_temp_nu_e += Q_pp_sub2(x_sub2,"nu_e",E/x_sub2,E,p_pr,N_pr_TeV,n_H,g_pr) 
        else:
            Q_pp_temp_nu_e += 0.

        if len(x_sub1) > 0.:
            norm_ind += 1
            x_sub1 = sorted(x_sub1)    
            Q_pp_temp_nu_e += Q_pp_sub1(x_sub1,"nu_e",E/x_sub1,E,p_pr,N_pr_TeV,n_H,g_pr)
        else:
            Q_pp_temp_nu_e += 0.
        Q_pp_nu_e_list.append(Q_pp_temp_nu_e)
    slope_nu_e, intercept_nu_e, r_value_nu_e, p_value_nu_e, std_err_nu_e = stats.linregress(np.log10(E_species[norm_ind:norm_ind+2]),np.log10(Q_pp_nu_e_list[norm_ind:norm_ind+2]))
    y_fit_nu_e = 10**(slope_nu_e*np.log10(E_species[norm_ind-1])+intercept_nu_e)
    norm_nu_e = y_fit_nu_e/Q_pp_nu_e_list[norm_ind-1]
    Q_pp_nu_e_list[:norm_ind] = np.multiply(norm_nu_e,Q_pp_nu_e_list[:norm_ind])
    return Q_pp_nu_e_list

# converts flux to luminosity
def nuL_nu_obs(nu_F_nu,Dist_in_pc,delta,R0):
    return np.multiply(nu_F_nu,1.)*(4.*np.pi*(Dist_in_pc*pc)**2.)/delta**4.

# converts luminosity to flux
def nuF_nu_obs(nu_L_nu,Dist_in_pc,delta,R0):
    return np.multiply(nu_L_nu,delta**4.)/(4.*np.pi*(Dist_in_pc*pc)**2.)

#computes total photon spectrum by adding different spectral components
def photons_tot(nu_syn,nu_bb,photons_syn,nu_ic,photons_IC,nu_tot,photons_bb,photons_pl,photons_user):
    return 10**(interp_log_numba(np.log10(nu_tot),np.log10(nu_bb),np.log10(photons_bb)))+10**(interp_log_numba(np.log10(nu_tot),np.log10(nu_syn),np.log10(photons_syn)))+10**(interp_log_numba(np.log10(nu_tot),np.log10(nu_ic),np.log10(photons_IC)))+photons_pl+photons_user


@njit(cache=True, fastmath=True)
def thomas_numba(a, b, c, d):
    n = len(d)
    # Create copies of the arrays to avoid modifying the originals
    cp = np.zeros(n-1)
    dp = np.zeros(n)
    x = np.zeros(n)
    
    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    
    # Back substitution
    x[n-1] = dp[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    
    return x
    
