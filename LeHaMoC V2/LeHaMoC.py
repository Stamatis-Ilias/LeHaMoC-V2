# This is the leptohadronic version of a radiative transfer code LeHaMoC developed by S. I. Stathopoulos 
# in collaboration with M. Petropoulou. When using this code, make reference to the
# following publication: Stathopoulos et al., 2023, A&A 

import numpy as np
import astropy.units as u
from astropy import constants as const 
from astropy.modeling.models import BlackBody
import pandas as pd
import sys
from tqdm import tqdm
import LeHaMoC_f as f # imports functions
import time
import warnings
warnings.filterwarnings("ignore")

#######################
#tables# 
#######################
f_k_i = pd.read_csv('f(xi).csv',names=("k_i","fk_i"))
Cross_Section_pg = pd.read_csv('cross_section.csv',names=('Ph_En','C_S'))
kp_pg = pd.read_csv('kp_pg.txt',names=('e','k'),sep=" ")

# ######################
# Data Preparation
# ######################
ln_k_i = f_k_i['k_i'].values  # Assuming k_i is already in log10(kappa)
ln_fk_i = f_k_i['fk_i'].values  # Assuming fk_i is in log10(f_kappa)

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
erg_to_TeV = 0.624151
E_rp_TeV = m_pr*c**2.*erg_to_TeV
################################

if len(sys.argv) != 1:
    print('incorrect parameters passed')
    print('try something like this')
    print('python LeHaMoC.py Parameters.txt out1')
    quit()

#Define output files
out1 = "Pairs_Distribution.txt"
out2 = "Photons_Distribution.txt"
out3 = "Protons_Distribution.txt"
out4 = "Neutrinos_Distribution.txt"
TINY = 1e-260

fileName = "Parameters.txt"
fileObj = open(fileName)
params = {}
for line in fileObj:
    line=line.strip()
    key_value = line.split("=")
    params[key_value[0].strip()] = float(key_value[1].strip())
    
# --------------------------------------------------------------------------
# Assign parameters to local variables. These are dimensionless or logs of
# physical quantities (e.g., time, Lorentz factor limits, luminosities, etc.).
# --------------------------------------------------------------------------
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

errors = []
if time_end <= time_init:
    errors.append("time_end must be greater than time_init")
if step_alg <= 0:
    errors.append("step_alg must be > 0")

if grid_g_el < 10:
    errors.append("grid_g_el must be at least 10")
if grid_g_pr < 10:
    errors.append("grid_g_pr must be at least 10")
if grid_nu < 10:
    errors.append("grid_nu must be at least 10")

if g_min_el >= g_max_el:
    errors.append("g_min_el must be smaller than g_max_el")
if g_min_pr >= g_max_pr:
    errors.append("g_min_pr must be smaller than g_max_pr")

if g_el_PL_min > g_el_PL_max:
    errors.append("g_el_PL_min must be <= g_el_PL_max")
if g_pr_PL_min > g_pr_PL_max:
    errors.append("g_pr_PL_min must be <= g_pr_PL_max")

if BB_flag and temperature <= 0:
    errors.append("temperature must be provided when BB_flag = 1")
if PL_flag and nu_min_ph >= nu_max_ph:
    errors.append("nu_min_ph must be smaller than nu_max_ph when PL_flag = 1")

if errors:
    raise ValueError("Invalid parameter file:\n  - " + "\n  - ".join(errors))

# ----------------------------------------------------------------------------
# Time initialization and volume
# ----------------------------------------------------------------------------
start_time = time.time()               # track CPU time
time_real = time_init                  # current time in code units
dt = step_alg*R0/c                     # PDE solver time step (in seconds)
day_counter = 0.                       # helps write output periodically
Radius = R0                            # initial radius
V_R0 = f.Volume(R0)                    # volume at R0 (for normalization)

# ---------------------------
# Particle grids
# ---------------------------
g_el, g_el_mp, dg_el, dg_l_el = f.build_log_grid(g_min_el, g_max_el, int(grid_g_el))
g_pr, g_pr_mp, dg_pr, dg_l_pr = f.build_log_grid(g_min_pr, g_max_pr, int(grid_g_pr))

index_PL_min_el, index_PL_max_el = f.find_injection_bounds(g_el, g_el_PL_min, g_el_PL_max)
index_PL_min_pr, index_PL_max_pr = f.find_injection_bounds(g_pr, g_pr_PL_min, g_pr_PL_max)

# ---------------------------
# Frequency grids
# ---------------------------    
nu_syn = np.logspace(7.5, np.log10(7.*f.nu_c_numba(g_el[-1],B0))+1.4, int(grid_g_el/2))
nu_op = np.logspace(10., 34., int(grid_g_el/2))
nu_nu = np.logspace(10.,22.,50) * eV/h
nu_tot = np.logspace(np.log10(nu_syn[0]), np.log10(max(nu_op[-1],nu_syn[-1])), int(grid_nu))

# ---------------------------
# External BB / GB field
# Units: (nu, dN/dVdnu)
# ---------------------------
if BB_flag == 0.:
    dN_dVdnu_BB = np.ones(2)*TINY
    nu_bb = np.array([nu_syn[0], nu_syn[-1]])
else:
    bb = BlackBody(temperature*u.K)
    nu_bb = np.array(np.logspace(np.log10(5.879*10**10*temperature)-6., np.log10(5.879*10**10*temperature)+1.5,60)*u.Hz)
    photons_bb = np.array(4.*np.pi/c*bb(nu_bb)/(h*nu_bb))                       
    GB_norm = np.trapz(photons_bb*h*nu_bb**2.,np.log(nu_bb))/(GB_ext) 
    dN_dVdnu_BB = photons_bb/GB_norm

# ---------------------------
# External PL field
# Units: (nu, dN/dVdnu)
# ---------------------------     
if PL_flag == 0.:
    dN_dVdnu_pl = np.zeros(len(nu_tot))
else:
    nu_ph_ext_sp = np.logspace(nu_min_ph,nu_max_ph,100)
    k_ph = dE_dV_ph/np.trapz(nu_ph_ext_sp**(-s_ph+1.), nu_ph_ext_sp)/h
    dN_dVdnu_pl_temp = k_ph*nu_ph_ext_sp**(-s_ph)
    dN_dVdnu_pl_temp[-1] = TINY
    dN_dVdnu_pl = 10**np.interp(np.log10(nu_tot),np.log10(nu_ph_ext_sp),np.log10(dN_dVdnu_pl_temp))
    mask = np.log10(nu_tot) < nu_min_ph
    dN_dVdnu_pl[mask] = TINY

# ---------------------------
# User-defined external field
# Units: (nu, dN/dVdnu)
# ---------------------------
if User_ph == 0.:
    dN_dVdnu_user = np.ones(len(nu_tot))*TINY
else:  
    Photons_spec_user = pd.read_csv('Photons_spec_user.txt',names=('logx','logy'),sep=",")
    nu_user = 10**np.array(Photons_spec_user.logx)
    dN_dVdnu_user_temp = 10**np.array(Photons_spec_user.logy)
    dN_dVdnu_user_temp[-1] = TINY
    dN_dVdnu_user = 10**np.interp(np.log10(nu_tot),np.log10(nu_user),np.log10(dN_dVdnu_user_temp))

# ---------------------------
# Initial Condition
# ---------------------------
N_el = np.zeros_like(g_el)
Q_ee = np.zeros(len(g_el) - 2)
el_inj = np.full_like(g_el, TINY)

N_pr = np.zeros_like(g_pr)
pr_inj = np.full_like(g_pr, TINY)

N_nu = np.zeros_like(nu_nu)
a_gg_f = np.zeros_like(nu_op)

photons_syn = np.full_like(nu_syn, TINY)
photons_op = np.full_like(nu_op, TINY)
photons = np.full_like(nu_tot, TINY)

photons_syn = np.append(photons_syn, TINY)
dN_dVdnu_BB = np.append(dN_dVdnu_BB, TINY)

nu_syn = np.append(nu_syn, nu_tot[-1])
nu_bb = np.append(nu_bb, nu_tot[-1])

nu_syn_mp = np.array([(nu_syn[im+1]+nu_syn[im-1])/2. for im in range(0,len(nu_syn)-1)])
nu_op_mp = np.array([(nu_op[im+1]+nu_op[im-1])/2. for im in range(0,len(nu_op)-1)])

dnu = np.array([(nu_syn[nu_ind+1]-nu_syn[nu_ind-1])/2. for nu_ind in range(1,len(nu_syn)-1)])
dnu_op = np.array([(nu_op[nu_ind+1]-nu_op[nu_ind-1])/2. for nu_ind in range(1,len(nu_op)-1)])

a_gg_f_syn = np.zeros(len(nu_syn) - 2)
a_gg_f_op = np.zeros(len(nu_op) - 2)
a_gg_f_temp = np.zeros(len(nu_tot) - 2)

if gg_flag == 1.:
    K_gg_op = f.build_a_gg_kernel(nu_op, nu_tot)
    K_gg_syn = f.build_a_gg_kernel(nu_syn, nu_tot)
else:
    K_gg_op = None
    K_gg_syn = None

Spec_list = []

# ---------------------------
# Constants / compactness
# ---------------------------
C_syn_el = sigmaT * c / (h * 24.0 * np.pi**2.0 * 0.8975) * (4.0 * np.pi * m_el * c / (3.0 * q)) ** (4.0 / 3.0)
C_syn_pr = sigmaT * c / (h * 24.0 * np.pi**2.0 * 0.8975) * (4.0 * np.pi * m_pr * c / (3.0 * q)) ** (4.0 / 3.0) * (m_el/m_pr)**2.

const_el = 4.0 / 3.0 * sigmaT / (8.0 * np.pi * m_el * c)
const_pr = const_el * (m_el / m_pr) ** 3.0

el_inj = f.inj_spectrum(V_R0, g_el, index_PL_min_el, index_PL_max_el, p_el1, L_el, m_el, PL_inj, g_el_br, p_el2)
pr_inj = f.inj_spectrum(V_R0, g_pr, index_PL_min_pr, index_PL_max_pr, p_pr1, L_pr, m_pr, PL_inj, g_pr_br, p_pr2)   

interv = 0.
N_el = el_inj.copy()*V_R0
N_pr = pr_inj.copy()*V_R0
dN_el_dVdg_el = np.zeros(len(g_el))
dN_pr_dVdg_pr = np.zeros(len(g_pr))
Spec_temp_tot = np.zeros(len(nu_tot))
g13_el = g_el ** (1./3.)
g2_el = g_el ** 2.
g13_pr = g_pr ** (1./3.)
g2_pr = g_pr ** 2.

gg_max_iter = 3
gg_tol_dex = 1e-3
gg_floor = 1e-300

# Solution of the PDEs
with open(out1,'w') as f1, open(out2,'w') as f2, open(out3,'w') as f3, open(out4,'w') as f4:
    for i in tqdm(range(int(time_end/step_alg)),desc="Progress..."):
        time_real += dt   
        Radius = f.R(R0, time_real, time_init, Vexp)
        V_t = f.Volume(Radius) 
        M_F = f.B(B0,R0,Radius,m)
        if V_t > V_R0:
            el_inj = f.inj_spectrum(V_t, g_el, index_PL_min_el, index_PL_max_el, p_el1, L_el, m_el, PL_inj, g_el_br, p_el2)
            pr_inj = f.inj_spectrum(V_t, g_pr, index_PL_min_pr, index_PL_max_pr, p_pr1, L_pr, m_pr, PL_inj, g_pr_br, p_pr2)  
        
        a_cr_el = 3.*q*M_F/(4.*np.pi*m_el*c)
        a_cr_pr = 3.*q*M_F/(4.*np.pi*m_pr*c)  
        

        if Ad_l_flag == 1.: 
            b_ad = Vexp/Radius
            dgdt_ad_el_m = b_ad*np.divide(np.power(g_el_mp[0:-1],1.),dg_el)
            dgdt_ad_el_p = b_ad*np.divide(np.power(g_el_mp[1:],1.),dg_el)
            dgdt_ad_pr_m = b_ad*np.divide(np.power(g_pr_mp[0:-1],1.),dg_pr)
            dgdt_ad_pr_p = b_ad*np.divide(np.power(g_pr_mp[1:],1.),dg_pr)
            dnudt_ad_syn_m = b_ad*np.divide(nu_syn_mp[0:-1],dnu)
            dnudt_ad_syn_p = b_ad*np.divide(nu_syn_mp[1:],dnu)
            dnudt_ad_op_m = b_ad*np.divide(nu_op_mp[0:-1],dnu_op)
            dnudt_ad_op_p = b_ad*np.divide(nu_op_mp[1:],dnu_op)
        else:
            dgdt_ad_el_m = dgdt_ad_el_p = np.zeros(len(g_el)-2)
            dgdt_ad_pr_m = dgdt_ad_pr_p = np.zeros(len(g_pr)-2)
            dnudt_ad_syn_m = dnudt_ad_syn_p = np.zeros(len(nu_syn)-2)
            dnudt_ad_op_m = dnudt_ad_op_p = np.zeros(len(nu_op)-2)
        
        if Syn_l_flag == 1.:
            b_syn_el = const_el*M_F**2.
            b_syn_pr = const_pr*M_F**2.
            dgdt_Syn_el_m = b_syn_el*np.divide(np.power(g_el_mp[0:-1],2)-1.,dg_el)
            dgdt_Syn_el_p = b_syn_el*np.divide(np.power(g_el_mp[1:],2)-1.,dg_el)
            dgdt_Syn_pr_m = b_syn_pr*np.divide(np.power(g_pr_mp[0:-1],2)-1.,dg_pr)
            dgdt_Syn_pr_p = b_syn_pr*np.divide(np.power(g_pr_mp[1:],2)-1.,dg_pr)
        else :
            dgdt_Syn_el_m = dgdt_Syn_el_p = np.zeros(len(g_el)-2)  
            dgdt_Syn_pr_m = dgdt_Syn_pr_p = np.zeros(len(g_pr)-2)  
    
        if IC_l_flag == 1.:
            U_ph = f.U_ph_KN(g_el,nu_tot,photons)
            b_Com_el = 4./3.*sigmaT*np.multiply(c,U_ph)/(m_el*c**2.)
            dgdt_op_el_m = b_Com_el[1:-1]*np.divide(np.power(g_el_mp[0:-1],2.)-1.,dg_el)
            dgdt_op_el_p = b_Com_el[2:]*np.divide(np.power(g_el_mp[1:],2.)-1.,dg_el)
        else:
            dgdt_op_el_m = dgdt_op_el_p = np.zeros(len(g_el)-2)    
            
        if pg_pi_l_flag == 1.:
            dgdt_pg_pi_m = np.array(f.dg_dt_pg(g_pr_mp[0:-1],nu_tot,photons))/dg_pr
            dgdt_pg_pi_p = np.array(f.dg_dt_pg(g_pr_mp[1:],nu_tot,photons))/dg_pr
        else:
            dgdt_pg_pi_m = dgdt_pg_pi_p = np.zeros(len(g_pr)-2)
            
        if pg_BH_l_flag == 1.:
            dgdt_pg_BH_m = np.array(f.dg_dt_BH(g_pr_mp[0:-1],nu_tot,photons,ln_k_i,ln_fk_i))/dg_pr
            dgdt_pg_BH_p = np.array(f.dg_dt_BH(g_pr_mp[1:],nu_tot,photons,ln_k_i,ln_fk_i))/dg_pr
        else:
            dgdt_pg_BH_m = dgdt_pg_BH_p = np.zeros(len(g_pr)-2)
           
        if pp_l_flag == 1.: 
            dgdt_pp_pi_m=np.divide(0.65*c*n_H*f.cs_pp_inel(g_pr_mp[0:-1]*m_pr*c**2.*erg_to_TeV)*(g_pr_mp[0:-1]-1.),dg_pr)
            dgdt_pp_pi_p=np.divide(0.65*c*n_H*f.cs_pp_inel(g_pr_mp[1:]*m_pr*c**2.*erg_to_TeV)*(g_pr_mp[1:]-1.),dg_pr)         
        else:
            dgdt_pp_pi_m = dgdt_pp_pi_p = np.zeros(len(g_pr)-2)

        V1 = np.zeros(len(g_pr)-2)
        V2 = 1.+dt*(c/Radius*esc_flag_pr+dgdt_ad_pr_m+dgdt_Syn_pr_m+dgdt_pg_pi_m+dgdt_pg_BH_m+dgdt_pp_pi_m)
        V3 = -dt*(dgdt_Syn_pr_p+dgdt_ad_pr_p+dgdt_pg_pi_p+dgdt_pg_BH_p+dgdt_pp_pi_p)
        S_ij = N_pr[1:-1]+np.multiply(pr_inj[1:-1],dt)*V_t
        N_pr[1:-1] = f.thomas_numba(V1, V2, V3, S_ij)
        dN_pr_dVdg_pr = np.array(N_pr/V_t)

        if pg_BH_emis_flag == 1.:
            Q_pg_BH = f.Q_BH_sol(g_el,g_pr,dN_pr_dVdg_pr,nu_tot,np.array(photons))[1:-1]
        else:
            Q_pg_BH = np.zeros(len(g_el)-2)
        
        if pg_pi_emis_flag == 1.:
            Q_pg_pi = np.nan_to_num(f.Qp_g_mod(g_el,nu_op,dN_pr_dVdg_pr,g_pr,photons,nu_tot,"e+")[1:-1],nan=0.0)+np.nan_to_num(f.Qp_g_mod(g_el,nu_op,dN_pr_dVdg_pr,g_pr,photons,nu_tot,"e-")[1:-1],nan=0.)
            Q_pg_g = np.nan_to_num(f.Qp_g_mod(g_el,nu_op,dN_pr_dVdg_pr,g_pr,photons,nu_tot,"2_g")[1:-1],nan=0.0) 
            if neutrino_flag == 1.:
                Q_pg_nu = np.multiply(f.Qp_g_mod(g_el,nu_op,N_pr,g_pr,photons,nu_tot,"nu_mu")+f.Qp_g_mod(g_el,nu_op,N_pr,g_pr,photons,nu_tot,"bar_nu_mu")+f.Qp_g_mod(g_el,nu_op,N_pr,g_pr,photons,nu_tot,"nu_e")+f.Qp_g_mod(g_el,nu_op,N_pr,g_pr,photons,nu_tot,"bar_nu_e"),dt)[1:-1]    
            else:
                Q_pg_nu = np.zeros(len(nu_nu)-2)
        else:
            Q_pg_pi = np.zeros(len(g_el)-2)
            Q_pg_g = np.zeros(len(nu_op)-2)
            Q_pg_nu = np.zeros(len(nu_nu)-2)

        if pp_ee_emis_flag == 1.:
            Q_pp_ee = np.multiply(f.Q_e_pp(g_el,g_pr,dN_pr_dVdg_pr/E_rp_TeV,p_pr1,n_H)[1:-1],m_el*c**2.*erg_to_TeV)
        else:
            Q_pp_ee = np.zeros(len(g_el)-2)
            
        if gg_flag == 0.:
            Q_ee = np.zeros(len(g_el)-2)
        else: 
            Q_ee = f.Q_ee_f(nu_tot,photons,nu_tot,photons,g_el,Radius)[1:-1]      

        V1 = np.zeros(len(g_el)-2)
        V2 = 1.+dt*(c/Radius*esc_flag_el+dgdt_Syn_el_m+dgdt_op_el_m+dgdt_ad_el_m)
        V3 = -dt*(dgdt_Syn_el_p+dgdt_op_el_p+dgdt_ad_el_p) 
        if inj_flag == 1.:
            S_ij = N_el[1:-1]+np.multiply(el_inj[1:-1]+Q_ee+Q_pg_pi+Q_pg_BH+Q_pp_ee,dt)*V_t
        if inj_flag == 0.:
            S_ij = N_el[1:-1]+(Q_ee+Q_pg_pi+Q_pg_BH+Q_pp_ee)*dt*V_t
        N_el[1:-1] = f.thomas_numba(V1, V2, V3, S_ij)    
        dN_el_dVdg_el = np.array(N_el/V_t)
        if Syn_emis_flag == 1.:
            Q_Syn_el = f.Q_syn_space(dN_el_dVdg_el, M_F, nu_syn, a_cr_el, C_syn_el, np.log(g_el), g13_el, g2_el)
            Q_Syn_pr = f.Q_syn_space(dN_pr_dVdg_pr, M_F, nu_syn, a_cr_pr, C_syn_pr, np.log(g_pr), g13_pr, g2_pr)
        else: 
            Q_Syn_el = Q_Syn_pr = np.zeros(len(nu_syn)-2)
        if IC_emis_flag == 1.:
            Q_IC = f.Q_IC(dN_el_dVdg_el, g_el, nu_op, photons, nu_tot)

        else:
            Q_IC = np.zeros(len(nu_op)-2)  
        if SSA_l_flag == 1.:
            aSSA_space_syn = -np.abs(f.aSSA(dN_el_dVdg_el, M_F, nu_syn, g_el, dg_l_el))
            aSSA_space_op = -np.abs(f.aSSA(dN_el_dVdg_el, M_F, nu_op, g_el, dg_l_el))

        else:
            aSSA_space_syn = np.zeros(len(nu_syn)-2) 
            aSSA_space_op = np.zeros(len(nu_op)-2)   
            
        if pp_g_emis_flag == 1.:
            Q_pp_g = np.multiply(f.Q_g_pp(nu_op,g_pr,dN_pr_dVdg_pr/E_rp_TeV,p_pr1,n_H)[1:-1],h*erg_to_TeV)
        else:
            Q_pp_g = np.zeros(len(nu_op)-2)

        if pp_nu_emis_flag == 1.:
            Q_pp_nu = 2.*np.multiply(f.Q_nu_e_pp(nu_nu,g_pr,dN_pr_dVdg_pr/E_rp_TeV,p_pr1,n_H)[1:-1],dt)*h*erg_to_TeV*V_t+2.*np.multiply(f.Q_nu_mu_pp(nu_nu,g_pr,dN_pr_dVdg_pr/E_rp_TeV,p_pr1,n_H)[1:-1],dt)*h*erg_to_TeV*V_t
        else:
            Q_pp_nu = np.zeros(len(nu_nu)-2)

        photons_syn_old = photons_syn.copy()
        photons_op_old = photons_op.copy()
        photons_guess = photons.copy()
        
        if gg_flag == 1.:
            n_gg_iter = gg_max_iter
        else:
            n_gg_iter = 1
        
        for gg_iter in range(n_gg_iter):        
            if gg_flag == 1.:
                a_gg_f_syn_iter = f.apply_a_gg_kernel(K_gg_syn, photons_guess)[1:-1]
                a_gg_f_op_iter = f.apply_a_gg_kernel(K_gg_op, photons_guess)[1:-1]
            else:
                a_gg_f_syn_iter = np.zeros(len(nu_syn) - 2)
                a_gg_f_op_iter = np.zeros(len(nu_op) - 2)
        
            V1 = np.zeros(len(nu_syn) - 2)
            V2 = 1. + dt * (c / Radius + dnudt_ad_syn_m - aSSA_space_syn * c + a_gg_f_syn_iter * c)
            V3 = -dt * dnudt_ad_syn_p
            S_ij = (photons_syn_old[1:-1] + 4. * np.pi * (Q_Syn_el + Q_Syn_pr) * dt * V_t)
            photons_syn_new = photons_syn_old.copy()
            photons_syn_new[1:-1] = f.thomas_numba(V1, V2, V3, S_ij)        

            V1 = np.zeros(len(nu_op) - 2)        
            V2 = 1. + dt * (c / Radius + dnudt_ad_op_m - aSSA_space_op * c + a_gg_f_op_iter * c)
            V3 = -dt * dnudt_ad_op_p
            S_ij = (photons_op_old[1:-1] + (Q_IC + Q_pg_g + Q_pp_g) * dt * V_t)
            photons_op_new = photons_op_old.copy()
            photons_op_new[1:-1] = f.thomas_numba(V1, V2, V3, S_ij)
        
            photons_new = (f.photons_tot(nu_syn,nu_bb,photons_syn_new+TINY,nu_op,photons_op_new+TINY,nu_tot,dN_dVdnu_BB * V_t,dN_dVdnu_pl * V_t,dN_dVdnu_user * V_t)/ V_t)
            photons_new = np.nan_to_num(photons_new, nan=0.0, posinf=0.0, neginf=0.0,)
        
            if gg_flag == 1.:
                err = np.nanmax(np.abs(np.log10(photons_new + gg_floor) - np.log10(photons_guess + gg_floor)))
                photons_guess = photons_new
                if err < gg_tol_dex:
                    break
            else:
                photons_guess = photons_new
        
        photons_syn = photons_syn_new
        photons_op = photons_op_new
        photons = photons_new
            
        V1 = np.zeros(len(nu_nu)-2)
        V2 = 1.+dt*(c/Radius*np.ones(len(nu_nu)-2))
        V3 = np.zeros(len(nu_nu)-2)
        S_ij = N_nu[1:-1]+Q_pp_nu+Q_pg_nu
        N_nu[1:-1] = f.thomas_numba(V1, V2, V3, S_ij)
        interv += 1
        
        if day_counter < time_real:            
            day_counter = day_counter+step_alg*R0/c

            Spec_temp_tot = np.multiply(photons,h*nu_tot**2.)*4.*np.pi/3.*Radius**2.*c  
            pr1 = [[str(el_list) for el_list in np.log10(g_el) ],[str(el_list) for el_list in np.log10(dN_el_dVdg_el) ]]
            pr2 = [[str(el_list) for el_list in np.log10(nu_tot) ],[str(el_list) for el_list in np.log10(Spec_temp_tot) ]]
            pr3 = [[str(el_list) for el_list in np.log10(g_pr) ],[str(el_list) for el_list in np.log10(dN_pr_dVdg_pr) ]]
            pr4 = [[str(el_list) for el_list in np.log10(nu_nu) ],[str(el_list) for el_list in np.log10(N_nu) ]]
            # here is where you unpack everything
            for row in zip(*pr1):
                f1.write(' '.join(row) + '\n')
            for row in zip(*pr2):
                f2.write(' '.join(row) + '\n')
            for row in zip(*pr3):
                f3.write(' '.join(row) + '\n')
            for row in zip(*pr4):
                f4.write(' '.join(row) + '\n')         
        
print("--- %s seconds ---" % "{:.2f}".format((time.time() - start_time)))
