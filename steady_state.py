import numpy as np
from scipy import optimize

# local
import blocks

def household_ss(Bq,par,ss):
    """ household behavior in steady state """

    ss.Bq = Bq

    # a. find consumption using final savings and Euler
    for i in range(par.A):

        a = par.A-1-i
        if i == 0:
            RHS = par.mu_B*(Bq/ss.P_C)**(-par.sigma)
        else:
            RHS = par.beta*(1+par.r_hh)*ss.C_a[a+1]**(-par.sigma)

        ss.C_a[a] = RHS**(-1/par.sigma)

    # b. find implied savings
    for a in range(par.A):

        if a == 0:
            B_lag = 0.0
        else: 
            B_lag = ss.B_a[a-1]
        
        ss.t_inc = ss.w*ss.L+par.U_B*ss.w*ss.U
        ss.t_inc_a[a] = ss.w*ss.L_a[a]+par.U_B*ss.w*ss.U_a[a]

        ss.B_a[a] = (1+par.r_hh)/(1+ss.pi_hh)*B_lag + par.yps*((1-ss.tau)*ss.t_inc_a[a]+ss.Bq/par.A) - ss.P_C*ss.C_a[a]        

    # c. aggreagtes
    ss.C = np.sum(ss.C_a) + ((1-par.yps)*((1-ss.tau)*ss.t_inc+ss.Bq/par.A))/ss.P_C
    ss.B = np.sum(ss.B_a)

    return ss.Bq-ss.B_a[-1]

def find_ss(par,ss,m_s,do_print=True):

    ss.m_s = m_s

    # a. price normalizations
    ss.P_E = 1.0*(par.eta_C/(par.eta_C-1))
    ss.P_Y = 1.0*(par.eta_C/(par.eta_C-1))
    ss.P_F = 1.0*(par.eta_C/(par.eta_C-1))
    ss.P_M_C = 1.0*(par.eta_C/(par.eta_C-1))
    ss.P_M_G = 1.0*(par.eta_C/(par.eta_C-1))
    ss.P_M_I = 1.0*(par.eta_C/(par.eta_C-1))
    ss.P_M_X = 1.0*(par.eta_C/(par.eta_C-1))
    
    # b. pricing in repacking firms
    ss.P_C_G = blocks.CES_P(ss.P_M_C,ss.P_Y,par.mu_M_C,par.sigma_C_G)
    ss.P_C = blocks.CES_P(ss.P_E,ss.P_C_G,par.mu_E_C,par.sigma_C)
    ss.P_G = blocks.CES_P(ss.P_M_G,ss.P_Y,par.mu_M_G,par.sigma_G)
    ss.P_I = blocks.CES_P(ss.P_M_I,ss.P_Y,par.mu_M_I,par.sigma_I)
    ss.P_X = blocks.CES_P(ss.P_M_X,ss.P_Y,par.mu_M_X,par.sigma_X)

    ss.pi_hh = 0.0

    # c. labor supply and search and matching
    for a in range(par.A):
        
        if a == 0:
            ss.S_a[a] = 1.0
            ss.L_ubar_a[a] = 0.0
        elif a >= par.A_R:
            ss.S_a[a] = 0.0
            ss.L_ubar_a[a] = 0.0            
        else:
            ss.S_a[a] = (1-ss.L_a[a-1]) + par.delta_L_a[a]*ss.L_a[a-1]
            ss.L_ubar_a[a] = (1-par.delta_L_a[a])*ss.L_a[a-1]

        ss.L_a[a] = ss.L_ubar_a[a] + ss.m_s*ss.S_a[a]
        ss.U_a[a] = 1 - ss.L_a[a]

    ss.S = np.sum(ss.S_a)
    ss.U = np.sum(ss.U_a)
    ss.L = np.sum(ss.L_a)
    ss.L_ubar = np.sum(ss.L_ubar_a)

    ss.delta_L = (ss.L-ss.L_ubar)/ss.L
    ss.curlyM = ss.delta_L*ss.L
    ss.v = (ss.m_s**(1/par.sigma_m)*ss.S**(1/par.sigma_m)/(1-ss.m_s**(1/par.sigma_m)))**par.sigma_m
    ss.m_v = ss.curlyM/ss.v

    if do_print:

        print(f'{ss.S = :.2f}')
        print(f'{ss.L = :.2f}')
        print(f'{ss.delta_L = :.2f}')
        print(f'{ss.v = :.2f}')
        print(f'{ss.m_v = :.2f}')

    # d. capital agency FOC
    ss.r_K = (par.r_firm + par.delta_K)*ss.P_I

    if do_print: print(f'{ss.r_K = :.2f}')

    # e. production firm pricing
    ss.r_ell = ((1-par.mu_K*(ss.r_K)**(1-par.sigma_Y_KL))/(1-par.mu_K))**(1/(1-par.sigma_Y_KL))
    ss.P_Y_KL = blocks.CES_P(ss.r_K,ss.r_ell,par.mu_K,par.sigma_Y_KL)
    ss.r_E = ((1-par.mu_E*(ss.P_Y_KL)**(1-par.sigma_Y))/(1-par.mu_E))**(1/(1-par.sigma_Y))

    if do_print: print(f'{ss.r_ell = :.2f}')
    if do_print: print(f'{ss.r_E = :.2f}')

    # f. labor agency
    ss.ell = ss.L - par.kappa_L*ss.v
    ss.w = ss.r_ell*(1-par.kappa_L/ss.m_v + (1-ss.delta_L)/(1+par.r_firm)*par.kappa_L/ss.m_v)

    if do_print: print(f'{ss.ell = :.2f}')
    if do_print: print(f'{ss.w = :.2f}')

    # h. production firm FOCs
    ss.K = par.mu_K/(1-par.mu_K)*(ss.r_ell/ss.r_K)**par.sigma_Y_KL*ss.ell

    if do_print: print(f'{ss.K = :.2f}')

    # i. capital accumulation equation
    ss.iota = ss.I = par.delta_K*ss.K

    if do_print: print(f'{ss.I = :.2f}')

    # j. output in production firm
    ss.Y_KL = blocks.CES_Y(ss.K,ss.ell,par.mu_K,par.sigma_Y_KL)
    ss.E = par.mu_E/(1-par.mu_E)*(ss.P_Y_KL/ss.r_E)**par.sigma_Y*ss.Y_KL
    ss.Y = blocks.CES_Y(ss.E,ss.Y_KL,par.mu_E,par.sigma_Y)

    if do_print: print(f'{ss.Y = :.2f}')

    # g. government
    ss.B_G = float(0) # debt in ss is 0, arbitrary number 
    ss.G = par.G_share_Y*ss.Y # this is an arbitrary number
    ss.tau = (par.r_b*ss.B_G+par.U_B*ss.w*ss.U+ss.P_G*ss.G)/(ss.w*ss.L+par.U_B*ss.w*ss.U) # based on expenses = income in period t, no change in debt in ss
    if do_print: print(f'{ss.G = :.2f}')
    if do_print: print(f'{ss.B_G = :.2f}')
    if do_print: print(f'{ss.tau = :.2f}')

    # g. household behavior
    if do_print: print(f'solving for household behavior:',end='')

    result = optimize.root_scalar(household_ss,bracket=[0.01,100],method='brentq',args=(par,ss,))
    if do_print: print(f' {result.converged = }')
    
    household_ss(result.root,par,ss)

    if do_print: print(f'{ss.C = :.2f}')
    if do_print: print(f'{ss.B = :.2f}')

    # k. CES demand in packing firms
    ss.C_E = blocks.CES_demand(par.mu_E_C,ss.P_E,ss.P_C,ss.C,par.sigma_C)
    ss.C_G = blocks.CES_demand((1-par.mu_E_C),ss.P_C_G,ss.P_C,ss.C,par.sigma_C)
    ss.C_M = blocks.CES_demand(par.mu_M_C,ss.P_M_C,ss.P_C_G,ss.C_G,par.sigma_C_G)
    ss.C_Y = blocks.CES_demand((1-par.mu_M_C),ss.P_Y,ss.P_C_G,ss.C_G,par.sigma_C_G)

    ss.G_M = blocks.CES_demand(par.mu_M_G,ss.P_M_G,ss.P_G,ss.G,par.sigma_G)   
    ss.G_Y = blocks.CES_demand((1-par.mu_M_G),ss.P_Y,ss.P_G,ss.G,par.sigma_G)

    ss.I_M = blocks.CES_demand(par.mu_M_I,ss.P_M_I,ss.P_I,ss.I,par.sigma_I)     
    ss.I_Y = blocks.CES_demand((1-par.mu_M_I),ss.P_Y,ss.P_I,ss.I,par.sigma_I)    

    # m. market clearing
    ss.X_Y = ss.Y - (ss.C_Y+ss.I_Y+ss.G_Y)
    ss.chi = ss.X_Y/(1-par.mu_M_X)
    ss.X = ss.X_Y/(1-par.mu_M_X)
    ss.X_M = blocks.CES_demand(par.mu_M_X,ss.P_M_X,ss.P_X,ss.X,par.sigma_X)
    
    ss.M = ss.C_M+ss.I_M+ss.X_M+ss.G_M+ss.C_E+ss.E

    if do_print: print(f'{ss.X = :.2f}')
    if do_print: print(f'{ss.M = :.2f}')

    # n. bargaining
    ss.w_ast = ss.w
    ss.w_U = par.U_B*ss.w
    ss.MPL = ((1-par.mu_K)*ss.Y_KL/ss.ell)**(1/par.sigma_Y_KL)
    par.phi = (ss.w_ast-ss.w_U)/(ss.r_ell-ss.w_U+(ss.v/ss.S)*par.kappa_L)
    if do_print: print(f'{par.phi = :.3f}')
