########################################################
# Author of this version: Luca Morf - luca.morf@uzh.ch #
########################################################

import numpy as np
import functools

from PyToF.color import c

class ToF:

    """
    This class contains standard routines when using the Theory of Figures.
    """

    def _default_opts(self, kwargs):

        """
        This function implements the standard options used for the ToF class,
        except for the kwargs given by the user.
        """

        opts      =    {'N':                2**10,  #Number of gridpoints, i.e. level surfaces used
                        'n_bin':            -1,     #Numbers of bins for calculation speed-up, -1 or n_bin==N means that the figure functions will be calculated for all the points supplied.
                        'order':            4,      #Order of the Theory of Figures to be used
                        'dJ_tol':           1e-10,  #Relative tolerance in the iterative procedure for the Js
                        'drot_tol':         1e-10,  #Relative tolerance in the iterative procedure for the rotational parameter
                        'drho_tol':         1e-10,  #Relative tolerance in the iterative procedure for the densities
                        'MaxIterShape':     100,    #Maximum amount of iterations in AlgoToF when calling relax_to_shape()
                        'MaxIterUpdate':    100,    #Maximum amount of iterations when using _update_densities_barotrope() or _apply_atmosphere()
                        'MaxIterBar':       100,    #Maximum amount of times relax_to_barotrope() calls relax_to_shape()
                        'MaxIterDen':       100,    #Maximum amount of times relax_to_density() calls relax_to_shape()
                        'verbosity':        0,      #Higher numbers lead to more verbosity output in the console
                        
                        #Numbers are taken from Wisdom & Hubbard 2016, Differential rotation in Jupiter: A comparison of methods: 
                        'G':                6.6738480e-11,                                                      #Newtons gravitational constant in SI units
                        'M_phys':           126686536.1*(1000)**3/6.6738480e-11,                                #Mass of the planet in SI units
                        'R_ref':            None,                                                               #Reference radius of the planet in SI units
                        'R_phys':           [71492000, 'equatorial'],                                           #Physical radius of the planet in SI units, including type specification
                        'Period':           2*np.pi/np.sqrt((126686536.1*(1000)**3*0.089195487)/(71492000**3)), #Initial rotation period of the planet in SI units
                        'P0':               0,                                                                  #Initial surface pressure of the planet in SI units
                        'Target_Js':        np.array([   1.398851089834702e-2,                                  #J2
                                                        -5.318281001092907e-4,                                  #J4
                                                         3.011832290533641e-5,                                  #J6
                                                        -2.132115710725050e-6,                                  #J8
                                                         1.740671195871128e-7,                                  #J10
                                                        -1.568219505602588e-8,                                  #J12
                                                         1.518099230068580e-9,                                  #J14
                                                        -1.551985393081485e-10,                                 #J16
                                                         1.655948019619652e-11,                                 #J18
                                                        -1.829544870258362e-12]),                               #J20
                        'Sigma_Js':         np.array([  np.spacing(1),                                          #J2
                                                        np.spacing(1),                                          #J4
                                                        np.spacing(1),                                          #J6
                                                        np.spacing(1),                                          #J8
                                                        np.spacing(1),                                          #J10
                                                        np.spacing(1),                                          #J12
                                                        np.spacing(1),                                          #J14
                                                        np.spacing(1),                                          #J16
                                                        np.spacing(1),                                          #J18
                                                        np.spacing(1)]),                                        #J20
                        'alphas':           np.zeros(12),                                                       #Barotropic differential rotation parameters

                        'rho_MAX':          2e4,    #Maximal density that is physically acceptable, in SI units
                        'P_MAX':            1e13,   #Maximal pressure that is physically acceptable, in SI units

                        'baro_param_init':  None,   #Parameters used by the barotrope function set via set_barotrope()
                        'dens_param_init':  None,   #Parameters used by the density function set via set_density_function()
                        
                        'use_atmosphere':   False,  #See _update_densities(), will use an atmospheric model for the outermost densities if True
                        'atmosphere':       None,   #A function of the mean level surface and the pressure that returns the density compatible with an atmospheric model
                        'atmosphere_until': None,   #Pressure value in SI units that marks the earliest acceptable end of the atmospheric model
                        
                        'H':                0.,     #work in progress
                        }

        #Update the standard numerical parameters with the user input provided via the kwargs
        for kw, v in kwargs.items():

            if kw in opts:

                opts[kw] = v

            else:

                print(str(kw) + c.WARN + ' is an invalid keyword!' + c.ENDC)

        return opts

    def _set_IC(self):

        """
        Initializes...

        - li:               Array, level surfaces l, Eq. (B.1) in arXiv:1708.06177v1
        - rhoi:             Array, densities at the level surfaces li 
        - Pi:               Array, pressures at the level surfaces li
        - Js:               Array, gravitational harmonics, Eq. (B.11) in arXiv:1708.06177v1
        - R_calc            Float, calculated equatorial radius in SI units
        - m_rot_calc:       Float, dimensionless rotational parameter
        - ss:               List of arrays, figure functions introduced in Eq. (B.1) in arXiv:1708.06177v1
        - SS:               List of arrays, dimensionless volume integrals introduced in Eq. (B.7) in arXiv:1708.06177v1
        - A0:               Array, important to determine the total potential introduced in Eq. (B.3) in arXiv:1708.06177v1
        - R_eq_to_R_m:      Float, ratio of the equatorial radius to the to outermost mean surface layer 
        - R_po_to_R_m:      Float, ratio of the polar radius to the to outermost mean surface layer 
        - baro_param_calc:  Possibly updated parameters used by the barotrope function set via set_barotrope()
        - dens_param_calc:  Possibly updated parameters used by the density function set via set_density_function()
        """

        self.li                 = np.linspace(1, 1/self.opts['N'], self.opts['N'])*self.opts['R_phys'][0]
        self.rhoi               = np.ones(self.opts['N'])*self.opts['M_phys']/(4*np.pi/3*self.opts['R_phys'][0]**3)
        self.Pi                 = np.zeros(self.opts['N'])
        self.Js                 = np.hstack((-1,        np.zeros(self.opts['order'])))
        self.Js_error           = np.hstack((-1, np.nan*np.zeros(self.opts['order'])))
        self.R_calc             = self.opts['R_phys'][0]
        self.m_rot_calc         = (2*np.pi/self.opts['Period'])**2*self.li[0]**3/(self.opts['G']*self.opts['M_phys'])
        self.ss                 = (self.opts['order']+1)*[np.zeros(self.opts['N'])]
        self.SS                 = (self.opts['order']+1)*[np.zeros(self.opts['N'])]
        self.A0                 = np.zeros(self.opts['N'])
        self.R_eq_to_R_m        = 1.
        self.R_po_to_R_m        = 1.
        self.baro_param_calc    = self.opts['baro_param_init']
        self.dens_param_calc    = self.opts['dens_param_init']
        self.atmosphere_index   = 0
                
    def __init__(self, **kwargs):

        """
        Initializes the ToF class. Options can be provided via **kwargs, otherwise the default options from _default_opts() will be implemented.
        """

        #Set initial values:
        self.opts  = self._default_opts(kwargs)
        assert len(self.opts['R_phys']) == 2, c.WARN + 'R_phys must be a list-like object with length 2! First entry: float: radius in SI units, Second entry: string: \'equatorial\', \'mean\', \'polar\'' + c.ENDC
        assert self.opts['R_phys'][1] in ['equatorial', 'mean', 'polar'], c.WARN + 'The second entry of R_phys must be a string, options: \'equatorial\', \'mean\', \'polar\'' + c.ENDC

        if self.opts['R_ref'] is None:
            
            print(c.WARN + 'No reference radius supplied by the user. PyToF assumes R_ref = R_phys.' + c.ENDC)
            self.opts['R_ref'] = self.opts['R_phys'][0]

        self._set_IC()

        #Define routines for the user: 
        from PyToF.FunctionsToF import get_r_l_mu, set_barotrope, set_density_function, relax_to_shape, relax_to_barotrope, relax_to_density, get_U_l_mu, get_NMoI

        self.get_r_l_mu             = functools.partial(get_r_l_mu,             self)
        self.set_barotrope          = functools.partial(set_barotrope,          self)
        self.set_density_function   = functools.partial(set_density_function,   self)
        self.relax_to_shape         = functools.partial(relax_to_shape,         self)
        self.relax_to_barotrope     = functools.partial(relax_to_barotrope,     self)
        self.relax_to_density       = functools.partial(relax_to_density,       self)
        self.get_U_l_mu             = functools.partial(get_U_l_mu,             self)
        self.get_NMoI               = functools.partial(get_NMoI,               self)

        from PyToF.MonteCarloToF import set_check_param, baro_cost_function, dens_cost_function, run_baro_MC, run_dens_MC, classify_and_save_state

        self.set_check_param            = functools.partial(set_check_param,            self)
        self.baro_cost_function         = functools.partial(baro_cost_function,         self)
        self.dens_cost_function         = functools.partial(dens_cost_function,         self)
        self.run_baro_MC                = functools.partial(run_baro_MC,                self)
        self.run_dens_MC                = functools.partial(run_dens_MC,                self)
        self.classify_and_save_state    = functools.partial(classify_and_save_state,    self)

        from PyToF.PlotToF import plot_xy, plot_shape, plot_ss, plot_state_xy, plot_state_corr_xy, plot_autocorr

        self.plot_xy            = functools.partial(plot_xy,            self)
        self.plot_shape         = functools.partial(plot_shape,         self)
        self.plot_ss            = functools.partial(plot_ss,            self)
        self.plot_state_xy      = functools.partial(plot_state_xy,      self)
        self.plot_state_corr_xy = functools.partial(plot_state_corr_xy, self)
        
        self.plot_autocorr   = plot_autocorr