import builtins as blt
# ======== constants ========#
blt.C = 299.792  # velocity of light in Km/s


def params():
    blt.KB = 1.38		 	# Boltzman constant in Jy.m^2/mK
    
    blt.NSIDE = 16  # Nside for Healpix maps which sets the resolution

    # Input < C_{\ell}= Amp * \ell^{\beta} >parameters for simulation
    blt.Amp = 100  # amplitude of the input angular power spectrum in mK^2
    blt.beta = -2

    # observation input parameters
    blt.ra_ini = 0  # initial pointing RA of MWA during observation in degree
    # initial phase center of observation w.r.t. dec of MWA (-26.7 deg)
    blt.dec_ini = 0.0
    blt.obs_time = 0.0  # total observation time in mins
    blt.ptg_time = 0.0  # pointing time in secs

    # System parameter
    blt.Nthreads = 40  # number of parallel threads
    blt.Nthreads1 = 40  # number of parallel threads #for baseline loop
    
    # No of realizations
    blt.Nrea = 10 # number of realizations of the simulations
