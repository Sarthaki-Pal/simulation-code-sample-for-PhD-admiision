import sys
import builtins as blt
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import healpy as hp
import time
import multiprocessing as mul
from functools import partial
import numexpr as ne

import params_sim as psim
import image_gen as ig
import vis_gen as vg

psim.params()           # simulation parameters
# ======== constants ========#
blt.C = 299.792 # velocity of light in Km/s


def params(tele):
	if (tele == 'MWA'):
		#MWA system params:
		blt.b = 4.0				#antenna dimensions in meters
		blt.nu_c = 154.25 		#central frequency in MHz
		#blt.Nchan = 154.24			# No. of channel
		blt.B_bw = 30.72			#in MHz units bandwidth
		#blt.dnu_c = blt.B_bw/blt.Nchan 		# channel width in MHz units
		#blt.lam_c = C/nu_c 		# Wavelength corresponding to central frequency in meter
		blt.NA = 126			#no of antennas
		blt.Nbl = NA*(NA-1)/2	#no of baselines
		blt.A_dBdT = 1			#(A*(dB/dT)^2)-1 (mk/Jy)^2 # 4 factor due to modified pbeam
		blt.dec_mwa = -26.79522222222223		#over head declination of MWA in degree
#		blt.Tsys = 150 			# system temperature in K
#		blt.eta = 0.6		 	# efficiency of the telescope	
		
        # Cosmology of MWA	
#		blt.r = 6845.5			#comoving distance in Mpc
#		blt.rp = 11.5			#dr/dnu in Mpc/MHz
#		blt.dBdT = 3.27			#(dB/dT)_326.5 in Jy/mK 
	
		blt.KB = 1.38		 	# Boltzman constant in Jy.m^2/mK
		
		# # Healpy parameters for simulation
		# blt.NSIDE = 64		#Nside for Healpix maps which sets the resolution
	
		# # Input < C_{\ell}= Amp * \ell^{\beta} >parameters for simulation
		# blt.Amp = 1.0		#amplitude of the input angular power spectrum in mK^2
		# blt.beta = -2.0 
		
		# #System parameter
		# blt.Nthreads = None #number of parallel threads
	else :
		print ("Please enter telescope name")
	return (0)


def beam_mwa(nu, dne1, dne2):
    """Beam function for MWA

    Args:
        nu : Central frequency in MHz
        dne1 :  angular separation from phase centre along e1 axis
        dne2 :  angular separation from phase centre along e2 axis
    
    Returns:
        Returns a double precision number.
    """
    Ae1 = np.sinc((b*nu/C)*dne1)**2.0
    Ae2 = np.sinc((b*nu/C)*dne2)**2.0
    return(Ae1*Ae2)


def visgen_mwa(nside , in_map , ra_ptg, dec_ptg, bl_file, nu, ang_range = 90.0): # NOT OPTIMAL FOR MULTIPLE REALIZATION
    Npix = hp.nside2npix(nside)     #Number of pixels given N_side
    res = hp.nside2resol(nside, arcmin = True)      #resolution given N_side
    lmax = nside * 3 - 1        #maximum number of \ell multipoles used
    print("Nside=",nside, 'resolution=', res, 'min', Npix, 'pixels' )

    pb = ig.pbeam_hpmap_gen(nside, ra_ptg, dec_ptg, nu, beam_mwa) #compute beam
    # sky_nd_pb = in_map * pb       #sky through beam for a single realization

    Nrea = in_map.shape[0]
    # vis = np.empty(Nrea, dtype='object')
    vis = np.zeros((Nrea, len(bl_file)), dtype=np.complex64)
    
    for ii in range(Nrea):
        sky_nd_pb = in_map[ii] * pb
        vis[ii] = vg.vis_comp_fast(nside, sky_nd_pb, ra_ptg, dec_ptg, bl_file, nu, ang_range = 90.0)

    # vis = []
    # vis = vg.vis_comp_fast(nside, sky_nd_pb, ra_ptg, dec_ptg, bl_file, nu, ang_range = 90.0)     #visibility computation
    vis_all = np.array(vis) 

    return(vis_all)


def dn_comp(ipix, a0, d0):
    """This function calculates the difference vector between the phasecentre and a pixel in the sky map and returns the projection of this difference vector 
    onto the local tangent plane at the phase centre. In this function, e1, e2 & e3 represents a basis in a local tangent plane atthe phase center. e1 is a 
    vector pointing in the RA direction, e2 is a vector in the declination direction and e3 is the vector pointing toward the phase center itself. n is the 3D 
    vector corresponding to the direction of the pixel in the sky, calculated based on the RA and Dec of the pixel. The vector p (= e3) is the reference direction, 
    which is taken to be same as that pointing to the phasecenter. dn is the difference between the pixel's direction (n) and the phasecenter direction (p).

    Args:
        ipix : This is the array of pixels in the sky
        a0 : angle of the phase centre w.r.t the zenith, which is taken as 0
        d0 : Declination of the MWA
    
    Returns:
        Returns three numpy arrays containing the three projections of the difference vector along the three unit vectors.
    """
    RA, dp = hp.pix2ang(NSIDE, ipix, lonlat=True)

    ap = np.deg2rad(RA)  # pixel RA
    dp = np.deg2rad(dp)  # pixel dec
    a0 = np.deg2rad(a0)  # RA of the phase center
    d0 = np.deg2rad(d0)  # Phase center dec of the observation

    e1 = np.array([-np.sin(a0), np.cos(a0), 0])
    e2 = np.array([-np.cos(a0)*np.sin(d0), -np.sin(d0)*np.sin(a0), np.cos(d0)])
    e3 = np.array([np.cos(a0)*np.cos(d0), np.sin(a0)*np.cos(d0), np.sin(d0)])

    n = np.array([np.cos(ap)*np.cos(dp), np.sin(ap)*np.cos(dp), np.sin(dp)])
    p = e3
    dn = n - p

    dne1 = np.inner(dn, e1)
    dne2 = np.inner(dn, e2)
    dne3 = np.inner(dn, e3)

    return(dne1, dne2, dne3)


def cal_phase(dn, bl, nu):  
    """This function calculates the phase factor φ=e2πiˆd· ̃U
    
    Args:
        dn : This is the vector ̃d
        bl : The baseline array
        nu :  Central frequency in MHz
    """
    ble1, ble2, ble3 = bl 
    ppi = np.pi #3.141592653589793
    dn0, dn1, dn2 = dn[:, 0].astype(np.float32), dn[:, 1].astype(
        np.float32), dn[:, 2].astype(np.float32)
    phase = ne.evaluate('exp(1j*2.0*ppi*(dn0*ble1 + dn1*ble2 + dn2*ble3))')
    # phase = np.exp(1j*2.0*ppi*(dn0*ble1 + dn1*ble2 + dn2*ble3))
    phaseT = phase.astype(np.complex64)
    return(phaseT)


def visgen_mwa_multi(nside, in_map, ra_ptg, dec_ptg, bl_file, nu, ang_range = 90.0): # OPTIMAL FOR MULTIPLE REALIZATION
    """This function generates the visibilities for multiple realizations.

    Args:
        nside : Healpix parameter
        in_map : The input Healpix map (map_grf) for all realizations
        ra_ptg : Pointing RA
        dec_ptg : Pointing DEC
        bl_file : Baseline numpy array (bln)
        nu : Central frequency in MHz
        ang_range : 90.0 degrees. Defaults to 90.0.
    
    Returns:
        Returns a complex array of visibilities.
    """
    start = time.time()
    
    Npix = hp.nside2npix(nside)     # Number of pixels given N_side
    res = hp.nside2resol(nside, arcmin = True)      # resolution given N_side
    lmax = nside * 3 - 1        #maximum number of \ell multipoles used

    pb = ig.pbeam_hpmap_gen(nside, ra_ptg, dec_ptg, nu, beam_mwa) #compute beam

    # a list of pixel numbers for multiprocessing
    pix_num = list(range(0, Npix))
    dOmega = hp.nside2resol(nside)**2.0  # resolution in steradians

    MWA_vec = hp.ang2vec(0, dec_ptg, lonlat=True)
    ipix_mask = hp.query_disc(nside=nside, vec=MWA_vec, radius=np.radians(90))

    # ------- make array of dn to be used for vis_cal ----#
    dn_red = partial(dn_comp, a0=0, d0=dec_ptg)

    # Storing \Delta n values for iterative use
    pool = mul.Pool()
    dn = np.array(pool.map(dn_red, pix_num))
    pool.close()
    pool.join()
    
    #--------- done make dn array --------------------#
    
    lam = C/nu                  # wavelength in m
    Q_nu = 2.0 * KB / lam**2.0  # in Jy/mK
    bln = bl_file / lam         # baselines in units of wavelength    
    mm, nn = bln.shape          # number of baselines
    
    # visibility calculation 
    vis = np.zeros((Nrea, mm), dtype=np.complex64)    
    sky_nd_pb_NR = np.zeros((len(ipix_mask), Nrea)) # NpxNr matrix
    for jj in range(Nrea):
        sky_nd_pb_NR[:, jj] = in_map[jj, ipix_mask] * pb[ipix_mask]
    
    for ii in range(mm): # baseline loop
        phaseT = cal_phase(dn[ipix_mask], bln[ii,:], nu) # calculate phase
        vis[:, ii] = phaseT @ sky_nd_pb_NR # (1xNp)@(NpxNr)     
    
    vis =  Q_nu * dOmega * vis  
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken in all baseline loops -", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    
    return(np.array(vis))
