"""
This script contains most of the functions that are called by pysme_exec.py.
It fetches the spectra and parameters from the WEAVE FITS files, and also 
calls functions to get gaia parameters and determine stellar parameters.
"""

import numpy as np
import os

from astropy.io import fits
from scipy.interpolate import interp1d
from astroquery.gaia import Gaia

#import pysme_update_logg
import pysme_gaia_dist_extinc

# 150405000901378 is the sun."""

def setup_object(obj_ind):
    """ Basic set up for the object to be able to find the object in the large
    data releases.

    Returns
    -------
    obs_name: string
        Combination of object number, method of calculation, and the data
        set being used.
    obs_file: string
        Obs_name but with the addition of the file type. We are currently
        using .dat
    obj_for_obs: string
        The WEAVE identifier in coloumn CNAME
    field_end: string
        Displays the method of calculation. Must be one of ['Lbol' | 'Seis'].
        Takes the final 4 characters of field_for_obs
    setup_for_obs: string
        The data set being used.  field_for_obs: string
        Shows the method of calculation, but also other parameters such as
        whether it's a test, benchmark, is
        the sun, etc.
    iterations: list
        How many iterations to run Pysme for (iterations[x] for x
        pysme_execute run) and how many times to run pysme_execute
        (len(iterations))
    """

    # Path to the data file containing the WEAVE spectrum
    path = \
        '/proj/snic2020-16-23/shared/KY_pysme_WEAVE/DATA'
#    data_path = './data'

    # The blue and red wavelength regions
    data_file_B = fits.open(path + 
                    '/stacked_1002082__stacked_1002081_exp_b_APS.fits')

    # The green wavelength region
    data_file_G = fits.open(path + 
                    '/stacked_1002118__stacked_1002117_exp_b_APS.fits')

    # get catalogue name corresponding to the current object (passed in as a 
    # sys.arg from the script call)
#    obj_for_obs = 'WVE_19203729+3840075' # input("Object, please.")
    obj_for_obs = data_file_B[2].data['CNAME'][int(obj_ind)] # input("Object, please.")

    field_for_obs = "fixed"  # input("lbol or seis?")

    field_end = "fixed"

    # Which segment of HR data to use, must be one of 
    # ('R' | 'B' | 'G' | '')
    wav_segment = "stitched"

    # Displays the data set being used. Used to find the files in the Galah
    # folder.
#    setup_for_obs = "WEAVE_HR_R"  # input("Setup, please.")
    setup_for_obs = "WEAVE_HR_%s" %wav_segment  # input("Setup, please.")

    # Name of the file that will be saved.
    obs_name = str(field_for_obs) + "_" + str(obj_for_obs) + "_" + \
                   str(setup_for_obs) + "_Sp"


    # The object data file to find. Not in obs_name as we don't want to save as
    # .dat
    obs_file = str(obs_name + ".dat")

    # Four SME calls : normalise, iterate, normalise, iterate
#    iterations = [1, 2, 1, 20]
    iterations = [1, 10]

    return obs_name, obs_file, obj_for_obs, field_end, setup_for_obs, \
              field_for_obs, wav_segment, iterations

def set_atmosphere():
    """
    Setting up of the atmosphere file names.

    Returns
    -------
    atmosphere_grid_file: string
        The name of the atmosphere file to use in pysme.
    line_list: string
        The name of the file that contains all atomic data we use in our
        regular non-balmer runs.
    atomic_abundances_free_parameters:
        Unused at the moment but may be required in future.
    atmosphere_abundance_grid: array_like
        A list containing the NLTE element grid file names.
    """
    """Choosing atmoshpere, grids, and NLTE."""
    # Atmosphere grid file to use.
    atmosphere_grid_file = 'marcs2014.sav'
    # Linelist containing atomic data
#    line_list = 'master_hfs.lin'
    line_list = 'galah_master_v5.2.fits'
    # Unused
    atomic_abundances_free_parameters = np.zeros(99)
    # NLTE file names. [0] must be H and [1] Fe, or change it in pysme_run_sme
    # atmosphere_abundance_grid = \
    #        ['marcs2012_H2018.grd', 'marcs2012_Fe2016.grd']
    atmosphere_abundance_grid = ['', '']

    return atmosphere_grid_file, line_list, \
               atomic_abundances_free_parameters, atmosphere_abundance_grid

def sme_variables():
    """
    Other variable we use in our files.

    Returns
    -------
    broad_lines:
        The balmer lines and other important lines to note. Primarily
        used in balmer_runs for Halpha and beta depthmin: minimum depth
        required to include atomic lines. Too low will include too many and
        take too long to
    run line_cores:
        Important line number to note.
    """
    depthmin = 0.1
    print('Using minimum depth ', depthmin)
    # Balmer lines are the first two. Adding here gives them into
    # alllinesindex which we use to identify desired  linelist lambdas.
    # Hbeta, Mg Ib triplet, Halpha 
    broad_lines = [4861.3230, 5167.3216, 5172.6843, 5183.6042, 6562.7970]
    line_cores = [8498.0200]
    # Imaginary number as 0 and the like are reasonable results for some
    # things.

    return broad_lines, depthmin, line_cores


"""Part 2: read in spectra and extract relevant information"""


def get_wav_flux(obj_for_obs, wav_segment):
    """
    Get wavelength, flux and error for the WEAVE spectrum.

    Parameters
    -------
        obj_for_obs: CNAME of the object of interest

    Returns
    -------
        spec_wav: Wavelength of the spectrum
        spec_flux: Flux values of the spectrum
        spec_flux_err: Flux error (inverse variance) for the spectrum
        data_ind: the index corresponding to the object id
        data_file: fits_hdu
            The data file containing all the information about the spectrum
    """
#    path = '/home/kryo6156/work/DATA/WEAVE/golden_sample/CCG_NGC6791_F1W1/HRB'
    path = '/proj/snic2020-16-23/shared/KY_pysme_WEAVE/DATA'

#    spectra_file = fits.open(path + 'stacked_1003798_1003797.aps.fits')
    # This is the stacked L2 products, but it lacks some header information
    # about the instrumental set up.
    # The blue and red wavelength regions
    data_file_B = fits.open(path + 
                    '/stacked_1002082__stacked_1002081_exp_b_APS.fits')

    # The green wavelength region
    data_file_G = fits.open(path + 
                    '/stacked_1002118__stacked_1002117_exp_b_APS.fits')
    # This is the L1 product which contains extra info about the observation
    # set up
    head_file = fits.open(path + 
                    '/stack_1002081.fit')

    # get index corresponding to the object id for L2 files
    data_ind_B = np.where(data_file_B[4].data['CNAME'] == obj_for_obs)[0][0]
    data_ind_G = np.where(data_file_G[4].data['CNAME'] == obj_for_obs)[0][0]

    # get index corresponding to the object id for L1 files
    head_ind = np.where(head_file[6].data['CNAME'] == obj_for_obs)[0][0]

    print('object (Blue)--> %s' %data_file_B[4].data['CNAME'][data_ind_B])
    print('object (Green)--> %s' %data_file_G[4].data['CNAME'][data_ind_G])
    print('gaia id --> %s' %data_file_B[4].data['TARGID'][data_ind_B])

    """#####Add in a catch for wrong file names or non single objects#####"""
    # The fits file contains 9 header units: 1 - CLASS_TABLE, 2 -STAR_TABLE 
    # 3 - GALAXY_TABLE, 4 - CLASS_SPEC, 5 - STAR_SPEC, 6 - GALAXY_SPEC. 
    # The unormalized spectra are in number 4.

    # Stitch together the wavelenth segments
    if wav_segment == "stitched":

        print(np.concatenate((data_file_B[4].data['LAMBDA_RR_B'][data_ind_B,:],
                         data_file_G[4].data['LAMBDA_RR_G'][data_ind_G,:],
                         data_file_B[4].data['LAMBDA_RR_R'][data_ind_B,:]), 
                            axis=None))

        spec_wav = np.concatenate((data_file_B[4].data['LAMBDA_RR_B'][data_ind_B,:],
                             data_file_G[4].data['LAMBDA_RR_G'][data_ind_G,:],
                             data_file_B[4].data['LAMBDA_RR_R'][data_ind_B,:]),
                                axis=None)
  
        spec_flux = np.concatenate((data_file_B[4].data['FLUX_RR_B'][data_ind_B,:],
                             data_file_G[4].data['FLUX_RR_G'][data_ind_G,:],
                             data_file_B[4].data['FLUX_RR_R'][data_ind_B,:]),
                                axis=None)

        spec_flux_err = np.concatenate((data_file_B[4].data['IVAR_RR_B'][data_ind_B,:],
                             data_file_G[4].data['IVAR_RR_G'][data_ind_G,:],
                             data_file_B[4].data['IVAR_RR_R'][data_ind_B,:]),
                                axis=None)

    elif wav_segment == "R":

        spec_wav = data_file_B[4].data['FLUX_RR_R'][data_ind_B,:]
        spec_flux = data_file_B[4].data['FLUX_RR_R'][data_ind_B,:]
        spec_flux_err =  data_file_B[4].data['IVAR_RR_R'][data_ind_B,:]

    return spec_wav, spec_flux, spec_flux_err, data_file_B, data_ind_B, \
            head_file, head_ind

def get_resolution(obj_for_obs, head_file, spec_wav, head_ind):
    """
    Interpolation to produce resolutions for different wavelengths we produce
    during the run (e.g from doppler shift)

    Parameters
    -------
        obj_for_obs: string
           CNAME for object
        head_file: fits_hdu
            The data file containing extra information about the spectrum
        spec_wave: array_like
            wavelength values of the spectrum
        head_ind: the index corresponding to the object id
    Returns
    -------
        interpolation: Interpolation equation to find resolution at other
            wavelengths
        res_type: Whether we have a high or low
            resolution. It's applied to the later resolutions."""

    # Find whether the spectrum is high or low resolution
    res_type = head_file[0].header['MODE']

    # resolution of the observed fibre
    res_fibre = head_file[6].data['RESOL']

    """I think this is wavelength and resolution power at given wavelength
    this may not be correct though so maybe come back to check this later"""
#    resolution = spec_wav[:10] * res_fibre[head_ind]
    resolution = spec_wav * res_fibre[head_ind]

    interpolation = interp1d(spec_wav, spec_wav * res_fibre[head_ind])

    return interpolation, res_type, head_file

"""Part 3: Getting preliminary guesses for stellar parameters: Teff, logg,
[M/H], micro_turb, macro_turb, rot_velocity, rad_velocity"""


def get_starting_params(data_file, data_ind):
    """
    Fetches the stellar parameters computed by WEAVE APS pipeline for our star
    which we use as a starting estimate, and then iterates to make them more
    accurate. Inside those functions, they call on the data provided to them
    and modify it to account for physics not included in raw data. For example,
    modifies gravity for cool dwarf stars.

    Parameters
    -------
        data_file: fits_hdu
            The data file containing all the information about the spectrum
        data_ind: the index corresponding to the object id
    Returns
    -------
        starting_params: dictionary
            A dictionary containing several parameters:
                Teff
                gravity
                metallicity
                vrad_global
                rotational_vel
                micro_turb_vel
                macro_turb_vel
    """

    # We use a dictionary to keep starting parameters in.
    starting_params = {}

    # extension 8 corresponds to the stellar table RVS w/ all the params
    starting_params['Teff'] = data_file[2].data['TEFF_RVS'][data_ind]
    starting_params['gravity'] = data_file[2].data['LOGG_RVS'][data_ind]
    starting_params['metallicity'] = data_file[2].data['FEH_RVS'][data_ind]
    starting_params['vrad_global'] = data_file[2].data['VRAD'][data_ind]
    starting_params['micro_turb_vel'] = data_file[2].data['MICRO'][data_ind]
    starting_params['rotational_vel'] = data_file[2].data['VSINI_RVS'][data_ind]
    starting_params['macro_turb_vel'] = 0

    return starting_params


def microturbulence_macroturbulence(temperature, gravity):
    """
    Micro and macro turbulence affect the flux we see, changing our guesses on
    gravity and T etc...but macro turbulence effects are so similar to radial
    velocity effects that our resolution cannot distinguish them hence we
    scrap the macro and just use vsini (radial velocity)

    Parameters
    -------
        data_file: fits_hdu
            The data file containing all the information about the spectrum
        data_ind: the index corresponding to the object id
    Returns
    -------
        starting_params: A dictionary containing several parameters:
            Teff
            gravity
            metallicity
            vrad_global
            rotational_vel
            micro_turb_vel
            macro_turb_vel
    """
    # Prevents warning on reference being assigning.
    ai1 = 0
    ai2 = 0
    ai3 = 0
    aa1 = 0
    ba1 = 0
    ca1 = 0
    if gravity <= 4.2 or temperature >= 5500:
        ai1 = 1.1
        ai2 = 1E-4
        ai3 = 4E-7
        aa1 = 1.5
        ba1 = 0
        ca1 = 1E-6
        # ai1 = 1 ai2 = -2.5E-4 ai3 = 6.5E-7
    elif gravity >= 4.2 and temperature < 5500:
        ai1 = 1.1
        ai2 = 1.6E-4
        ai3 = 0
        aa1 = 1.5
        ba1 = 0.2E-3
        ca1 = 0
    # It should not be able to miss the two other if statements.
    if ai1 == 0:
        print("Something wrong with microturbulence code and its if", \
                "requirements. Check it out --> pysme_WEAVE line 410.")
        exit()
    temperature_base = 5500
    gravity_base = 4.0
    # vmic =      ai1 + ai2*(t -t0) + ai3 * (t-t0)^2 + bi1 + bi2 * (g-g0) + bi3 * (g-g0)^2 + ci1 + ci2 * z + ci3 * z^2
    # vmac = 3 * (aa1 + ba1*(t -t0) + ca1 * (t-t0)^2 + ba2*(g-g0) + ca2*(g-g0)^2 + ba3*z + ca3*z^2) + 2.0
    micro_turb_vel = ai1 + ai2 * (temperature - temperature_base) + \
                               ai3 * (temperature - temperature_base) ** 2

    macro_turb_vel = 3 * (aa1 + ba1 * (temperature - temperature_base) + ca1
                                    * (temperature - temperature_base) ** 2)

    return micro_turb_vel, macro_turb_vel


# Skipping for now.
def gaussian_attempt_3():
    print("Neither Cannon nor GUESS appropriate, trying Gaussian approach. Not yet implemented.")
    exit()

# Account for special modes: LBOL, SEIS, FIXED
# If Seis, prepare asteroseismic info
# If LBOL, prepare photomettric/astrometric info
# If Lbol/Seis, update logG based on provided info.
def update_mode_info(field_end, field_for_obs, data_file, data_ind):
    """
    Input:
        field_end: Type of mode. LBOL, SEIS, FIXED
        field_for_obs: Checks for if it is the sun. Requires special measures for if it is.
        reduction_and_analysis_data: Information on the star to produce the astromtric/asteroseismic information.
    Output:
        reduction_variable_dict: Contains the astromtric/asteroseismic data such as distance, parallax, magnitude of
                                k band, and more.
    """
    # Dictionary to put the variables in.
    reduction_variable_dict = {}

    # Elif just running in lbol mode, but not the sun we use a mnormal lbol update.
    if 'lbol' in field_for_obs:
        # Checks to see if it exists with good information.
        if reduction_and_analysis_data['r_est'] > 0:
            reduction_variable_dict = lbol_update(reduction_and_analysis_data)
        # If it doesn't, we can't continue.
        else:
            print("Star not in Gaia DR2, but lbol-keyword activated. Cancel.")
            exit()

    # If SEIS mode: get numax. A separate method to lbol to retrieve required astroseismic data.
    # Numax: The value of frequency at which the oscillation displays its strongest pulsation amplitude.
    if (field_end.lower()) == 'seis':
        try:
            numax = (reduction_and_analysis_data['numax'])
            e_numax = 0
            reduction_variable_dict = {'nu_max': numax[0], 'e_numax': e_numax}
        except KeyError as missingvalue:
            print("No numax value. Returning.")
            exit()

    return reduction_variable_dict


def get_gaia_params(head_file, head_ind):
    """
    Get parameters from Gaia. The extinction and distance parameters are 
    retrieved from external catalogues and this happens inside other functions
    which are called here.

    functions
    
    Parameters
    -------
        data_file: fits_hdu
            The data file containing all the information about the spectrum
        data_ind: Int
            The index corresponding to the object id
    Returns
    -------
        gaia_params_dict: dictionary
             The relevant Gaia parameters including:
                 G_mag,
                 R_mag,
                 I_mag,
                 GG_mag,
                 BP_mag,
                 RP_mag,
                 ebv,
                 plx,
                 e_plx,
                 dist,
                 dist_lo,
                 dist_hi,
                 e_dist,
                 e_g_mag,
                 a_g,
                 e_a_g,
                 lbol
    """
    print("Star in Gaia DR2. Running LBOL update.")

    # Magnitude information is contained in the first header extension
    spec_table = head_file[6].data[head_ind]

    # The magnitudes come directly from the WEAVE fits files
    G_mag = spec_table['MAG_G']
    e_G_mag = spec_table['EMAG_G']
    R_mag = spec_table['MAG_R']
    e_R_mag = spec_table['EMAG_R']
    I_mag = spec_table['MAG_I']
    e_I_mag = spec_table['EMAG_I']
    GG_mag = spec_table['MAG_GG']
    e_GG_mag = spec_table['EMAG_GG']
    BP_mag = spec_table['MAG_BP']
    e_BP_mag = spec_table['EMAG_BP']
    RP_mag = spec_table['MAG_RP']
    e_RP_mag = spec_table['EMAG_RP']

    # Get distances from CBJ catalogue (computed for gaia edr3)
    cbj_dict = pysme_gaia_dist_extinc.get_cbj_dist(head_file, head_ind)

    #Get extinction values for gaia
    extinction_dict = pysme_gaia_dist_extinc.get_extinction(cbj_dict, 
                                                                head_file, 
                                                                head_ind)

    #Get parallax and error directly from WEAVE spec file
#    plx = spec_table['TARGPARAL']
#    e_plx = nan 
    #Get parallax and error from either cbj or gaia edr3 catalogue
    gaia_id = extinction_dict['gaia_id']
    plx = extinction_dict['plx']
    e_plx = extinction_dict['e_plx']
    dist = cbj_dict['r_med']
    ebv = extinction_dict['ebv']
    ebv_lower = extinction_dict['ebv_lower']
    ebv_upper = extinction_dict['ebv_upper']
    a_g = extinction_dict['a_g']
    a_g_lower = extinction_dict['a_g_lower']
    a_g_upper = extinction_dict['a_g_upper']
    lbol = extinction_dict['lbol']
    lbol_lower = extinction_dict['lbol_lower']
    lbol_upper = extinction_dict['lbol_upper']
    bp_rp = extinction_dict['bp_rp']
    bprp_red = extinction_dict['bprp_red']
    bprp_red_lower = extinction_dict['bprp_red_lower']
    bprp_red_upper = extinction_dict['bprp_red_upper']


    # Compute uncertainties by assuming a symmetric distribution and taking 
    # the mean of the upper and lower percentiles

    def compute_unc(upper, lower):

        if upper != '--' and lower != '--':
            e_val = 0.5 * np.abs(upper - lower)

        else:
            e_val = '--'

        return e_val

    e_a_g = compute_unc(extinction_dict['a_g_upper'], 
                            extinction_dict['a_g_lower'])

    e_lbol = compute_unc(extinction_dict['lbol_upper'], 
                             extinction_dict['lbol_lower'])

    e_bprp_red = compute_unc(extinction_dict['bprp_red_upper'], 
                                 extinction_dict['bprp_red_lower'])

    e_ebv = compute_unc(extinction_dict['ebv_upper'], 
                                   extinction_dict['ebv_lower'])

    e_dist = compute_unc(cbj_dict['r_hi'], cbj_dict['r_lo'])

    #    if dist < 100:
    #        print("Star within 100pc, setting a_g and ebv to 0")
    #        a_g = 0
    #        ebv = 0

    reduction_variable_dict = {'gaia_id': gaia_id,
                               'G_mag': G_mag, 'e_G_mag': e_G_mag, 
                               'R_mag': R_mag, 'e_R_mag': e_R_mag,
                               'I_mag': I_mag, 'e_I_mag': e_I_mag, 
                               'GG_mag': GG_mag, 'e_GG_mag': e_GG_mag,
                               'BP_mag': BP_mag, 'e_BP_mag': e_BP_mag,
                               'RP_mag': RP_mag, 'e_RP_mag': e_RP_mag,
                               'BP_RP': bp_rp, 'a_g': a_g, 'e_a_g': e_a_g,
                               'plx': plx, 'e_plx': e_plx, 
                               'dist': dist, 'e_dist': e_dist,
                               'dist_lo': cbj_dict['r_lo'],
                               'dist_hi': cbj_dict['r_hi'], 
                               'lbol': lbol, 'e_lbol': e_lbol,
                               'ebv': ebv, 'e_ebv': e_ebv,
                               'bprp_red': bprp_red, 'e_bprp_red': e_bprp_red}

    return reduction_variable_dict


def update_gravity_function(reduction_variable_dict, field_end, 
                                starting_params):
    """
    We run the pysme_update_logg to modify gravity based on the starting 
    parameters we have. We continiously run that file throughout, but with the 
    updated parameters instead. Also updates the velocity parameters based on a 
    new gravity.

    Parameters
    -------
        reduction_variable_dict: dictionary
            Photometric and astrometric parameters including distance and 
            parallax and more, but not temperature, etc.
        field_end: string
            The type of run we are performing. Lbol or Seis.
        starting_params: dictionary
            The initial parameters of temperature, gravity, metallicity etc...
    Returns
    -------
        starting_params: dictionary
            With modified gravity and velocity parameters
    """
    # Update gravity. How we update it depends on Seis or Lbol
    # Send the paramters to be updated by the ELLI code
    gravity = \
        pysme_update_logg.optimize_logg_with_extra_input(starting_params, 
                                                       reduction_variable_dict, 
                                                       field_end)

    # The velocities depend on gravity as well, so we now must modify them.
    micro_turb_vel, macro_turb_vel = \
        microturbulence_macroturbulence(starting_params['Teff'], gravity)
    rotational_vel = macro_turb_vel
    macro_turb_vel = 0

    if field_end == 'lbol':
        print("Running in LBOL mode, glob_free are TEF, FEH, VRAD, (VSINI), \
               GRAV adjusted from TEFF, GMAG, PLX")

    # It shouldn't be able to reach here, but if it does then something 
    # unexpected occured with field_end and we stop.
    else:
        print("Error with naming convention in pysme_WEAVE.py. Field_end \
                should be lbol or seis, but is %s" %field_end)
        exit()

    starting_params['micro_turb_vel'] = micro_turb_vel
    starting_params['macro_turb_vel'] = macro_turb_vel
    starting_params['rotational_vel'] = rotational_vel
    starting_params['gravity'] = gravity

    return starting_params


"""Main part 3.3/5"
Ensure reasonable parameters"""


def reasonable_parameters(atmosphere_grid_file, starting_params):
    """
    We check to see if our first guess parameters are within reasonable limits. 
    If not, we adjust them slightly. We adjust gravity if the temperature is 
    out of bounds.

    Parameters
    -------
        atmosphere_grid_file: string
            The name of the atmosphere we use, as each will have different 
            limitations on their predictions.
        starting_params: dictionary 
            The initial parameters of temperature, gravity, metallicity etc...
    Returns
    -------
        starting_params: dictionary
            With modified gravity metallicity and temperature parameters
    """
    # Brings the values to within those boundaries.
    Teff = np.clip(starting_params['Teff'], 3000, 7500)
    gravity = np.clip(starting_params['gravity'], 0.5, 5)
    metallicity = np.clip(starting_params['metallicity'], -3, 0.5)

    # Adjust logg to marcs' lower grid limits of teff. Beyond that is not 
    # accurate enough to use these atmospheres.
    if (atmosphere_grid_file == 'marcs2014.sav' 
            or atmosphere_grid_file == 'marcs2012.sav'):
        # Sets grav to the maximum of either 2 or gravity (clipping with only a 
        # minimum value, no max)
        if 7500 > Teff > 6500:
            gravity = np.maximum(2, gravity)
        elif 6500 > Teff > 5750:
            gravity = np.maximum(1.5, gravity)
        elif 5750 > Teff > 5000:
            gravity = np.maximum(1, gravity)
        elif 5000 > Teff > 4000:
            gravity = np.maximum(0.5, gravity)

    # Do the same but with the stagger file limits for temperature, although 
    # still adjusting gravity according to temp.
    elif (atmosphere_grid_file == 'stagger-t5havni_marcs64.sav' or
            atmosphere_grid_file == 'stagger-t5havni_marcs.sav' or
            atmosphere_grid_file == 'stagger-tAA.sav'):

        # Sets grav to the maximum of either 2 or gravity (clipping with only a 
        # minimum value, no max)
        if 4750 > Teff:
            gravity = np.maximum(1.75, gravity)
        elif 5250 > Teff > 4750:
            gravity = np.maximum(2.75, gravity)
        elif 5750 > Teff > 5250:
            gravity = np.maximum(3.25, gravity)
        elif 6250 > Teff > 5750:
            gravity = np.maximum(3.75, gravity)
        elif Teff >= 6250:
            gravity = np.maximum(4.15, gravity)

    print("\nStarting values are: \nTeff:", Teff, 
          "\nGravity (log_g):", gravity,
          "\nmetallicity:", metallicity, "\n")

    starting_params['Teff'] = Teff
    starting_params['metallicity'] = metallicity
    starting_params['gravity'] = gravity

    return starting_params
