"""
We read the segment mask segm, removing the comment lines and columns.
We proceed to open the ccd1-4 files and take their wavelength and resolution 
out to plot a resolution mask and interpolating to be able to find resolution 
at peak wavelengths that we calculate at that point of the segment mask. We 
open the 4 ccd fits files to produce a phoneenumber_Sp file that contains 
Wavelength array, observed flux sign (sob) uncertainity in flux (uob)
(essentially error), smod (not important now) and mod that we call in 
make_struct to modify the wavelength array by the radial velocity doppler shift
and make some sme variables. Running create_structure should be the only thing 
needed and is called from pysme_execute Running it for the first time will take 
a good 5-10 minutes, but after that it should take a few seconds at a time. 
It's a trade off.
"""

import pysme_readlines
import pysme_run_sme
import pysme_interpolate_depth
import pandas as pd
import numpy as np
import pickle

from scipy.interpolate import interp1d
from datetime import date
from astropy.io import fits
from pysme.linelist.vald import ValdFile


# We add the final products to the sme_input (makestruct dictionary) before we 
# run pysme itself in the pysme_run_sme
# file.
def create_structure(makestruct_dict, reduction_variable_dict,
                     atomic_abundances_free_parameters=np.zeros(99), 
                     normalise_flag=False, sel_elem="all", run="Synthesise"):

    # Adjusts the accuracies ("resolution") of the spectra depending on the 
    # type of run.
    makestruct_dict = update_accuracies(makestruct_dict)

    makestruct_dict = update_atmosphere(makestruct_dict)
    # Updates the broadening profile (usually gauss) to adjust the spectra, and
    # also the values for mu and nmu, the
    # number of angles to calculate intensity.
    makestruct_dict = update_profile(makestruct_dict)
    # Produce the basic blocks of which wavelengths etc are inside our segment
    # masks. (The important atoms)
    # Produces an array of all wavelengths, and the indices to find which 
    # represent the start and end of the segments,
    # and the wavelength values of those as well. Calls many functions inside.
    ccd_wavelengths_inside_segmask_array, \
    ccd_flux_inside_segmask_array, \
    ccd_flux_error_inside_segmask_array, segment_begin_end, \
    wavelength_start_end_index, ccd_resolution_of_segment_array = \
    object_array_setup(makestruct_dict)

    # Number of segments from our segment mask.
    number_of_segments = len(wavelength_start_end_index)
    # Opens the files containing data on the linemasks and/or continuum.
    continuum_mask_data, linemask_data = open_masks(makestruct_dict)

    # We always want an array for pysme. Sets the v_rad and cscale for PySME. 
    # Adjusted during its run.
    radial_velocity = np.zeros(number_of_segments)
#    radial_velocity = \
#            np.repeat(makestruct_dict['vrad_global'], number_of_segments)
    continuum_scale = np.ones(number_of_segments)

    print("Normalise flag", normalise_flag)
    # Only normalise the data if that's how this file is called.
    # Runs the normalisation run to take flux down from the 10s of thousands to
    # below 1
    if normalise_flag:
        print("Running pre-normalise.")
        ccd_flux_norm_inside_segmask_array, \
            ccd_flux_norm_error_inside_segmask_array = \
                pre_normalise(ccd_wavelengths_inside_segmask_array, 
                              ccd_flux_inside_segmask_array,
                              ccd_flux_error_inside_segmask_array, 
                              segment_begin_end)

    # AFTER prenorm has been run once, run sme will have saved these normalise 
    # fluxes, and they have been loaded instead of un-normalied ones from 
    # the WEAVE data files in load_spectra, so we don't need to modify them.
    else:
        # Just to change to the new normalised variable name.
        ccd_flux_norm_inside_segmask_array = ccd_flux_inside_segmask_array
        ccd_flux_norm_error_inside_segmask_array = \
                ccd_flux_error_inside_segmask_array

    # Produce the array that tells us what fluxes to ignore, which are 
    # contiuum fluxes, error-laden, or atomic lines.
    flagged_fluxes = \
        create_observational_definitions(makestruct_dict, 
                            ccd_wavelengths_inside_segmask_array,
                            ccd_flux_norm_inside_segmask_array,
                            ccd_flux_norm_error_inside_segmask_array,
                            segment_begin_end, 
                            continuum_mask_data, 
                            linemask_data)

    # Information on atomic lines such as wavelength, species, ionization. 
    # Primarily used during the PySME run itself. During solve runs we take 
    # only atomic data in the segment mask, but that still appears to be the 
    # majority.
#    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, \
#            lower_level, upper_level, lu_lande, master_line_list = \
#        produce_indexed_atomic_data(makestruct_dict, segment_begin_end, 
#                                        linemask_data, run=run)

    cut_line_list, makestruct_dict = \
            produce_indexed_atomic_data(makestruct_dict, 
                                        segment_begin_end, 
                                        linemask_data, 
                                        sel_elem=sel_elem,
                                        run=run)

    # if we set lineatomic to 0 due to a lack of lines we give up on the run. 
    # Likely a balmer run so we have to make sure to flag that as False now, 
    # as we have just cancelled it.
#    if isinstance(line_atomic, int):
    if len(cut_line_list) == 0:
        makestruct_dict['balmer_run'] = False
        print("Line atomic is 0, cancelling balmer run.")
        return makestruct_dict

    # If we have any value for atomic abundances free parameters then we 
    # combine it and the globfree (vrad etc). So far un-needed and probably 
    # won't be used for a long time, until we adapt galah_ab
    if np.any(atomic_abundances_free_parameters):
        makestruct_dict["global_free_parameters"] = np.concatenate(
            np.asarray(makestruct_dict["global_free_parameters"]),
            atomic_abundances_free_parameters, axis=None)

#    # Else we just use free global parameters. but add Vsin for 
#    # non-normalisation runs if rot_vel is above 1. If it's too low it's 
#    # unimportant and we ignore it and set to 1 to adjust for it.
#    if run == "Solve":
#        if (makestruct_dict['rotational_velocity'] > 0 
#                and 'VSINI' not in makestruct_dict["global_free_parameters"]):
#            makestruct_dict["global_free_parameters"].append('VSINI')
#        else:
#            makestruct_dict['rotational_velocity'] = 1

    # Array of ags (0 or 1), specifying for which of the spectral lines the gf 
    # values are free parameters.
    spectral_lines_free_parameters = \
            np.zeros(len(cut_line_list['species'])) # gf_free
#
    # Strings to tell PYSME what kind of atmosphere we're looking for.
    atmosphere_depth, atmosphere_interpolation, atmoshpere_geometry, \
            atmosphere_method = \
                set_atmosphere_string(makestruct_dict['atmosphere_grid_file'],
                                      makestruct_dict['gravity'])

    # Turning our single large array into a list of smaller segmented arrays to 
    # comply with pysme iliffe standards. Can be done earlier, but now 
    # currently the code is dependent on them being single arrays so this seems
    # like an easier fix. Plus arrays are better to handle for large data so it 
    # might still be correct.
    iliffe_wave, iliffe_flux_norm, iliffe_error_norm, \
            iliffe_flags = segment_arrays(ccd_wavelengths_inside_segmask_array, 
                               ccd_flux_norm_inside_segmask_array,
                               ccd_flux_norm_error_inside_segmask_array, 
                               flagged_fluxes, segment_begin_end)

    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, \
            lower_level, upper_level, lu_lande = \
        cut_line_list['wlcent'], cut_line_list['lande'], \
        cut_line_list['depth'], cut_line_list['reference'], \
        cut_line_list['species'], cut_line_list[['j_lo', 'e_upp', 'j_up']], \
        cut_line_list['term_lower'], cut_line_list['term_upper'], \
        cut_line_list[['lande_lower', 'lande_upper']]

    # Updating the PySME input dictionary with the final variables. We modify 
    # the names to those that PySME accepts.
    sme_input = {'obs_name': makestruct_dict['obs_name'], 
                 'id': date.today(), 'spec': iliffe_flux_norm,
                 'teff': makestruct_dict['Teff'],
                 'depth': depth, 'logg': makestruct_dict['gravity'], 
                 'species': species, 'feh': makestruct_dict['metallicity'], 
                 'sob': ccd_flux_norm_inside_segmask_array,
                 'field_end': makestruct_dict['field_end'], 
                 'uob': ccd_flux_norm_error_inside_segmask_array,
                 'monh': makestruct_dict['metallicity'], 
                 'mu': makestruct_dict['intensity_midpoints'],
                 'vmic': makestruct_dict['micro_turb_vel'], 
                 'abundances': makestruct_dict['abundances'],
                 'vmac': makestruct_dict['macro_turb_vel'], 
                 'lande': lande_mean, 'chirat': 0.001,
                 'vsini': makestruct_dict['rotational_vel'], 
                 'vrad': radial_velocity, 'line_lulande': lu_lande,
                 'vrad_flag': makestruct_dict['vrad_flag'], 
                 'atomic': line_atomic,
                 'cscale': continuum_scale, 
                 'nmu': makestruct_dict['specific_intensity_angles'],
                 'gam6': 1, 'nlte_abund': makestruct_dict['nlte_abund'],
                 'mask': iliffe_flags,
                 'accwi': makestruct_dict['wavelength_interpolation_accuracy'],
                 'accrt': makestruct_dict['specific_intensity_accuracy_min'],
                 'maxiter': makestruct_dict['current_iterations'], 
                 'line_term_low': lower_level,
                 'atmo': {'source': makestruct_dict['atmosphere_grid_file'], 
                          'method': 'grid', 'depth': atmosphere_depth, 
                          'interp': atmosphere_interpolation,
                          'geom': atmoshpere_geometry}, 
                 'ipres': ccd_resolution_of_segment_array,
                 'object': makestruct_dict['obj_for_obs'], 
                 'iptype': makestruct_dict['broadening_profile'],
                 'uncs': iliffe_error_norm,
                 'gf_free': spectral_lines_free_parameters, 
                 'lineref': data_reference_array,
                 'line_term_upper': upper_level, 'nseg': number_of_segments,
                 'line_extra': j_e_array,
                 'wran': segment_begin_end, 'wave': iliffe_wave,
                 'wob': ccd_wavelengths_inside_segmask_array, 
                 'wind': wavelength_start_end_index[:, 1],
                 'balmer_run': makestruct_dict['balmer_run'], 
                 'mob': flagged_fluxes,
                 'cscale_flag': makestruct_dict['continuum_scale_flag'],
                 'fitparameters': makestruct_dict["global_free_parameters"],
                 'cscale_type': makestruct_dict['continuum_scale_type'],
                 'run_type': run
                 }

    # Save our input dict using pickle to allow for later runnin of sme 
    # manually. Only run on the first attempt as those are the original inputs.
    if not makestruct_dict['load_file'] and not makestruct_dict['balmer_run']:
        store_sme_input(sme_input)
    if run == "Synthesise":
        print("Dumping latest synth input to OUTPUT/")
        pickle.dump(sme_input, open("OUTPUT/latest_synth_input.pkl", "wb"))
    "And here we go. We finally run pysme."
    a =0
    for x in sme_input['wave']:
        a += len(x)
    print("len of wave after sme", a)

    sme_out = pysme_run_sme.start_sme(sme_input, cut_line_list, run)

    return makestruct_dict, sme_out


def update_accuracies(makestruct_dict):
    """
    Updates the accuracies (resolutions) of the specta required depending on 
    whether it's a balmer run or not. Also where we set the base accuracy.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Used to check whether it's a balmer run, and to add accuracy keys
    Returns                                                                     
    -------                                                                     
        makestruct_dict: dictionary
            Updated accuracy keys
    """
    # We want a higher accuracy for balmer runs but it's less important for non
    # balmer due to the wider segments.
    if makestruct_dict['balmer_run']:
#        makestruct_dict['wavelength_interpolation_accuracy'] = 0.00005
        makestruct_dict['wavelength_interpolation_accuracy'] = 1e-7
#        makestruct_dict['specific_intensity_accuracy_min'] = 0.00005
        makestruct_dict['specific_intensity_accuracy_min'] = 1e-3
    else:
        # Minimum accuracy for linear spectrum interpolation vs. wavelength.
#        makestruct_dict['wavelength_interpolation_accuracy'] = 0.00005
        makestruct_dict['wavelength_interpolation_accuracy'] = 1e-7
        # accwi in IDL

        # Minimum accuracy for sme.sint (Specific intensities on an irregular
        # wavelength grid given in sme.wint.)
        # at wavelength grid points in sme.wint. (Irregularly spaced 
        # wavelengths for specific intensities in sme.sint.)
        # Values above 10-4 are not meaningful.
#        makestruct_dict['specific_intensity_accuracy_min'] = 0.00005
        makestruct_dict['specific_intensity_accuracy_min'] = 1e-3
        # accrt in IDL

    return makestruct_dict


def update_atmosphere(makestruct_dict):
    """
    We set the atmosphere file to the backup if we didn't set it earlier

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Used to check for atmosphere
    Returns                                                                     
    -------                                                                     
        makestruct_dict: dictionary
            Updated atmosphere
    """
    try:
        makestruct_dict['atmosphere_grid_file'] = \
                makestruct_dict['atmosphere_grid_file']

    except AttributeError or NameError:
        # If we don't have a previously set atmosphere grid we use a backup 
        # 2012. Try is faster as we usually will have it set.
        print("Using marcs2012 instead")
        makestruct_dict['atmosphere_grid_file'] = 'marcs2012.sav'

    return makestruct_dict

 
def update_profile(makestruct_dict):
    """
    Set the profile broadening instruments and the angles used to calculate 
    specific intensity.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            sme input dictionary, just to update its keys
    Returns                                                                     
    -------                                                                     
        makestruct_dict: dictionary
            Updated profile keys
    """

    # Number of "equal-area" angles at which to calculate specific intensity.
    # Helps calculate the midpoints of the intensities
    specific_intensity_angles = 7
    # nmu

    # Type of profile used for instrumental broadening. Possible values are 
    # gauss sinc, or table. See Section 3.4.
    broadening_profile = "gauss"
    # iptype

    # The equal-area midpoints of each equal-area annulus for which specific 
    # intensities were calculated. values for Gaussian quadrature are not 
    # conducive to subsequent disk integration, named mu in sme.
    intensity_midpoints = np.flip(
        np.sqrt(0.5 * (2 * np.arange(1, specific_intensity_angles + 1)) 
                          / specific_intensity_angles))

    makestruct_dict['specific_intensity_angles'] = specific_intensity_angles
    makestruct_dict['intensity_midpoints'] = intensity_midpoints
    makestruct_dict['broadening_profile'] = broadening_profile

    return makestruct_dict


def object_array_setup(makestruct_dict):
    """
    Sets up the arrays of the wavelengths and fluxes that are inside our 
    segments. The output is not segmented, that is done later. However, it 
    would be a good idea to segment them at some point, but requires a decent 
    amount of code re-writing.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Contains wavelength and flux information, as well as file names to 
            open.
    Returns                                                                     
    -------                                                                     
        makestruct_dict: dictionary
            Updated profile keys
        ccd_wavelengths_inside_segmask_array: array_like
            Wavelengths inside our segments only.
        ccd_flux_inside_segmask_array: array_like 
            Corresponding fluxes
        ccd_flux_error_inside_segmask_array: array_like
            Corresponding error
        segment_begin_end: 
            The first and final wavelength of each segment
        wavelength_start_end_index: 
            The indices of the first and final wavelength of each segment
        ccd_resolution_of_segment_array: array_like
            Resolutions of the wavelengths we have
    """

#    # Unique file to the object in question.
#    segment_mask = makestruct_dict['segment_mask']

    # Contains the wavelengths we are most interested in.
    segment_mask_data_with_res = segment_mask_creation(makestruct_dict) 

    # Uses our resolution interpolation to find the resolution at the peak 
    # wavelengths that we also are finding here.
    segment_mask_data_with_res = interpolate_peak_resolution(
                                     segment_mask_data_with_res, 
                                     makestruct_dict['interpolation'])

    # Checks for a negative range
    if min(segment_mask_data_with_res['Wavelength_End'] - pd.to_numeric(
            segment_mask_data_with_res['Wavelength_Start'])) <= 0:
        print("Segment in %s has a negative range!" 
                    %makestruct_dict['segment_mask'])
        return

    # Checks for overlapping segments if there's more than one.
    if len(segment_mask_data_with_res['Wavelength_End']) > 1:
        if max(segment_mask_data_with_res['Wavelength_End'][
               0:len(segment_mask_data_with_res['Wavelength_End'])]
               - pd.to_numeric(segment_mask_data_with_res['Wavelength_Start'][
               1:len(segment_mask_data_with_res['Wavelength_Start'])])) > 0:

            print("Overlapping segments")

            return

    # We load in the wavelength etc variables to be used a lot in makestruct, 
    # first created in pysme_WEAVE and later by SME.
    total_ccd_wavelength, total_ccd_flux, total_ccd_flux_error = \
            load_spectra(makestruct_dict)


    # We limit our wavelength array to those within the segment mask we loaded,
    # and their appropriate fluxes and resolutions.
    ccd_wavelengths_inside_segmask_array, \
        ccd_flux_inside_segmask_array, \
        ccd_flux_error_inside_segmask_array, \
        ccd_resolution_of_segment_array, \
        wavelength_start_end_index = \
        wavelengths_flux_inside_segments(segment_mask_data_with_res, 
                                         total_ccd_wavelength, 
                                         total_ccd_flux,
                                         total_ccd_flux_error)

    # Number of segments that contain visible spectra.
    if len(wavelength_start_end_index) == 0:
        print("No observations in segment mask")
        return

    # Creates an array with the beginning and end wavelengths of the segments.
    # Different to the start end array as that's indices rather than the
    # wavelength values
    segment_begin_end = find_segment_limits(wavelength_start_end_index, 
                                                total_ccd_wavelength)

    return ccd_wavelengths_inside_segmask_array, \
           ccd_flux_inside_segmask_array, \
           ccd_flux_error_inside_segmask_array, \
           segment_begin_end, wavelength_start_end_index, \
           ccd_resolution_of_segment_array


def segment_mask_creation(makestruct_dict):
    """
    Loads the segment mask and the resolutions from the csv file. The segments 
    represent the wavelengths of the atoms we are interested in.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Contains the location and name of the segment mask file
    Returns                                                                     
    -------                                                                     
        segment_mask_data_with_res: pandas data_frame
            The data of the segment mask file, particularly the wavelengths of 
            each segment.
    """

    # Segm_mask is _Segm.data, unsurprisingly. It takes the start, end 
    # wavelength, and the resolution base guess of 21k which is about the
    # middle of the range we expect for WEAVE (~13000 - 25000).

    segment_mask_data_with_res = pd.read_csv(
        makestruct_dict['original_location'] + makestruct_dict['segment_mask'],
        delim_whitespace=True,
        header=None,
        names=["Wavelength_Start", "Wavelength_End", "Resolution", "comment", 
                    "overflow"],
        engine='python',
        skipinitialspace=True,
        comment=';',
        usecols=["Wavelength_Start", "Wavelength_End",
                    "Resolution"])

    # Sort in ascending order of starting wavelength
    segment_mask_data_with_res.sort_values(by=['Wavelength_Start', 
                                                    'Wavelength_End'])

    return segment_mask_data_with_res


def interpolate_peak_resolution(segment_mask_data_with_res, interpolation):
    """
    Uses our previously created resolution interpolation to interpolate at the
    peak wavelength (which we create)

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Has the resolution factor from galah_sp1
        segment_mask_data_with_res: pandas data_frame
            Has the wavelngth information of the segments
        interpolation: Equation for producing resolution at wavelength center.
    Returns                                                                     
    -------                                                                     
        segment_mask_data_with_res: pandas data_frame
            With now added center wavelengths and their resolution as an 
            additional column.
    """

    # Create new column with the central wavelength in each segment
    segment_mask_data_with_res.insert(0, "Wavelength_Peak", 
            0.5 * (segment_mask_data_with_res.Wavelength_Start 
            + segment_mask_data_with_res.Wavelength_End))

    # Replace resolution values with the values from our previously computed
    # interpolation function
    segment_mask_data_with_res['Resolution'] = \
        interpolation(segment_mask_data_with_res.Wavelength_Peak)

    return segment_mask_data_with_res


def load_spectra(makestruct_dict):
    """
    Opens the appropriate file depending on which iteration of SME we are
    running. If it's the first run before SME, we just use the un-normalised 
    spectra which we will later normalise. It also uses our previously created 
    resolution interpolation to interpolate at the peak wavelength (which we 
    create)

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Information on the run type and whether we need to load the data 
            file, and its file name.
    Returns                                                                     
    -------                                                                     
        spectra_data: dictionary or array_like 
            Regardless of what spectra_data is, we return its component arrays 
            of wavelength, flux, and error
    """
    # Change it if load file is set to the one SME outputs.
    if makestruct_dict['load_file']:
        # During non balmer synth runs we load the normal spectra, and then 
        # save a duplicate version for the balmer run to be able to copy 
        # exactly all the data that the normalise run is using.
        if not makestruct_dict['balmer_run']:
            spectra_data = \
                pickle.load(open("OUTPUT/SPECTRA/" 
                + makestruct_dict['obs_name'] 
                + "_SME_spectra.pkl", "rb"))

            # Creating a duplicate for the next balmer run
            pickle.dump(spectra_data, 
                    open("OUTPUT/SPECTRA/Temp_Balmer_spectra_input_%s.pkl" 
                            %makestruct_dict['obj_for_obs'], "wb"))

        # For the balmer run we want to load the same spectra that the 
        # normalisation run just loaded, NOT what it created due to the 
        # doppler shifting and potential pysme edits.
        elif makestruct_dict['balmer_run']:
            spectra_data = pickle.load(
                    open("OUTPUT/SPECTRA/Temp_Balmer_spectra_input_%s.pkl" 
                            %makestruct_dict['obj_for_obs'], "rb"))
    # This is for before SME runs we open the one created in pysme_exec.
    else:
        spectra_data = makestruct_dict['unnormalised_spectra']
    # If the first value in the wavelength is an array with all wavelengths in 
    # the first segment. If it's not an array it's not segmented, the first 
    # value is just a float and we don't need to desegment it.
    # mask exists in the dict, but not used in calculations.
    if isinstance(spectra_data['wave'][0], np.ndarray):

        wave = spectra_data['wave'].flatten()
        flux = spectra_data['flux'].flatten()
        error = spectra_data['error'].flatten() 

        return wave, flux, error

    return spectra_data['wave'], spectra_data['flux'], spectra_data['error']


def wavelengths_flux_inside_segments(segment_mask_data_with_res, 
                                     total_ccd_wavelength, 
                                     total_ccd_flux,
                                     total_ccd_flux_error):
    """
    Finds the wavelengths that we have that are also inside the segments.

    Parameters                                                                  
    -------                                                                     
        segment_mask_data_with_res: pandas data_frame 
            Contains the start and end of the desired segments of wavelengths
        total_ccd_wavelength: array_like
            All wavelength values of the spectrum
        total_ccd_flux: array_like
            Flux of the wavelengths
        total_ccd_flux_error: array_like
            Error of flux
        total_ccd_wavelength: All wavelength array

    Returns                                                                     
    -------                                                                     
        ccd_wavelengths_inside_segmask_array: array_like 
            A large array of ALL wavelengths inside ALL segments
        ccd_flux_inside_segmask_array: array_like
            Their corresponding fluxes
        ccd_flux_error_inside_segmask_array: array_like 
            Their corresponding error
        ccd_resolution_of_segment_array: array_like
            Resolutions of the x/y pair
        wavelength_start_end_index: array_like
            The indices in our array that represent the start and end of the 
            segments.
    """
    # Can't be sure how many data points there will be, so use a temp list that
    # we can fill as we go.
    ccd_wavelengths_inside_segmask = []
    ccd_flux_inside_segmask = []
    ccd_flux_error_inside_segmask = []
    ccd_resolution_of_segment = []

    # Array for the first and final wavelength indices of each segment.
    wavelength_start_end_index = \
            np.zeros((len(segment_mask_data_with_res["Wavelength_Start"]), 2))

    # For each segment in segmask, find the values of dopplered wavelength 
    # (and associated flux from indexing) that are inside. Despite having this 
    # array we still use np.where most times to find the wavelengths in the 
    # segments, maybe change this if there's time.
    for segment in range(len(segment_mask_data_with_res["Wavelength_Start"])):
        # Beginning wavelength and end of that segment. Put as variables here 
        # for readability.

        seg_start = \
                segment_mask_data_with_res["Wavelength_Start"].values[segment]
        seg_stop = \
                segment_mask_data_with_res["Wavelength_End"].values[segment]

        # Finding the index of values inside the segment, using "logical and"
        # is a neccesity.
#        wavelength_inside_segmask_index_1 = np.where(
#            np.logical_and(seg_stop >= total_ccd_wavelength, 
#                               total_ccd_wavelength >= seg_start))

        wavelength_inside_segmask_index = np.where(
                                          (seg_stop >= total_ccd_wavelength) 
                                        & (total_ccd_wavelength >= seg_start))

        # Adding the wavelengths inside the segment to our list of wavelengths 
        # of ALL segment wavelengths. We resegment it later.
        ccd_wavelengths_inside_segmask.extend(
            total_ccd_wavelength[wavelength_inside_segmask_index])
        ccd_flux_inside_segmask.extend(
            total_ccd_flux[wavelength_inside_segmask_index])
        ccd_flux_error_inside_segmask.extend(
            total_ccd_flux_error[wavelength_inside_segmask_index])

        # Numpy array of indices of the first and final wavelengths per segment 
        # with column 0 being the first
        if wavelength_inside_segmask_index[0].size != 0:
            wavelength_start_end_index[segment, 0] = \
                    (wavelength_inside_segmask_index[0][0])
            wavelength_start_end_index[segment, 1] = \
                    (wavelength_inside_segmask_index[-1][-1])
        ccd_resolution_of_segment.append(
                    segment_mask_data_with_res['Resolution'][segment])

    # Convert lists into arrays for later numpy indexing with np.where.
    ccd_wavelengths_inside_segmask_array = \
            np.array(ccd_wavelengths_inside_segmask)
    ccd_flux_inside_segmask_array = \
            np.array(ccd_flux_inside_segmask)
    ccd_flux_error_inside_segmask_array = \
            np.array(ccd_flux_error_inside_segmask)
    ccd_resolution_of_segment_array = np.array(ccd_resolution_of_segment)

    return ccd_wavelengths_inside_segmask_array, \
           ccd_flux_inside_segmask_array, \
           ccd_flux_error_inside_segmask_array, \
           ccd_resolution_of_segment_array, wavelength_start_end_index


def find_segment_limits(wavelength_start_end_index, total_ccd_wavelength):
    """
    Converts the segement windows from indices to wavelength values.

    Parameters                                                                  
    -------                                                                     
        Wavelength_start_end_index: 
            The indices of the beginning and ending of each segment in our 
            wavelength array. 
        total_ccd_wavelength: array_like
            All wavelengths in all segments.
    Returns                                                                     
    -------                                                             
        segment_begin_end: array_like
            A two column, multi row wavelength array. [:, 0] is the start, 
            [:, 1] the end of each segment
    """

    # An array with two columns, the first and last recorded wavelength in each 
    # segment
    # copy to avoid overwriting wind. Just using it for size.
    segment_begin_end = np.copy(wavelength_start_end_index)  
    for windex_row in range(len(wavelength_start_end_index)):
        # At indices 0,0 and 0,1 (and then 1,0 etc) of the index array 
        # 'wavelength_start_end_index' we take the value and apply it to the 
        # wavelngth array as the values we have taken are indices of the first 
        # and last wavelength of each segment. windrow, 0 is the segment 
        # beginning. , 1 is the end.
        segment_begin_end[windex_row, 0] = total_ccd_wavelength[
            int(wavelength_start_end_index[windex_row, 0])]
        segment_begin_end[windex_row, 1] = total_ccd_wavelength[
            int(wavelength_start_end_index[windex_row, 1])]

    return segment_begin_end


#def desegment(spectra):
#    """
#    Our code is actually set up to not use segmented arrays, so we desegment it
#    here, run it through the code, and later RE-segment it. Obviously this is 
#    not ideal.
#
#    Parameters                                                                  
#    -------                                                                     
#        spectra: dictionary
#            Contains wavelength, flux, and error arrays for the spectrum
#    Returns                                                                     
#    -------                                                             
#        wave: array_like
#            Wavelength values for the spectrum
#        flux: array_like
#            Flux values for the spectrum
#        error: array_like
#            Flux error values for the spectrum
#    """
#    wave, flux, error = [], [], []
#
#    for value in spectra['wave']:
#        wave.extend(value)
#    wave = np.asarray(wave)
#
#    for value in spectra['flux']:
#        flux.extend(value)
#    flux = np.asarray(flux)
#
#    for value in spectra['error']:
#        error.extend(value)
#    error = np.asarray(error)
#
#    return wave, flux, error


def pre_normalise(ccd_wavelengths_inside_segmask_array, 
                      ccd_flux_inside_segmask_array,
                      ccd_flux_error_inside_segmask_array, 
                      segment_begin_end):
    """
    This is the function to prenormalise the observed spectral line. Calls on 
    autonormalise, removes data that is too far away from the continuum and 
    uses the closer data to normalise itself.

    Parameters                                                                  
    -------                                                                     
        ccd_wavelengths_inside_segmask_array: array_like
            All wavelengths in all segments
        ccd_flux_inside_segmask_array: array_like
            The corresponding fluxes
        ccd_flux_error_inside_segmask_array: array_like
            The error in the flux
        segment_begin_end: array_like
            The start and final wavelengths of each segment, to segment 
            ccd_wave.. using np.where()

    Returns                                                                     
    -------                                                                     
        ccd_flux_inside_segmask_array: array_like
            The flux values, now normalised
        ccd_flux_error_inside_segmask_array: array_like
            The corresponding values for flux error
    """

    # Pre-normalization steps. ";Performs first-guess normalisation by robustly
    # converging straight line fit to high pixels"

    for segment_band in range(len(segment_begin_end)):

        # Finds the index where the wavelengths are between the start and end 
        # of each segment, to be able to loop each seg as i. We repeat this 
        # step quit often in the code, and things similar to it to find 
        # "inside" the segment. Sometimes it's useless as we already have a 
        # list of inside the segments, but this time it is segmenting it all.
#        segment_indices = (np.where(
#                               np.logical_and(
#                                   ccd_wavelengths_inside_segmask_array >=
#                                   segment_begin_end[segment_band, 0],
#                                   ccd_wavelengths_inside_segmask_array <=
#                                   segment_begin_end[segment_band, 1])))[0]

        segment_indices = np.where((ccd_wavelengths_inside_segmask_array >=
                                   segment_begin_end[segment_band, 0]) 
                                   & (ccd_wavelengths_inside_segmask_array <=
                                   segment_begin_end[segment_band, 1]))[0]

#        print('len of segment', len(segment_indices))
        # If count is greater than 20, we can normalise. len(segindex) is the 
        # number of values that fit our criteria.
        if len(segment_indices) > 20:
            # Take the coefficients of the polyfit, and the flux of it too 
            # using the equation from IDL and then applies a ploynomial fit, 
            # removing outlying values until it no longer changes. The outlying
            # value limit is hardcoded. We then use the equation fit to it to 
            # normalise it
            continuous_function = autonormalisation(
                ccd_wavelengths_inside_segmask_array[segment_indices],
                ccd_flux_inside_segmask_array[segment_indices], 1, 0)

            # Puts it to a relative flux out of 1 that we see in the abundance 
            # charts. Have to take relative indices for cont_func to have the 
            # correct shape.
#            print('flux in segment', ccd_flux_inside_segmask_array[segment_indices][-40:])
#            print('normalising function', continuous_function[-40:])
#            print('length of flux in segment', len(ccd_flux_inside_segmask_array[segment_indices]))
#            print('length of normalising function', len(continuous_function))
            ccd_flux_inside_segmask_array[segment_indices] = \
                    ccd_flux_inside_segmask_array[segment_indices] \
                    / continuous_function
            ccd_flux_error_inside_segmask_array[segment_indices] = \
                    ccd_flux_error_inside_segmask_array[segment_indices] \
                    / continuous_function
#            print(ccd_flux_inside_segmask_array[segment_indices][-40:])

        # If we don't have enough points, we just use the mean value instead. 
        # Numpy mean did not work sometimes.
        else:
            # np.mean had issues with our results. Making a variable for 
            # readability.
            flux_mean = \
                (sum(ccd_wavelengths_inside_segmask_array[segment_indices])) \
                / len(ccd_wavelengths_inside_segmask_array[segment_indices])

            # Must be ordered correctly or we modify the sob before we use it 
            # to modify uob!
            ccd_flux_error_inside_segmask_array[segment_indices] = \
                ccd_flux_error_inside_segmask_array[segment_indices] \
                / flux_mean

            ccd_flux_inside_segmask_array[segment_indices] = \
                (ccd_flux_inside_segmask_array[segment_indices]) / flux_mean

    # end of prenormalisation. We return the normalised spectrum 

    return ccd_flux_inside_segmask_array, ccd_flux_error_inside_segmask_array


def autonormalisation(wavelength_array, flux_array, polynomial_order, 
                          fit_parameters):
    """
    We use a hard coded equation to aid in the normalisation of the flux 
    depending on the polynomial order we choose.

    Parameters                                                                  
    -------                                                                     
        wavelength_array: array_like
            Wavelength values in the segment
        flux_array: array_like
            Fluxe values in the segment
        polynomial_order: int
            Order of polynomial used to fit this segment
        fit_parameters: int
            More paramters used to adjust the fit

    Returns                                                                     
    -------                                                                     
        continuous_function: array_like
            Normalization factors with which to normalise the flux in a given
            segment
    """

    # To be used when checking to see if we've removed too many values and when
    # creating cont_fn.
    original_wave = wavelength_array
    if polynomial_order == 0:
        polynomial_order = 2
    if fit_parameters == 0:
        fit_parameters = 1.5

    # Stops the IDE throwing a "not created" fit. Unimportant and completely 
    # useless otherwise.
    polyfit_coefficients = 1
    inlier_index = 0
    continuous_function = 1

    # Using numpy polyfit to replace the idl Robust_poly_fit. Simply gets a 
    # polynomial fit for the spectra, and repeats
    # until either converged, or reaches 99.
    for polyfit_loop in range(0, 99):
        # Gets the coefficients for a fit of the order polynomial_order
        polyfit_coefficients = \
                np.polynomial.polynomial.polyfit(wavelength_array, 
                                                     flux_array, 
                                                     polynomial_order)

        # Uses these to get an array of the y values of this line
        fitted_flux = np.polynomial.polynomial.polyval(wavelength_array, 
                                                          polyfit_coefficients)

        # Creates an array of the error (sigma) of the line compared to 
        # original data, to find outliers by getting the standard deviation of 
        # the difference
        fitted_sigma = np.std(flux_array - fitted_flux)
        # Find the fluxes that exist below the linear fit + error but above the
        # lower error boundary (* p)
        # So not outliers, but inliers. We take the first value from the output
        # which is the index array itself
        inlier_index = (np.where(
                            np.logical_and(
                                flux_array < (fitted_flux 
                                              + (2 * fitted_sigma)),
                                (flux_array 
                                    > (fitted_flux - (fit_parameters 
                                                      * fitted_sigma)))
                                          )))[0]

        # If poly_order is wrong, we just stick with a value of 1 to keep 
        # everything the same
        continuous_function = 1
        if polynomial_order == 2:
            continuous_function = polyfit_coefficients[0] \
                                  + (polyfit_coefficients[1] * original_wave) \
                                  + (polyfit_coefficients[2] 
                                     * original_wave ** 2)
        elif polynomial_order == 1:
            continuous_function = polyfit_coefficients[0] \
                                  + (polyfit_coefficients[1] * original_wave)

        # Stops when no more convergence occurs, breaks the loop. Again, 
        # np.where gives a tuple with dtype the second condition uses the 
        # original non edited wavelength array
        if (len(inlier_index) == len(wavelength_array)) \
                or (float(len(inlier_index)) / float(len(original_wave)) 
                                                                    <= 0.1):
            break
        if polyfit_loop >= 98:
            print("Normalization polynomial fit did not converge")
            break
        # Replace the array with only values that lie inside the error 
        # boundaries we have.
        wavelength_array = wavelength_array[inlier_index]
        flux_array = flux_array[inlier_index]
    # Currently non-useful due to not needing a 2nd order polynomial fit.
    if polynomial_order == 2:
        # co in idl, compared to the continuous flux of c. These variables, 
        # man.. Does not get returned in make struct. Additional just means 
        # there's something different that I don't know.
        continuous_function_additional = \
                polyfit_coefficients[0] + polyfit_coefficients[1] * \
                wavelength_array[inlier_index] + polyfit_coefficients[2] * \
                wavelength_array[inlier_index] ** 2
    elif polynomial_order == 1:
        continuous_function_additional = \
                polyfit_coefficients[0] + polyfit_coefficients[1] * \
                wavelength_array[inlier_index]
    else:
        continuous_function_additional = 1

    return continuous_function


def open_masks(makestruct_dict):
    """
    Open up the line and continuum masks for making mob/flagged fluxes. 
    Linemask is xxx_Sp.dat, segment mask is xxx.Segm.dat, and continuum mask is
    xxx.Cont.dat. They contain the information on the important atom 
    wavelengths, and the length of our desired continuum.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary 
            Contains the file name unique to the object if created.

    Returns                                                                     
    -------                                                                     
        continuum_mask_data: pandas data_frame 
            Information on the length and size of the continuum
        linemask_data: pandas data_frame
            Information on the wavelengths and species of important atomic 
            segments
    """
    # Reads out the columns which are centre/peak wavelength, start and end of 
    # wavelength peak (all simulated, not observed), and atomic number. 
    # Seperated by 2 spaces (sep), no headers as it is only data, and the names 
    # are what we assign them as to be used later with 
    # linemask_data["Sim_wave" etc.] Careful with the python engine, it's 
    # slower. If we are looking at BIG data files this might be bad.
    print("Linemask name here is ", makestruct_dict['line_mask'])
    linemask_data = pd.read_csv(makestruct_dict['line_mask'], 
                                    delim_whitespace=True, 
                                    header=None,
                                    engine='python', 
                                    comment=';',
                                    names=["Sim_Wavelength_Peak", 
                                           "Sim_Wavelength_Start", 
                                           "Sim_Wavelength_End", 
#                                           "Atomic_Number"])
                                           "Atom"])

    # Read the start and end of the continuum
    continuum_mask_data = pd.read_csv(makestruct_dict['cont_mask'], 
                                          delim_whitespace=True, 
                                          header=None,
                                          engine='python', 
                                          names=["Continuum_Start", 
                                                 "Continuum_End"])

    return continuum_mask_data, linemask_data


def create_observational_definitions(makestruct_dict, 
                                     ccd_wavelengths_inside_segmask_array,
                                     ccd_flux_norm_inside_segmask_array, 
                                     ccd_flux_norm_error_inside_segmask_array,
                                     segment_begin_end, 
                                     continuum_mask_data, 
                                     linemask_data):
    """
    Sums the flagged flux creation functions in one to produce a final product.
    Flags the wavelengths of the peaks, continuum, and ones that need to be 
    removed. ---> 2: Continuum, 1: Atomic line, 0: Removed/Ignore

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary 
            Contains the file name unique to the object if created.
        ccd_wavelengths_inside_segmask_array: array_like
            Wavelengths to associate with atomic lines from linemask
        ccd_flux_norm_inside_segmask_array: array_like
            Fluxes to see what is a good continuum line, and what is not
        ccd_flux_norm_error_inside_segmask_array: array_like
            Identify high error poor continuum choices to remove
        segment_begin_end: 
            Start and end values of each segment, as they must be flagged 
            individually
        continuum_mask_data: pandas data_frame 
            Information on the length and size of the continuum
        linemask_data: pandas data_frame
            Information on the wavelengths and species of important atomic 
            segments

    Returns                                                                     
    -------                                                                     
        flagged_fluxes: array_like
            Indicates which fluxes to remove, keep, and which are the important 
            atomic lines
    """
    # We first flag fluxes that appear to be absorption lines,
    # then the continuum, then remove those with too low a flux to be used
    # inputting the previously created array each time.
    flagged_fluxes = flag_absorption(linemask_data, 
                                     ccd_wavelengths_inside_segmask_array,
                                     ccd_flux_norm_inside_segmask_array, 
                                     ccd_flux_norm_error_inside_segmask_array)

    # We only need the continuum during synthesize runs where cscale flag is 
    # linear, otherwise it's fixed. Same as flagging the absorption lines, we 
    # compared to noise and flag all non-AL as continuum that we then narrow 
    # down in the next function.
    if makestruct_dict['continuum_scale_flag'] == 'linear':
        flagged_fluxes = \
                flag_continuum(ccd_wavelengths_inside_segmask_array,
                                   ccd_flux_norm_inside_segmask_array, 
                                   ccd_flux_norm_error_inside_segmask_array,
                                   flagged_fluxes, 
                                   continuum_mask_data)

    # Now we remove the fluxes that are below our desired continuum line as 
    # they will affect our steady flux = 1 normalisation. We leave the 
    # absorption lines alone here.
    flagged_fluxes = cutting_low_flux(ccd_wavelengths_inside_segmask_array, 
                                      ccd_flux_norm_inside_segmask_array,
                                      segment_begin_end, 
                                      flagged_fluxes)

    # Finally, we remove the lines close to the NLTE-dominated cores
    flagged_fluxes = removing_nlte(makestruct_dict, 
                                   ccd_wavelengths_inside_segmask_array,
                                   ccd_flux_norm_inside_segmask_array, 
                                   flagged_fluxes)

    return flagged_fluxes


def flag_absorption(linemask_data, 
                    ccd_wavelengths_inside_segmask_array, 
                    ccd_flux_norm_inside_segmask_array,
                    ccd_flux_norm_error_inside_segmask_array):
    """
    Flag the fluxes that correspond to the atomic absorption lines. Other 
    functions set the continuum and undesirables. All values inside Sp are 
    flagged as they are the desired wavelengths. Checks the fluxes averages 
    over 4 data points each and compares it to the noise to check suitability.

    Parameters                                                                  
    -------                                                                     
        linemask_data: pandas data_frame
            Information on the wavelengths and species of important atomic 
            segments
        ccd_wavelengths_inside_segmask_array: array_like
            Wavelengths to associate with atomic lines from linemask
        ccd_flux_norm_inside_segmask_array: array_like
            Fluxes per wavelength
        ccd_flux_norm_error_inside_segmask_array: array_like
            Flux error

    Returns                                                                     
    -------                                                                     
        flagged_fluxes: array_like
            An array of 0s (Default ignores) and 1s (Absorption lines). 2s are 
            added in the continuum function. Same size (and represents) the 
            wavelengths.
    """

    # Mask of observed pixels, just for placement to then flag them as 
    # continuum, etc.
    flagged_fluxes = np.zeros(len(ccd_wavelengths_inside_segmask_array))

    # We flag the fluxes that are probably peaks inside our segments that we 
    # care about that are probably atomic absorbtion lines. We loop for each 
    # segment.
    for line_loop in range(0, len(linemask_data['Sim_Wavelength_Start'])):

        # Usual case of making sure the wavelengths we want to use are in the 
        # lines.
        wavelengths_inside_linemask_index = np.where(np.logical_and(
            ccd_wavelengths_inside_segmask_array 
                >= linemask_data['Sim_Wavelength_Start'][line_loop],
            ccd_wavelengths_inside_segmask_array <= 
                linemask_data['Sim_Wavelength_End'][line_loop]))

        # running_snr in idl, sets values to 1 if they're below the max noise 
        # spike. This means we flag all good points at 1 that are below spikes 
        # in noise to be used. Acts as a proxy for the signal to noise ratio 
        signal_to_noise = []
        # We're trying to find signal to noise ratios, where 1.5 is the limit 
        # for our noise
        # Averages over +/-4 values.
        for flux_row in range(0, len(ccd_flux_norm_inside_segmask_array)):
            # Indexes the obs flux from ii-4 to ii + 4 (or limits of length)
            signal_to_noise.append(max(
                [1 + 10 / (np.mean(ccd_flux_norm_inside_segmask_array[
                    max(0, flux_row - 4):min(flux_row + 4, 
                        len(ccd_flux_norm_inside_segmask_array) - 1)] /
                    ccd_flux_norm_error_inside_segmask_array[
                    max(0, flux_row - 4):min(flux_row + 4, 
                        len(ccd_flux_norm_inside_segmask_array) - 1)])), 1.5]))
        signal_to_noise = (np.array(signal_to_noise))

        # If there are some wavelengths in the segment, we can see if they're 
        # peaks.
        if len(wavelengths_inside_linemask_index[0]) > 0:

            # If the flux exists and is less than the noise, set a marker to 1 
            # to indicate this. (Atomic line)
            if (min(ccd_flux_norm_inside_segmask_array[
                        wavelengths_inside_linemask_index[0]]) > 0 
                    and max(ccd_flux_norm_inside_segmask_array[
                                wavelengths_inside_linemask_index[0]]) 
                    < max(signal_to_noise[
                            wavelengths_inside_linemask_index[0]])):
                # 1 is a good thing! it means it fits nicely in the peak and 
                # the noise is nothing to worry about
                flagged_fluxes[wavelengths_inside_linemask_index[0]] = 1

    # Return our array of which wavelengths are flagged as possible peaks to be 
    # modified further in contiuum and more.
    return flagged_fluxes


def flag_continuum(ccd_wavelengths_inside_segmask_array, 
                       ccd_flux_norm_inside_segmask_array,
                       ccd_flux_norm_error_inside_segmask_array, 
                       flagged_fluxes, 
                       continuum_mask_data):
    """
    Flags continuum lines during Synthesize runs, sets non absorption lines 
    that also meet our error/noise requirements to 2 to represent continuum. 
    If they don't meet the noise requirements, sets to 0 as they are unusabe.

    Parameters                                                                  
    -------                                                                     
        ccd_wavelengths_inside_segmask_array: array_like
            Wavelength values of spectrum 
        ccd_flux_norm_inside_segmask_array: array_like
            Normalized fluxes
        ccd_flux_norm_error_inside_segmask_array: array_like
            Error of normalized flux
        continuum_mask_data: pandas data_frame 
            Locating the continuum to check its wavelength region
        flagged_fluxes: array_like
            The previous located absorption lines, as we DO NOT want these as 
            continuum. Never.
    Returns                                                                     
    -------                                                                     
        flagged_fluxes: array_like
            An array of 0s (Default ignores), 1s (Absorption lines) 
            and 2s (Continuum). Same length as the wavelength array.
    """

    # For each segment in the continuum file, often will be one large range,
    for continuum_loop in range(0, 
                                len(continuum_mask_data['Continuum_Start'])):

        # A list to append to with the signal to noise ratios, either 1.5 or 
        # 1 + 10/mean(flux/error) from i to ii -/+4. So the higher the flux and 
        # lower the error, the more likely to have 1.5 as our baseline
        signal_to_noise = []
        for flux_row in range(0, len(ccd_flux_norm_inside_segmask_array)):
            signal_to_noise.append(
                max([1 + 10 / np.mean(
                    ccd_flux_norm_inside_segmask_array[
                        max(0, flux_row - 4) : min(flux_row + 4, 
                        len(ccd_flux_norm_inside_segmask_array))]
                    / ccd_flux_norm_error_inside_segmask_array[
                        max(0, flux_row - 4) : min(flux_row + 4, 
                        len(ccd_flux_norm_inside_segmask_array))]),
                    1.5]))
        signal_to_noise = (np.array(signal_to_noise))

        # Indexes where the wavelengths are inside the continuum.
        wavelengths_inside_continuum_index = np.where((np.logical_and(
            ccd_wavelengths_inside_segmask_array 
                >= continuum_mask_data['Continuum_Start'][continuum_loop],
            ccd_wavelengths_inside_segmask_array 
                <= continuum_mask_data['Continuum_End'][continuum_loop])))

        # This is cleaner code (albeit slower I imagine) than having 5 
        # np.logical ands in the above statement. If we haven't already flagged
        # the flux as peak flux, and it's less than the noise then we flag it 
        # with '2' to mark it as continuum
        if len(wavelengths_inside_continuum_index[0]) > 0:
            for continuum_index in wavelengths_inside_continuum_index[0]:

                if (flagged_fluxes[continuum_index] != 1 and
                        0 < ccd_flux_norm_inside_segmask_array[continuum_index] 
                          < signal_to_noise[continuum_index]):
                    flagged_fluxes[continuum_index] = 2

    return flagged_fluxes


def cutting_low_flux(ccd_wavelengths_inside_segmask_array, 
                     ccd_flux_norm_inside_segmask_array, 
                     segment_begin_end,
                     flagged_fluxes):
    """
    This function removes the lowest 70% of fluxes while retaining enough 
    points on both sides of peak to have a continuum

    Parameters                                                                  
    -------                                                                     
        ccd_wavelengths_inside_segmask_array: array_like
            Wavelength values of spectrum. Used to ensure a continuum on both 
            sides of the absorption line.
        ccd_flux_norm_inside_segmask_array: array_like
            Normalized fluxes. Checked to remove those with the lowest fluxes 
            that are no use for the continuum.
        segment_begin_end: array_like
            The beginning and end wavelengths for each segment, as each segment
            needs its own continuum
        flagged_fluxes: array_like
            Current continuum and absorption lines flagged. We do NOT remove 
            any atomic lines.
    Returns                                                                     
    -------                                                                     
        flagged_fluxes: array_like
            Now with the low fluxes set to 0 so they are ignored and not used 
            in the continuum.
    """
    # Number of segments we have. Made each function to avoid potential length 
    # errors.
    number_of_segments = len(segment_begin_end)

    # Deselect 70% lowest continuum points in each segment using synthesis. 
    # Ensure both ends have continuum points
    for segment_band in range(0, number_of_segments):
        # The fraction of points we want to remove. Start at 70% and then 
        # adjust in the while loop
        fraction = 0.7
        # We take the wavelength indices of the wavelengths that exist in the 
        # current segment of the loop.
        wavelength_inside_segment_index = np.where(
            np.logical_and(ccd_wavelengths_inside_segmask_array 
                               >= segment_begin_end[segment_band, 0],
                           segment_begin_end[segment_band, 1] 
                               >= ccd_wavelengths_inside_segmask_array))

        # While the fraction to remove is not 0% we continue looping. We check 
        # that we have enough points for the continuum before cutting fluxes, 
        # and if we don't we reduce the fraction by 10%, hence if it hits 0% we
        # have to stop. Otherwise if we do, we set fraction to 0 to break the 
        # loop. Python is setting it to E-17 in loop so this fixes it by using 
        # 0.01 instead of == 0.
        while fraction > 0.01:
            # The value of the fraction of the fluxes we chose, so how many IS 
            # 70% for example.
            value_of_fraction = int(len(ccd_flux_norm_inside_segmask_array[
                                            wavelength_inside_segment_index]) 
                                            * fraction)

            # Takes the index of the 70%th lowest value (our cut off point) 
            # from a sorted list. Sorting takes a long time but this doesn't 
            # appear to be the bottleneck. @@@
            cutting_flux_value = sorted(
                ccd_flux_norm_inside_segmask_array[
                    wavelength_inside_segment_index])[value_of_fraction]

            # We take a list of indices where the flux in the segment is below
            # our cut off point. No longer sorted.
            cutting_flux_index = np.where(np.logical_and(
                ccd_flux_norm_inside_segmask_array < cutting_flux_value,
                np.logical_and(ccd_wavelengths_inside_segmask_array 
                                   >= segment_begin_end[segment_band, 0],
                               segment_begin_end[segment_band, 1] 
                                   >= ccd_wavelengths_inside_segmask_array)))

            # We need to count how many values there are at the extreme ends of
            # the segment, as we need a continuum on both sides. Here we see 
            # how many continuum points are left in total. Those considered 
            # also have to be flagged as 2 - so no absorption lines are 
            # involved here.
            saved_continuum_index = np.where(np.logical_and(
                ccd_flux_norm_inside_segmask_array >= cutting_flux_value,
                np.logical_and(ccd_wavelengths_inside_segmask_array 
                                   >= segment_begin_end[segment_band, 0],
                               segment_begin_end[segment_band, 1] 
                                   >= ccd_wavelengths_inside_segmask_array)))

            # Again the [0] to take the first element only, and ignore the 
            # returned dtype= parameter from np.where
            # This tells us how many values are in the top and bottom 2/3 and 
            # 1/3, we use indices here as the wavelengths are ordered so it 
            # works out easily as [133] > [132] for example. If the wavelengths 
            # are somehow NOT ordered, then we must change how to determine how
            # many are in the top and bottom 2/3 and 1/3.
            low_continuum_index = np.where(
                saved_continuum_index[0] 
                    <= wavelength_inside_segment_index[0][
                        int(len(wavelength_inside_segment_index[0]) / 3)])

            high_continuum_index = np.where(
                saved_continuum_index[0] 
                    >= wavelength_inside_segment_index[0][
                        int(len(wavelength_inside_segment_index[0]) * 2 / 3)])

            # If we don't have enough points, decrease the fraction we remove.
            if (len(low_continuum_index[0]) < 5 
                    or len(high_continuum_index[0]) < 5):
                fraction -= 0.1

            # If we have enough points on both sides, we can continue and 
            # remove them by looping through the indices of the low fluxes.
            else:
                for index_loop in cutting_flux_index[0]:
                    # Checks if it's 2, as we don't want to remove spectra 
                    # values that are at 1.
                    if flagged_fluxes[index_loop] == 2:
                        # print("2")
                        flagged_fluxes[index_loop] = 0
                # Now we break this loop as we have applied what we needed to.
                fraction = 0

    return flagged_fluxes


def removing_nlte(makestruct_dict, 
                  ccd_wavelengths_inside_segmask_array, 
                  ccd_flux_norm_inside_segmask_array,
                  flagged_fluxes):
    """
    Removes the lines close to the NLTE-dominated cores.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Information on the line cores, and stellar metallicity.
        ccd_wavelengths_inside_segmask_array: array_like
            Wavelength values of spectrum. Used to compare to the wavelengths
            of the line cores.
        ccd_flux_norm_inside_segmask_array: array_like
            Normalized fluxes. Check to see if they are below the preset 
            minimum if they are close to the line cores.
        flagged_fluxes: array_like
            Current continuum and absorption lines flagged. We do NOT remove 
            any atomic lines.
    Returns                                                                     
    -------                                                                     
        flagged_fluxes: array_like
            Now with low fluxes near the line cores removed.
    """
    # Avoid strong NLTE-dominated cores in mask if the flux is below a certain 
    # threshhold.
    core_minimum = 0.6
    if makestruct_dict['metallicity'] < -2:
        core_minimum = 0.72
    elif makestruct_dict['metallicity'] < -1:
        core_minimum = 0.65

    # Checks for line_cores existing from pysme_WEAVE, if no variable exists 
    # then we can't do this.
    try:
        print("Using the line cores:", makestruct_dict['line_cores'])
        # if it's near the value of the line core wavelength, and the flux is 
        # below a preset min, we're setting it to 0.
        for line_core_loop in range(0, len(makestruct_dict['line_cores'])):
            line_core_wavelength_index = np.where(
                np.logical_and(
                    abs(ccd_wavelengths_inside_segmask_array 
                        - makestruct_dict['line_cores'][line_core_loop]) < 4,
                    ccd_flux_norm_inside_segmask_array < core_minimum))

            flagged_fluxes[line_core_wavelength_index] = 0

    except AttributeError:
        print("No line_cores found while performing continuum normalization")

    return flagged_fluxes


def produce_indexed_atomic_data(makestruct_dict, segment_begin_end, 
                                linemask_data, sel_elem="all",
                                run="Synthesise"):
    """
    We produce an indexed version of the large amount of atomic data that is 
    contained in the files used, taking only the atomic lines that are either 
    in the segments only, or in the linemask depending on what the variable 
    'run' is set to. We also can select to cut the linelist down to only 
    include specific elements with the sel_elem parameter.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Balmer run, depth, and broadline data.
        segment_begin_end: array_like
            Used segmented loops to check for broadline data
        linemask_data: 
            Information on the wavelengths of the line_mask
        run: string
            We use linemask or segment mask depending on the run.
        sel_elem: string
            Keyword to select all or just a subset of elements. For example, if 
            set to "Fe" the linelist and linemask will be adjusted only to 
            include Fe lines, if set to "all" the linemask will not be changed

    Returns                                                                     
    -------                                                                     
        Many variables, all containing information on atomic lines such as 
        ionization or their species, or the depth observed.

        cut_line_list: pandas data_frame
            The .lin file containing all of the spectral lines in the 
            wavelength range
    """

    # Finds the files containing atomic data and creates arrays to hold them in
    # a format suitable for PySME
#    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, \
#            lower_level, upper_level, lu_lande, master_line_list = \
#        indexed_stellar_information(makestruct_dict)
    
    # Skip making all of the arrays since we are loading in a .lin file which
    # already has the data in the form that SME requires (i.e a dataframe)
    master_line_list, line_list_type = load_linelist(makestruct_dict)
    sme_linelist = master_line_list
    
#    line_atomic = master_line_list['wlcent']
    print("line_atomic", len(master_line_list['wlcent']))

    # If run is Solve, we want the indices of wavelengths in segments we 
    # already have, else we want them from the linemask ranges.
    if run == "Synthesise":
        print("Running synth here")
        segment_mask_index = \
                atomic_lines_in_segments(makestruct_dict, segment_begin_end, 
                                         master_line_list['wlcent'])

        # make sure there are no dups (there shouldn't be anyway)
        segment_mask_index = np.unique(segment_mask_index)

        # Save the indices for later so we can match the depth values from SME
        # after the first synth run to the appropriate lines in the line list
        # (Skip this for the balmer run)
        if not makestruct_dict['balmer_run']:
            makestruct_dict['segment_mask_index'] = segment_mask_index

        # convert segment mask indices to a boolean array so it can be applied 
        # to the line list
        segment_mask_bool = np.zeros(len(master_line_list), dtype=bool)
        segment_mask_bool[segment_mask_index] = True

#        print('Cutting line list with depthmin %s' \
#                %makestruct_dict['depthmin'])

#        cut_line_list = master_line_list[segment_mask_bool &
#                                            (master_line_list['depth'] 
#                                                > makestruct_dict['depthmin'])]

    # No depth cut in synth runs, only in solve runs. The synth run is used
    # to compute the depths of all of the lines
        cut_line_list = master_line_list[segment_mask_bool]

        #optionally add an extra cut to run synthesize on only lines of a 
        #certain element
        if sel_elem != "all":
            cut_line_list = \
                    cut_line_list[cut_line_list.species[:2] == sel_elem]

    else:
        print("Running", run)
        # apply the segment mask (computed during synth run), then cut on depth
        # and then make then compute and apply the line mask
        # to the line list
        segment_mask_bool = np.zeros(len(master_line_list), dtype=bool)
        segment_mask_bool[makestruct_dict['segment_mask_index']] = True

        # We can now cut at the updated minimum depth (0.1)
        print('Cutting line list with depthmin %s' \
                %makestruct_dict['depthmin'])

        try:
            sme_var = pickle.load(open(r'OUTPUT/VARIABLES/' \
                          + makestruct_dict['obs_name'] + \
                          '_SME_variables.pkl', 'rb'))
            print('Using depths computed from synth run')

        except IOError:
            print('SME output file not found: Using depths from master line \
                    list (not recommended)')
            depth = master_line_list['depth']

        # First apply the segment mask and the depth cut using the depths 
        # computed from the synth run
        seg_cut_line_list = master_line_list[segment_mask_bool]
        temp_line_list = seg_cut_line_list[sme_var['depth'] \
                                              > makestruct_dict['depthmin']]

          
        #Add an extra cut to run synthesize on only lines of a certain element
        if sel_elem != "all":
            cut_line_list = \
                    cut_line_list[cut_line_list.species[:2] == sel_elem]

            linemask_data = linemask_data[linemask_data.Atom == sel_elem]

        # Compute the lines contained within the line mask
        line_mask_index = \
                atomic_lines_in_linemask(makestruct_dict,
                                         temp_line_list['wlcent'], 
                                         linemask_data)

        # apply the line mask to the line list
        line_mask_bool = np.zeros(len(temp_line_list), dtype=bool)
        line_mask_bool[line_mask_index] = True
        cut_line_list = temp_line_list[line_mask_bool]

#        print('line_list_mask_index %s' %makestruct_dict['line_list_mask_index'])

#        # Update the line list mask in makestruct in case we want to manipulate
#        # the line list again later
#        makestruct_dict['line_list_mask_index'] = segment_mask_index

    print("Balmer run is: ", makestruct_dict['balmer_run'])
    # If we can't find ANY atomic lines in the segments, we need to return 
    # something to continue despite this failed balmer run.
    if len(cut_line_list) == 0:
        print("No atomic lines within chosen Segments. Likely during a Balmer \
                line run, moving on.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    nselect = len(cut_line_list)
    # Showing how many lines were not used. 
    print("%d unique spectral lines are selected within wavelength segments \
            out of %d" %(len(cut_line_list),len(master_line_list)))

    # We want to index the atomic data, and so apply it to line_atomic first, 
    # as we then order them according to wavelength, which we want to apply 
    # directly to the other atomic information to keep the information in the 
    # right order.
#    line_atomic = line_atomic[desired_atomic_lines_index]
#    # Now we also sort them according to wavelength.
#    sort_line_index = np.argsort(line_atomic[:])

    # So now we apply these indices to the information taken from smerdlin. As 
    # these indices were taken using line atomic then the position of the index 
    # should be fine.
#    species = species[desired_atomic_lines_index][sort_line_index]
#    line_atomic = line_atomic[sort_line_index]
#    lande_mean = lande_mean[desired_atomic_lines_index][sort_line_index]
#    depth = depth[desired_atomic_lines_index][sort_line_index]
#    data_reference_array = \
#            data_reference_array[desired_atomic_lines_index][sort_line_index]
#    lower_level = lower_level[desired_atomic_lines_index][sort_line_index]
#    upper_level = upper_level[desired_atomic_lines_index][sort_line_index]
#    j_e_array = j_e_array[desired_atomic_lines_index][sort_line_index]
#    lu_lande = lu_lande[desired_atomic_lines_index][sort_line_index]

    return cut_line_list, makestruct_dict


def indexed_stellar_information(makestruct_dict):
    """
    Produces the arrays with the stellar information from the master file for 
    every wavelength. such as line_atomic etc. Either loads a previously 
    modified linelist, or makes a modified linelist using the data we load from
    WEAVE in load_linelist, and then modifying various parameters within it 
    such as the species name to include ionization.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Information on the chosen line list, and whether it's already been 
            modified.
        segment_begin_end: array_like
            Used segmented loops to check for broadline data
        linemask_data: 
            Information on the wavelengths of the line_mask
        run: string
            We use linemask or segment mask depending on the run.
    Returns                                                                     
    -------                                                                     
        Many variables containing atomic information on the wavelengths in the 
        line list. Returned in the following format:
                    |Description|      |Content|                     |Key|
                atomic/data array:    Atomic number,           -> ['atomic']
                                      Ionic number
                                      Lambda
                                      E_Low
                                      Log_gf
                                      Rad_damp
                                      Stark_Damp
                                      Vdw_Damp
                Lande_Mean:           Lande_mean               -> lande
                Depth:                Depth                    -> depth
                Lu_Lande:             lande_lo,                -> line_lulande
                                      lande_up
                j_e_array:            j_low,                   -> line_extra
                                      e_up,
                                      j_up
                lower_level:          label_low                -> line_term_low
                upper_level:          label_up                 -> line_term_upp
                Data_reference_array: lambda_ref,              -> lineref
                                      e_low_ref,
                                      log_gf_ref,
                                      rad_damp_ref,
                                      stark_damp_ref,
                                      vdw_damp_ref,
                                      lande_ref


    """
#    # The new location for us to save and load our linelist.
#    makestruct_dict['line_list_location'] = r'OUTPUT/LINELIST/'
#    # We try to open the indexed linelist (with modifications made to logg, 
#    # etc) that is created if makestruct has been run before. We're checking to
#    # see if the original linelist+modified exists as that's what we create.
#    try:
#        # During the first run '_modified' isn't part of line_list so we need 
#        # to make sure we try opening it with _modified attached in case we 
#        # created it in a completely different run before, but still on this 
#        # same object, as we keep the linelist around after the run
#        if '_modified' in makestruct_dict['line_list']:
#            sme_linelist = pickle.load(open(
#                makestruct_dict['line_list_location'] +
#                makestruct_dict['line_list'], "rb"))
#        # If the linelist hasn't been modified before in THIS run (a.k.a first 
#        # Synthsize run), we double check to see if we made it in a run at 
#        # another time.
#        else:
#            # [:-5] removes the .fits part of the file name
#            sme_linelist = pickle.load(open(
#                makestruct_dict['line_list_location']
#                + makestruct_dict['line_list'].split('.')[0] 
#                + '_modified.csv', "rb"))
#            makestruct_dict['line_list'] = \
#                    makestruct_dict['line_list'].split('.')[0] 
#                    + '_modified.csv'
#
#        print("Line_merge data file found, this will be faster!")
#    # If we never ran this object before, we must modify the data in the line 
#    # list (that we have carried forward in master line list) to be used with 
#    # PySME. Modifications include adding ionization to species name and more.
#    # Because this takes a while, we then save it as a file to be used in later 
#    # runs if we ever want to analyse the object again, or any others in the 
#    # same data release/linelist
#    except FileNotFoundError:
#        print("No line_merge data file created previously. "
#              "\nRunning a line merger to modify the data to allow for pysme "
#              "input. \nThis could take several minutes, and will create a "
#              "data file for later use for any star in this data release.")
        # Loads the data from the linelist initially
#        master_line_list, line_list_type = load_linelist(makestruct_dict)

#
#
#        # This function modifies the linelist data and saves the output in the 
#        # OUTPUT/ directory
#        pysme_readlines.run_merger(master_line_list, line_list_type, 
#                                   makestruct_dict['line_list_location'],
#                                   makestruct_dict['line_list'])
#        # The linelist is now modified so we modify the name in makestruct
#        # to ensure we call this new one with an updated depth
#        makestruct_dict['line_list'] = \
#                makestruct_dict['line_list'].split('.')[0] + '_modified.csv'
#
#        # Now we know it exists, we can do what we were trying to before.
#        sme_linelist = pickle.load(open(makestruct_dict['line_list_location'] +
#                                        makestruct_dict['line_list'], "rb"))


    # We only interpolate non balmer depths, as otherise we have a depth of 
    # 32000 for an atomic list of 2 (Balmer run)
#    if not makestruct_dict['balmer_run']:
#        # Finds a file in GALAH with the same parameter information as ours. 
#        # However, our depth is not correct, as it is stored in THESE files, so 
#        # we open them to find the newly corrected depth. Repeated with all new
#        # parameters
#        sme_linelist['depth'] = \
#                pysme_interpolate_depth.reduce_depth(makestruct_dict)

#    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, \
#            lower_level, upper_level, lu_lande = \
#        sme_linelist['atomic'], sme_linelist['lande'], \
#        sme_linelist['depth'], sme_linelist['lineref'], \
#        sme_linelist['species'], sme_linelist['line_extra'], \
#        sme_linelist['line_term_low'], sme_linelist['line_term_upp'], \
#        sme_linelist['line_lulande']

#    master_line_list, line_list_type = load_linelist(makestruct_dict)
#    sme_linelist = master_line_list


    line_atomic, lande_mean, depth, data_reference_array, species, j_e_array, \
            lower_level, upper_level, lu_lande = \
        sme_linelist['wlcent'], sme_linelist['lande'], \
        sme_linelist['depth'], sme_linelist['reference'], \
        sme_linelist['species'], sme_linelist[['j_lo', 'e_upp', 'j_up']], \
        sme_linelist['term_lower'], sme_linelist['term_upper'], \
        sme_linelist[['lande_lower', 'lande_upper']]

    return line_atomic, lande_mean, depth, data_reference_array, species, \
            j_e_array, lower_level, upper_level, lu_lande, master_line_list


def load_linelist(makestruct_dict):
    """
    Loads in the linelist in either .lin format or .fits

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Information on the linelist file.
    Returns                                                                     
    -------                                                                     
        master_line_list: 
            Line list returned as either an hdulist or a vald format (readable
            by SME), for .fits
            and .lin input formats, respectively.
        line_list_type: string
    """
    print("Opening original line list.")
    if makestruct_dict['line_list'].split('.')[1] == 'fits':
        master_line_list = fits.open(makestruct_dict['original_location'] 
                                        + makestruct_dict['line_list'])[1]
        line_list_type = 'fits'

    elif makestruct_dict['line_list'].split('.')[1] == 'lin':

        linelist_str = makestruct_dict['original_location'] \
                                        + makestruct_dict['line_list']

        master_line_list=ValdFile(linelist_str)
        line_list_type = 'lin'

    else:
        
        print('Linelist is not in either of .fits or .lin format, cannot be \
                loaded')
        raise

    return master_line_list, line_list_type


#def desired_atomic_indices(makestruct_dict, modded_line_list):
#    """
#    Taking in the linelist to see which atomic lines lie within our segments. 
#    This is always the modified pickle file as it happens after readlines which 
#    creates a pickle file from the .lin file.
#
#    Parameters                                                                  
#    -------                                                                     
#        makestruct_dict: dictionary
#            Information on the segment mask file.
#    Returns                                                                     
#    -------                                                                     
#        all_lines_index: array_like
#            Indexes of the atomic lines that lie within our segments.
#    """
#    # Grabs segment wavelength information with updated resolutions
#    segment_mask_data_with_res = adjust_resolution(makestruct_dict)
#    # the list that we turn into an np array containing the indices of the 
#    # parts of the master file we care for. Apply it to sme rdlin to save time.
#    all_lines_index = []
#    # We're taking the desired lines in the linelist that are inside our 
#    # segment mask and above min depth or are Hydrogen( :, 2) is the lambda 
#    # (wavelength) which is set in readline. It was done that way in IDL so I 
#    # just had to copy it.
#    len_seg_mask = len(segment_mask_data_with_res['Wavelength_Start'])
#    for wavelength_band in range(0, len_seg_mask):
#        # Finds the lines in the master linelist that are inside our wavelength 
#        # start and end, or is hydrogen
#        single_line = np.where(np.logical_and(np.logical_and(
#            modded_line_list['atomic'][:, 2] >= float(segment_mask_data_with_res[
#                                                          'Wavelength_Start'][wavelength_band]),
#            modded_line_list['atomic'][:, 2] <= float(segment_mask_data_with_res[
#                                                          'Wavelength_End'][wavelength_band])),
#            np.logical_or(modded_line_list['depth'] > makestruct_dict['depthmin'],
#                          str(modded_line_list['species']) == 'H')))
#        # If there are no non broad lines, all_lines_index are just broad, else 
#        # combine the two. These are the INDICES but we turn it into the real 
#        # thing when creating the smaller linelist of obsname.fits for 
#        # makestruct all_lines_index is plines in idl
#        all_lines_index.extend(single_line[0])
#    print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGLLLLLLLLAAAAAAAAAAAAAAAAAAAAa')
#    print(all_lines_index)
#    print(len(all_lines_index))
#    raise
#
#    broad_line_index = []
#    if 'broad_lines' in makestruct_dict:
#        # list over numpy array to store the indices of the broad lines where they equal the linelist.
#        for broadline in makestruct_dict['broad_lines']:
#            broad_line_index.extend((np.where(broadline == modded_line_list['atomic'][:, 2]))[0])
#
#    # If we have broad lines in the local variable definitions we want to add them.
#    # Out of loop to prevent repeated adding of the same ones.
#    if 'broad_lines' in makestruct_dict:
#        # all lines is plines in idl. Contains the regular lines in the wavelength bands, and the broad ones
#        # that impact it but with peaks that are out of the range.
#        # So theoretically, it could try to concatenate b l i if it doesn't exist if the previous if statement
#        # is skipped, but it can't happen as they have the same if requirements, so what's the issue?
#        # np.sort to keep in numerical order.
#        all_lines_index.extend(broad_line_index)
#    # Avoid pesky duplicates of broad lines which we otherwise get.
#    all_lines_index = np.unique(np.asarray(all_lines_index))
#
#    return all_lines_index


def adjust_resolution(makestruct_dict):
    """
    Load the segmnent mask to grab the appropriate resolution information to be 
    able to add new resolutions for our created wavelength peaks.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: Information on the segment mask for resolution.

    Returns                                                                     
    -------                                                                     
        segment_mask_data_with_res: Segment information with new resolutions 
        for wavelength peaks.
    """
    segment_mask_data_with_res = pd.read_csv(
        makestruct_dict['line_list_location'] 
        + makestruct_dict['segment_mask'], 
            delim_whitespace=True, 
            header=None,
            names=["Wavelength_Start", "Wavelength_End", "Resolution", 
                       "comment", "overflow"],
            engine='python', 
            skipinitialspace=True, 
            comment=';',
            usecols=["Wavelength_Start", "Wavelength_End", "Resolution"])

    # Sort in ascending order of starting wavelength
    segment_mask_data_with_res.sort_values(by=['Wavelength_Start', 
                                                    'Wavelength_End'])

    # Now we get the resolution from the resolution map.
    # We make a list to avoid calling pandas data frame repeatedly,
    temporarywavelengthpeaklist = []
    for wavelength_band_resolution in range(0, 
            len(segment_mask_data_with_res['Resolution'])):
        # Calculating the peak wavelength
        temporarywavelengthpeaklist.append(
            0.5 * (float(segment_mask_data_with_res.loc[
                             wavelength_band_resolution, 'Wavelength_Start'])
                   + float(segment_mask_data_with_res.loc[
                               wavelength_band_resolution, 'Wavelength_End'])))
        # Interpolating the resolution at the wavelength peak and replacing the 
        # resolution of that index with it
        segment_mask_data_with_res.loc[
                wavelength_band_resolution, 'Resolution'] = \
                        makestruct_dict['interpolation'](
                            temporarywavelengthpeaklist[
                                wavelength_band_resolution]) \
                        / makestruct_dict['resolution_factor']

    return segment_mask_data_with_res


# Indexes the total atomic lines to those that fit within the wavelength 
# segments we have (which themselves were indexed to the segment mask file), 
# run during synthesize runs.
def atomic_lines_in_segments(makestruct_dict, segment_begin_end, line_atomic):
    """
    Taking in the linelist to see which atomic lines lie within our segments. 
    This is always the modified pickle file as it happens after readlines which 
    creates a pickle file from the fits file.

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            For depth minimum, and broad line information
        segment_begin_end: array_like
            To compare to atomic lines to find those within segments.
        line_atomic: array_like
            The central wavelengths of each atomic line in the line list.
        depth: 
            Compare to depth min to only select lines above a minimum depth
    Returns                                                                     
    -------                                                                     
        desired_atomic_lines_index: list 
            Indexed atomic data to limit to the wavelength segments.
    """

    # The list we put the indices of the esired atomic lines into
    desired_atomic_lines_index = []
    number_of_segments = len(segment_begin_end)
    buffer = 0.7
    # Index the atomic data for only ones in our wavelength segments, with a 
    # given buffer.
    for segment in range(0, number_of_segments):
        # Here we reference the data created in sme_rdlin, which took about 70 
        # seconds for the full 300k lines. We test to see which parts are in 
        # the segments we have in segment_begin_end which is segment mask.
        desired_atomic_lines_index.extend(np.where(
            (line_atomic[:] > (segment_begin_end[segment, 0] - buffer)) &
            (line_atomic[:] < (segment_begin_end[segment, 1] + buffer)))[0])

        # If broad lines are near (but not inside), select those too.
        for broad_line_single in makestruct_dict['broad_lines']:

            # If any line is within 100a of a broadline we'll include it
            if (np.any(abs(broad_line_single 
                    - segment_begin_end[segment]) < 100)):
                # Where does this broad line exist in our atomic line array? 
                # rounds automatically in np where for the array. We have 
                # duplicates in line_atomic and (therefore?) the d_a_l_index, 
                # do we want to remove those? We add all(?) the lineatomic to 
                # the index, but that makes sense as it is preselected to 
                # include the indices of the wavelengths inside our segments.
                desired_atomic_lines_index.extend(np.where(
                                    line_atomic[:] == broad_line_single)[0])

    return desired_atomic_lines_index


def atomic_lines_in_linemask(makestruct_dict, line_atomic, linemask_data):
    """
    Now we see which are in the line list, which is the list of important 
    wavelengths to look out for, run during Solve.
    This is if we aren't doing atomiclinesinsegments. Different to when we 
    indexed the wavelengths as that used seg_mask

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            For depth minimum, and broad line information
        linemask_data: 
            To compare to atomic lines to find those within the linemasks.
        line_atomic: 
            Find the atomic lines in the segments of segment_begin_end.
        depth: 
            Compare to depth min to only select lines above a minimum depth

    Returns                                                                     
    -------                                                                     
        desired_atomic_lines_index: list 
            Indexed atomic data to limit to the linemasks.
    """
    # the list we put the indices of the esired atomic lines into
    desired_atomic_lines_index = []

    # de in makestruct.pro. A buffer to fit AROUND The edges of the linemask
    buffer = 0.7

    # Used to take the certain number of linemask indices
    nrline = int(20 + ((8000 - makestruct_dict['Teff']) 
                     / 1.3E3) ** 4)
    # We see which parts of our atomic line data lies within the linemask 
    # (the important wavelengths) as well as identifying the absorption 
    # wavelengths if we can, and the broad lines. We get a lot of duplicates 
    # here
    for line in range(0, len(linemask_data['Sim_Wavelength_Peak'])):

        # We find the wavelengths within the ranges, but this time it's the 
        # linemask ranges, not segments.
        inside_linemask_index = np.where(
            (line_atomic[:] > 
                (linemask_data['Sim_Wavelength_Start'][line] - buffer)) &
            (line_atomic[:] < 
                (linemask_data['Sim_Wavelength_End'][line] + buffer)))[0]

        # We reverse it to take the the largest indices which also mean the 
        # highest wavelengths
        inside_linemask_index = np.flip(inside_linemask_index)

        # Take either the last nrline number or all the index, whichever is 
        # smaller.
        desired_atomic_lines_index.extend(
            inside_linemask_index[0:min(nrline,len(inside_linemask_index))])
#                (inside_linemask_index[
#                    0:min([nrline, len(inside_linemask_index)])])[0])

        # ;always select the main line if it's present
        peak_index = (np.where(line_atomic[:] == 
                        float(linemask_data['Sim_Wavelength_Peak'][line])))[0]

        if peak_index.size != 0:
            desired_atomic_lines_index.extend(peak_index)
        else:
            print("No peak line (",
                  linemask_data['Sim_Wavelength_Peak'][line],
                  ") available in the atomic line list for line", line,
                  "(", linemask_data['Sim_Wavelength_Start'][line], "to", 
                  linemask_data['Sim_Wavelength_End'][line],
                  ")")

        # And of course, always select broad lines when close just like before 
        # (but that was broad lines in segments)
        for broad_line_single in makestruct_dict['broad_lines']:
            # If any line is within 100a of a broadline we'll include it
            if np.any(
                    abs(broad_line_single 
                    - float(linemask_data['Sim_Wavelength_Peak'][line])) 
                        < 100):

                print("Broad line", broad_line_single, " at", 
                    linemask_data['Sim_Wavelength_Peak'][line])

                desired_atomic_lines_index.extend(
                        np.where(line_atomic[:] == broad_line_single)[0])

    return desired_atomic_lines_index


def set_atmosphere_string(atmo_grid_file, logg):
    """
    Strings to tell PYSME what kind of atmosphere we're looking for. Similar to 
    what was done in pysme_WEAVE but more specific atmosphere variables.

    Returns                                                                     
    -------                                                                     
        atmosphere_depth: string 
            Type of depth.
        atmosphere_interpolation: string
            Method of interpolation.
        atmosphere_geometry: string
            Geometry of model atmosphere used (options are 
            PP = Plane parallel and SPH = Spherical).
        atmosphere_method: string
            How to organise the atmosphere.
    """
    # Tau is the optical depth at some wavelength reference for continuum to 
    # determine the deepest point of interest. You can plot T against log tau 
    # and when x is 0, it is opaque so anything below that we are viewing.
    # Rhox is the other option of column mass (accumulated mass) of the 
    # atmosphere. These are both pretty bad as they cause some kind of spike in 
    # the abundance of stars at temperatures but people are working on 
    # replacing them with a neural network.

    atmosphere_depth = "RHOX"
    atmosphere_interpolation = "TAU"
    # Options are PP (Plane parallel) and SPH (Spherical)
    # If using Marcs model atmospheres, then PP is used for stars with 
    # -1 < logg < 3.5 and SPH is used for giants with 3.5 < logg < 5
    # See https://marcs.astro.uu.se/ "Model parameters" for more info
    if atmo_grid_file == 'marcs2012.sav' or atmo_grid_file == 'marcs2014.sav':
        # SPH for giants
#        if -1.0 < logg <= 3.5: 
        if -1.0 < logg <= 4: 
            atmosphere_geometry = "SPH"
        # PP for dwarfs
#        elif 3.5 < logg < 5.5:
        elif 4 < logg < 5.5:
            atmosphere_geometry = "PP"
    else:
        atmosphere_geometry = "PP"
    atmosphere_method = "grid"

    return atmosphere_depth, atmosphere_interpolation, atmosphere_geometry, \
            atmosphere_method


def dataframe_the_linelist(linelist):
    """
    Parameters                                                                  
    -------                                                                     
        Linelist: dictionary
            The atomic line data including species, ionization, and depth 
            and lande. Used to convert its data into a PySME format.
    Returns                                                                     
    -------                                                                     
        linedata: pandas data_frame 
            A slightly modified version of the linelist, both returned and 
            saved to an OUTPUT/LINELIST/ directory
    """
    data = {
        "species": linelist['species'],
        "atom_number": linelist['atomic'][:, 0],
        "ionization": linelist['atomic'][:, 1],
        "wlcent": linelist['atomic'][:, 2],
        "excit": linelist['atomic'][:, 3],
        "gflog": linelist['atomic'][:, 4],
        "gamrad": linelist['atomic'][:, 5],
        "gamqst": linelist['atomic'][:, 6],
        "gamvw": linelist['atomic'][:, 7],
        "lande": linelist['lande'],
        "depth": linelist['depth'],
        "reference": linelist['lineref']
    }

    # PySME doesn't import lineref as an array, so we do something to combine 
    # all values in the array each line to allow us to modify it in the same 
    # way
    linerefjoined = \
            np.array(([''.join(array) for array in linelist['lineref']]))

    """
    The errors in the VALD linelist are given as strings in various formats. 
    (e.g. “N D”, “E 0.05”). Parse_line_error takes those and puts them in the 
    same format, I.e. relative errors. Now at the moment at least, the relative 
    errors on the line data, do not influence the calculations in any way. So 
    if you can’t find them in your data, it won’t be a problem.
    """
    error = [s[0:11] for s in linerefjoined]  # .strip removed
    error = np.ones(len(error), dtype=float)
    # We set error to be a vague amount as it does not yet influence the 
    # calculation.
    error.fill(0.5)
    data["error"] = error
    data["lande_lower"] = linelist['line_lulande'][:, 0]
    data["lande_upper"] = linelist['line_lulande'][:, 1]
    data["j_lo"] = linelist['line_extra'][:, 0]
    data["e_upp"] = linelist['line_extra'][:, 1]
    data["j_up"] = linelist['line_extra'][:, 2]
    data["term_lower"] = [t[10:].strip() for t in linelist['line_term_low']]
    data["term_upper"] = [t[10:].strip() for t in linelist['line_term_upp']]

    # We have an issue with lineref being 2d. Needs to be 1d. But it contains 
    # all 7 arrays from IDL so instead we input the previously mentioned 
    # linerefjoined to conglomorate all the arrays into one row per row.
    data['reference'] = linerefjoined
    linedata = pd.DataFrame.from_dict(data)
    linedata.to_pickle("OUTPUT/LINELIST/pysme_linelist_dataframe")

    return linedata


def store_sme_input(makestruct_dict):
    """
    IDL saves it as an inp file, but we don't like that. This is a copy of what 
    IDL does and where it saves it, called from make_struct function. We input 
    it directly by importing sme,

    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary
            Needed for name of object, and to save all.
    Returns                                                                     
    -------                                                                     
        None: 
            But the original SME input file is saved to the OUTPUT/ folder.
    """
    print("Saving the SME input to OUTPUT/full_OUTPUT/",
          makestruct_dict['obs_name'], "_Original_SME_input.pkl'")
    input_file = open(r'OUTPUT/full_output_' + makestruct_dict['obs_name'] 
                          + '_Original_SME_input.pkl', 'wb')

    pickle.dump(makestruct_dict, input_file)
    input_file.close()


def segment_arrays(wavelengths, flux, flux_error, 
                       flagged_flux, total_segments):
    """
    A function to take the wavelength etc arrays that are just a single array, 
    and turn it into a list of arrays seperated per segment, as required from 
    pysme.

    Parameters                                                                  
    -------                                                                     
        Non segmented arrays:
            wavelengths: array_like
            flux: array_like
            flux_error: array_like
            flagged_flux: array_like
            total_segments: array_like
    Returns                                                                     
    -------                                                                     
        The same arrays but segmented. Each list has many arrays inside of 
        it, each unique to our segment_mask choices
    """
    # Create lists to store the arrays of each type in.
    pysme_wave_list, pysme_flux_list, \
            pysme_error_list, pysme_flagged_flux_list = ([], [], [], [])
    # Run through each segment and find the indices of wavelengths inside it.
    for segment in total_segments:
        wavelength_segment_indices = \
                (wavelengths >= segment[0]) & (wavelengths <= segment[1])
        # Then apply the indices to wavelength arrays, and the other flux etc.
        pysme_wave_list.append(wavelengths[wavelength_segment_indices])
        pysme_flux_list.append(flux[wavelength_segment_indices])
        pysme_error_list.append(flux_error[wavelength_segment_indices])
        pysme_flagged_flux_list.append(
                flagged_flux[wavelength_segment_indices])

    return pysme_wave_list, pysme_flux_list, \
                pysme_error_list, pysme_flagged_flux_list
