"""We run pysme galah1-4, makestruct, and run_sme to run PySME. We use only 
data collected in Galah folders, and output into OUTPUT/ where we house the 
spectra, variables, and any other files created in the mean time. This file is 
mainly a collection of all other files. Everything involved is titled pysme_x

We use Galah data and, after modification, use Pysme to produce a synthetic 
spectra based on our variables. Then we modify those variables -or Free 
Parameters- and model the spectra again. When ours matches the observed spectra 
to a reasonable degree, we have accurate estimates on things such as abundance, 
temperature, gravity, and more."""

import sys 
import os
import time
import pickle

import pysme_WEAVE
#import test_pysme_WEAVE
import pysme_galah4
import params

# start timer
t_start = time.time()

## change location of log file for debugging
#from SME import util
#util.start_logging("log_file_pysme.log")

# Re-route standard in and standard out to a file. This will create log file 
# for output and delete old one first if it already exists
if params.print_to == 'out_file':
    sys.stdout = open("OUTPUT/OUTPUT_LOGS/output_%s.txt" %sys.argv[1], 'w')
    sys.stderr = open("OUTPUT/OUTPUT_LOGS/output_%s.txt" %sys.argv[1], 'w')

"""
We run each pysme_WEAVE which contains all of the data processing functions
then run file from pysme_x to make the pysme calls.

Step 1: Input names for atmosphere files, object number, etc. Some can be user 
input. Specifically the desired stellar object.
Step 2: Take the stellar spectra and adjust it for potential error in the 
atmosphere.
Step 3: Get starting guesses for variables such as Teff for the object."""


class read_iso:

    def __init__(self, age):
        self.num_cols=4
        self.columns = ['M_Mo', 'logTeff', 'logG', 'logL_Lo']
        self.num_ages = len(age)
        self.ages = age

    def fill_chemistry(self, m_h, fe_h, alpha_fe):
        self.FeH = fe_h
        self.Z = 10**m_h*0.0152
        self.aFe = alpha_fe

    def fill_iso(self, iso_input):
        self.data = iso_input


def collect_data():

    """
    Output: Makestruct dict: a dictionary that contains most variables we need 
    to modify our spectra and run sme

            Reduction_variable_dict: A less useful dictionary but still 
            contains some useful variables. These come directly from a WEAVE 
            data file and are not modified.
    """

    print('index %s' %sys.argv[1])

    # Use the standard parameters and the solar spectrum to test the code
    if (sys.argv[1] == 'sun') | (sys.argv[1] == 'arcturus'):
        #open pre-saved makestruct with solar standard params
        makestruct_dict = pickle.load(open(
                            'DATA/standard_stars/%s_makestruct_dict.pkl' 
                             %sys.argv[1], 'rb'))
#        reduction_variable_dict = np.load('red_var_dict_%s.pkl' %sys.argv[1])
        reduction_variable_dict = []
        makestruct_dict['field_for_obs'] = params.field_for_obs
        makestruct_dict['field_end'] = params.field_end
        makestruct_dict['atmosphere_grid_file'] = params.atmosphere_grid_file
        makestruct_dict['iterations'] = params.iterations
        makestruct_dict['line_list'] = params.line_list
        makestruct_dict['segment_mask'] = params.segment_mask
        makestruct_dict['line_mask'] = params.line_mask
        makestruct_dict['cont_mask'] = params.cont_mask
        makestruct_dict['vrad_flag'] = params.vrad_flag

        return makestruct_dict, reduction_variable_dict

    # Initial set up. We return the object name
    obs_name, obs_file, obj_for_obs, field_end, setup_for_obs, field_for_obs,\
            wav_segment, iterations, data_file_B, data_file_G \
                = pysme_WEAVE.setup_object(sys.argv[1])

    # Setting up of which atmosphere files to use in pysme. Returns primarily 
    # strings or lists.
    atmosphere_grid_file, line_list, atomic_abundances_free_parameters, \
        atmosphere_abundance_grid = pysme_WEAVE.set_atmosphere()

    # Collects a few important variables. Primarily line locations, and the 
    # minimum depth for atomic lines.
    broad_lines, depthmin, line_cores, vrad_flag = pysme_WEAVE.sme_variables()

    print('part_1_done')

    """Part 2: Read in spectra + resolution + interpolation to add resolution 
    and future wavelengths we decide + correct telluric & skyline error 
    (atmospheric errors)"""

    # Opens galah ccd data files to produce the light spectra graphs with 
    # wavelength, flux, and error as outputs.
    spec_wav, spec_flux, spec_flux_err, head_file, head_ind \
                = pysme_WEAVE.get_wav_flux(obj_for_obs, wav_segment,
                            data_file_B, data_file_G)

#"""likely don't need this"""
    # Opens a large data file to index it down to our given object only, 
    # containing data such as macroturbulent velocity
#    reduction_and_analysis_data = \
#        pysme_WEAVE.data_release_index(obj_for_obs)

#"""not sure if we need this"""
    # We use our spectra data so far and correct for telluric and skyline 
    # error, returning information in a cleaner
    # dictionary form. We also require barycentric velocity of the object.
#    ccd_data_dict = \
#        pysme_WEAVE.error_correction(reduction_and_analysis_data['v_bary'], \
#        spec_wav, spec_flux, \
#        spec_flux_err)

    
    print('part_2_done')

    """Part 3) Determine initial stellar parameters 3.1) Based on APS 
               paramters from initial WEAVE analysis
          3.2) If run with field_end LBOL, update initial params
          3.3) Ensure reasonable parameters"""

    # We take either cannon or GUESS data for the initial first choices of 
    # starting variables for the star, depending on quality and existance.
    starting_params = \
        pysme_WEAVE.get_starting_params(data_file_G, obj_for_obs)

    # convert relative error to absolute error
    # check that we actually need to do this (what is IVAR?) is it just %error?
#    total_ccd_flux_error_uob = spec_flux_err * spec_flux

    # apply doppler shift to input spectrum
    spec_wav_doppler = pysme_WEAVE.doppler_wavelengths(spec_wav, 
                                                starting_params['vrad_global'])

    ccd_data_dict = {"wave": spec_wav_doppler, "flux": spec_flux,
                     "error": spec_flux_err}

    # Produces interpolation to produce resolutions for different wavelengths 
    # we produce during the run
    interpolation, res_type = \
        pysme_WEAVE.get_resolution(obj_for_obs, head_file, spec_wav_doppler,
                                      head_ind)

    # Performs a lot of small variable modifications, such as j_mag and 
    # parallax. Used primarily in pysme_update_logg.
#    reduction_variable_dict = \
#        pysme_WEAVE.update_mode_info(field_end, field_for_obs, \
#        reduction_and_analysis_data)

#    reduction_variable_dict = \
#        pysme_WEAVE.update_mode_info(field_end, field_for_obs, \
#        reduction_and_analysis_data)

    reduction_variable_dict = pysme_WEAVE.get_gaia_params(head_file, head_ind)

    # We adjust the velocities and grav. for these types of runs only.
    # Here is where we run pysme_update_logg initially and is the last update 
    # before running makestruct unless it's unreasonable, in which case it'll 
    # be modified in th reasonable_parameters.
#    if field_end == 'lbol':
#        starting_params = \
#            pysme_WEAVE.update_gravity_function(
#                reduction_variable_dict, field_end, starting_params)
#
#    # leave logg as a free parameter to be solved by pySME
#    elif field_end == 'fixed':
#       pass 

    # We check to see if our first guess parameters are within reasonable 
    # limits. If not, we adjust them slightly fit. We adjust gravity if the 
    # temperature is out of bounds.
    starting_params = \
        pysme_WEAVE.reasonable_parameters(atmosphere_grid_file, \
        starting_params)


    # gam6 in the dictionary is a global correction factor to all van der 
    # Waals damping constants. Values of 1.5- 2.5 are sometimes used for iron

    # Setting up the dictionary we're going to input a lot of variables we want 
    # for makestruct, which in turn uses them for PySME. Setting it here as 
    # it's quite important and helps readability.

    makestruct_dict = {'setup_for_obs':          setup_for_obs,
                       'res_type':               res_type,
                       'obs_name':               obs_name,
                       'obj_for_obs':            obj_for_obs,
                       'field_for_obs':          field_for_obs,
                       'global_free_parameters': [],
                       'broad_lines':            broad_lines,
                       'unnormalised_spectra':   ccd_data_dict,
                       'atmosphere_grid_file':   atmosphere_grid_file,
                       'iterations':             iterations,
                       'depthmin':               depthmin,
                       'line_list':              line_list,
                       'balmer_run':             False,
                       'line_cores':             line_cores,
                       'normalise_flag':         False,
                       'vrad_flag':              vrad_flag,
                       'field_end':              field_end,
                       'original_location':      'DATA/LINELIST/',
                       'load_file':              False,
                       'gam6':                   1,
                       'segment_mask':           params.segment_mask, 
                       'original_line_list':     line_list,              
                       'nlte_abund':             atmosphere_abundance_grid,
                       'interpolation':          interpolation,
                       'line_list_location':     'DATA/LINELIST/',
                       'line_mask':              'DATA/LINELIST/' \
                                                 + params.line_mask,
                       'cont_mask':              'DATA/LINELIST/' \
                                                 + params.cont_mask,
                       'atomic_abundances_free_parameters':
                                             atomic_abundances_free_parameters
                       }


    # Now we add the starting parameters of temperature, gravity, metallicity, 
    # and velocities to the makestruct input
    makestruct_dict.update(starting_params.items())
    print("Starting variables:"
          "\nTemperature:",     makestruct_dict['Teff'],
          "\nLog_g:",           makestruct_dict['gravity'],
          "\nMetallicity:",     makestruct_dict['metallicity'],
          "\nRadial Velocity:", makestruct_dict['vrad_global'])

    """Part  4) Optimisation of stellar parameters (loop with alternating 4.1 
                    and 4.2)
             4.1) Segment selection and normalisation (fixed parameters)
             4.2) Iterative optimisation (fixed segments and normalisation)."""
    
#    import pickle
#    f1 = open('%s_makestruct_dict.pkl' %obj_for_obs, 'wb')
#    pickle.dump(makestruct_dict, f1)
#    f1.close()
#    f2 = open('%s_reduction_variable_dict.pkl' %obj_for_obs, 'wb')
#    pickle.dump(reduction_variable_dict, f2)
#    f2.close()
#    raise

    return makestruct_dict, reduction_variable_dict


# We run all code, and perform the total sme run. Outputs are stored in OUTPUT/ 
# where we can find the produced spectra
# and produced parameters such as temperature for the star chosen.

def execute():

    # Runs collect_data to produce a dictionary full of starting values to 
    # begin our spectra creation
    makestruct_dict, reduction_variable_dict = collect_data()
    # Iteratively runs PySME to produce the spectra or paramters, and modifies 
    # the output slightly to reach a higher accuracy.
    print(makestruct_dict)
    raise
    pysme_galah4.iterative_loop(makestruct_dict, reduction_variable_dict)

# Only run execute if this file is called directly, otherwise only run the
# collect_data() function.
if __name__ == "__main__":
    execute()
    print(time.time() - t_start)

if params.print_to == 'out_file':
    sys.stdout.close()
    sys.stderr.close()
