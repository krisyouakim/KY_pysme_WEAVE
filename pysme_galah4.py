"""Galah 4 primarily is in charge of determining the type of sme run (solve or 
synth, for calculating parameters or adjusting spectra respectively) and 
preparing the run for that, as well as modifications for balmer runs which 
require a more dedicated approach. It runs these a number of times according to 
the iterations list in the input makestruct dictionary."""

import numpy as np
import pysme_abund
import pysme_makestruct
import pysme_balmer_mask
import pysme_update_logg
import pandas as pd
import pickle
import os


# We repeat a number of times according to the length of iteration list, 
# whereas the number IN the list (e.g iter[x]) is the number of times Pysme 
# runs itself internally.
def iterative_loop(makestruct_dict, reduction_variable_dict):
    """                                                                         
    Updates the accuracies (resolutions) of the specta required depending on    
    whether it's a balmer run or not. Also where we set the base accuracy.      
                                                                                  
    Parameters                                                                  
    -------                                                                     
        makestruct_dict: dictionary                                             
            Dictionary created in pysme_execute containing necessary 
            information such as the stellar parameters and properties of the 
            spectrum.
    Returns                                                                     
    -------                                                                     
        Nothing directly but will end with spectra and variable files 
        containing the new stellar information.
    """
    # Determining whether it is a synthesize or solve run.
#    for optimisation_loop in range(0, len(makestruct_dict['iterations'])):
    for optimisation_loop in range(0, len(makestruct_dict['iterations'])):
        print('################## iterations is %s ######################' %makestruct_dict['iterations'])
        print('################## optimization loop is %s ######################' %optimisation_loop)
        # The FIRST run ever must normalise our data correctly to get a flat 
        # continuum at flux = 1
        if optimisation_loop == 0:
            print("STARTING LOOP 1 -> Pre-Normalisation")
        if optimisation_loop == 1:
            print("STARTING LOOP 2 -> First iterations (max. 2 iterations)")
        if optimisation_loop == 2:
            print("STARTING LOOP 3 -> Normalization")
        # There is an opportunity to run more than the basic 4 times.
        if optimisation_loop == 3 and len(makestruct_dict['iterations']) == 6:
            print("STARTING LOOP 4 -> Second iterations (max. 20 iterations)")
        if optimisation_loop == 3 and len(makestruct_dict['iterations']) == 4:
            print("STARTING LOOP 4 -> Final iterations (max. 20 iterations)")
        if optimisation_loop == 4:
            print("STARTING LOOP 5 -> Normalization")
        if optimisation_loop == 5:
            print("STARTING LOOP 6 -> Final iterations (max. 20 iterations)")
        makestruct_dict['current_iterations'] = \
                makestruct_dict['iterations'][optimisation_loop]
        print('################## current iterations is %s ######################' %makestruct_dict['current_iterations'])
        # Define initial abundances, initialising the MARCS model abundances. 
        # Only occurs on the first run.
        if optimisation_loop == 0:
            # Slight adjustment to starting metalicity for some stars.
            makestruct_dict['abundances'] = element_abundance()
            if makestruct_dict['gravity'] < 2:
                makestruct_dict['abundances'][6] += 0.4

            # Prenormalise for initial loop only.
            makestruct_dict['normalise_flag'] = True
        # If optimisation loop is not the first, set to 0 instead, and load the 
        # variables we created in pysme_run_sme during the previous run.
        else:
            # This is where we load the variable file, from this function which 
            # also sets the variables in the dict automatically.
            makestruct_dict = update_makestruct_variables(
                                    makestruct_dict, reduction_variable_dict)
            makestruct_dict['normalise_flag'] = False

        # This assumes that we only want to synthesise once in a row. While 
        # this holds, this if statement declares that we are in a synthesise 
        # run.
        if makestruct_dict['iterations'][optimisation_loop] <= 1:
            # The continuum is adjusted in this run, as we are modifying the 
            # spectra.
            makestruct_dict['continuum_scale_flag'] = "linear"
            makestruct_dict['continuum_scale_type'] = 'mask'
            # For synth runs we don't want any free parameters, and they would 
            # otherwise carry over from the solve runs.
            makestruct_dict['global_free_parameters'] = []
            # So we now run makestruct and pysme with a synthesise run.
            makestruct_dict = \
                    pysme_makestruct.create_structure(
                            makestruct_dict, 
                            reduction_variable_dict,
                            normalise_flag=makestruct_dict['normalise_flag'],
                            run="Synthesise")
            # Now we run a synthesise run but only for balmer lines. We do not 
            # want to adjust the spectra here so we change cscale information. 
            # This actually loads identical information that the previous 
            # synthesise run used, as we save the input data used previously to 
            # repeat here.
#            makestruct_dict['load_file'] = True
#            makestruct_dict['normalise_flag'] = False
#            makestruct_dict['continuum_scale_flag'] = 'fix'
#            makestruct_dict['continuum_scale_flag'] = "linear"
            makestruct_dict = balmer_makestruct(makestruct_dict, 
                                                reduction_variable_dict, 
                                                run="Synthesise")

            makestruct_dict['depthmin'] = 0.1
            print('depthmin changed to %.1f' %makestruct_dict['depthmin'])
            # Tells makestruct to load the file that sme has now produced for 
            # spectra and variables. setting after so we mimic the exact input 
            # of the prev. normalise run on the first one.
            makestruct_dict['load_file'] = True

        # Else if we plan on running pysme more than once (iter[x], it will be a Solve run.
        else:

            # We only modify parameters here, not the spectra.
#            makestruct_dict['continuum_scale_flag'] = "fix"
            makestruct_dict['continuum_scale_flag'] = "none"
            makestruct_dict['continuum_scale_type'] = 'mask'

            # Now we have different free parameters for pysme to use depending 
            # on what our file is called.
            makestruct_dict = \
                determine_free_parameters(makestruct_dict, 
                                          reduction_variable_dict)

            print("Starting Pysme Optimisation")
            makestruct_dict = \
                    pysme_makestruct.create_structure(makestruct_dict, 
                                                      reduction_variable_dict,
                                                      run="Solve")

    # The temporary file for the balmer input is not needed and is just quite 
    # confusing. It was confusing to make too, and for sure is NOT good code.

    # Rename output files to specify that they converged 
    # (add "conv_" to beginning of file name)
    os.rename(r"OUTPUT/SPECTRA/" + makestruct_dict['obs_name'] 
              + "_SME_spectra.pkl",       
                    r"OUTPUT/SPECTRA/conv_" + makestruct_dict['obs_name'] 
                    + "_SME_spectra.pkl")       
    os.rename(r"OUTPUT/VARIABLES/" + makestruct_dict['obs_name'] 
              + "_SME_variables.pkl",   
                    r"OUTPUT/VARIABLES/conv_" + makestruct_dict['obs_name'] 
                    + "_SME_variables.pkl")   

    # Remove all extra output files to clean up the output directories
    os.remove("OUTPUT/SPECTRA/Temp_Balmer_spectra_input_%s.pkl" 
                    %makestruct_dict['obj_for_obs'])
    os.remove(r"OUTPUT/SPECTRA/" + makestruct_dict['obs_name'] 
                    + "_Balmer_spectra.pkl")    
    os.remove(r"OUTPUT/VARIABLES/" + makestruct_dict['obs_name']                           
                    + "_Balmer_variables.pkl")                          
    os.remove(r'OUTPUT/full_output_' + makestruct_dict['obs_name']                         
                            + '_Original_SME_input.pkl')

    print("pysme_exec finished, output spectra in \
            SPECTRA/conv_%s_SME_spectra.pkl or .sme, whichever is prefered.\n \
            All other variables such as teff are in \
            OUTPUT/conv_%s_SME_variables.pkl or .sme, whichever is prefered." 
                %(makestruct_dict['obs_name'], makestruct_dict['obs_name']))


def update_makestruct_variables(makestruct_dict, reduction_variable_dict):
    """
    We update the variables of the dictionary from the file we produce in 
    pysme_run_sme
    Parameters                                                                  
    -------                                                                     
        Makestruct_dict: dictionary
            Used to load the new variables into, and check if we want to load 
            a file.
        Reduction_variable_dict: dictionary
            Contains variables for updating log_g
    Returns                                                                     
    -------                                                                     
        Makestruct_dict: dictionary
            With new parameters set such as Teff and Logg
    """
    # First we load the variables from either pysme_WEAVE.py or, if we've run 
    # SME before in this run, we get them from a saved file. We use this 
    # variable to check whether we want to load a file or use the premade 
    # starting info from pysme_WEAVE.py.
    if makestruct_dict['load_file']:
        stellar_variable_data = \
            pickle.load(open("OUTPUT/VARIABLES/" + makestruct_dict['obs_name'] 
                                                 + "_SME_variables.pkl", 
                                                    "rb"))

        print("Using SME updated variables.")
        # We now load them into the dictionary.
        makestruct_dict['effective_temperature'] = \
                stellar_variable_data["effective_temperature"]
        makestruct_dict['gravity'] = stellar_variable_data["gravity"]
        # We now want to update gravity using our own method, rather than 
        # having it as a free parameter.
        print("Starting logg update with teff, logg, feh of:", 
                makestruct_dict['effective_temperature'],
                makestruct_dict['gravity'], makestruct_dict['metallicity'])

        if ((makestruct_dict['field_end'] == 'seis') or 
                (makestruct_dict['field_end'] == 'lbol')):
                    makestruct_dict['gravity'] = \
                    pysme_update_logg.optimize_logg_with_extra_input(
                            makestruct_dict, reduction_variable_dict,
                            makestruct_dict["global_free_parameters"])

        makestruct_dict['metallicity'] = stellar_variable_data["metallicity"]
        makestruct_dict['radial_velocity_global'] = \
                stellar_variable_data["radial_velocity_global"]
        makestruct_dict['abundances'] = stellar_variable_data["abundances"]
        makestruct_dict['microturbulence_velocity'] = \
                stellar_variable_data["microturbulence_velocity"]
        makestruct_dict['macroturbulence_velocity'] = \
                stellar_variable_data["macroturbulence_velocity"]
        makestruct_dict['rotational_velocity'] = \
                stellar_variable_data["rotational_velocity"]
        print("ending logg update with teff, logg, feh of:", 
                    makestruct_dict['effective_temperature'],
                    makestruct_dict['gravity'], 
                    makestruct_dict['metallicity'])
    else:
        print("Using initial variables.")

    return makestruct_dict


# We must run synth with only balmer lines and a special balmer linelist. 
# This function does that by changing the name of the files we open to the 
# Hydrogen ones, and then modifies the line list to include these lines after 
# we find them by running pysme_balmer. We need to synthesize the Balmer lines 
# only in order to compute the regions where these broad lines are blend-free
# (and label those pixels as mask=1). This is not known a priori since the 
# Balmer line strength and amount of metal-blends vary a lot between different 
# stars. """
def balmer_makestruct(makestruct_dict, reduction_variable_dict, 
                          run="Synthesise"):
    """
    Parameters                                                                  
    -------                                                                     
        Makestruct_dict: dictionary
            Information such as spectra and parameters of teff...
        Reduction_variable_dict: dictionary
            Stellar parameters for update_logg
        Run: 
            Determines whether we're running solve or synthesize runs.
    """
    # Balmer runs don't save the data and variables as the normal type so we 
    # don't replace them.
    makestruct_dict['balmer_run'] = True

    # The non-balmer linelist for us to reset [line_list] to after we're done 
    # with galah_H.fits
    backup_linelist = makestruct_dict['line_list']
    # Using a different line list for these lines, that's the point of this.
    makestruct_dict['line_list'] = 'balmer_hfs.lin'

    print("Starting Balmer PySME run.")
    # Load these in case we encounter a bug and we over-write the originals 
    # with the balmer pysme run.
    full_sme_spectra = pickle.load(open(r"OUTPUT/SPECTRA/" + 
            makestruct_dict["obs_name"] + "_SME_spectra.pkl", "rb"))
    full_sme_variables = pickle.load(
        open(r"OUTPUT/VARIABLES/" + makestruct_dict["obs_name"] 
                                    + "_SME_variables.pkl", "rb"))

    # Now we're running with our new linelist information.
    makestruct_dict = pysme_makestruct.create_structure(
                            makestruct_dict, 
                            reduction_variable_dict,
                            normalise_flag=makestruct_dict['normalise_flag'],
                            run=run)

    # The center lines for balmer lines. We want to find their start and end 
    # point, and also the tef, grav, and feh. They have been added to broad 
    # lines previously in pysme_WEAVE.
    balmer0 = np.array((4861.3230, 6562.7970))

    # we set it to false if we have to break out (likely due to linelistatomic) 
    # so we don't break the whole run.
    if makestruct_dict['balmer_run']:

        balmer_spectra = \
            pickle.load(open('OUTPUT/SPECTRA/' + makestruct_dict["obs_name"] 
                                               + '_Balmer_spectra.pkl', 
                                                   "rb"))
        a=0
        for x in balmer_spectra['wave']:
            a += len(x)
        print("Before b, len of wave is", a)
        balmer0, balmer_st, balmer_en = \
            pysme_balmer_mask.balmer_mask(balmer0, full_sme_spectra, 
                                          full_sme_variables, balmer_spectra)

        # Only do it if both balmer lines are in segments. We change the 
        # segment mask name here if we create a
        # new one to one unique to the object.
        if len(balmer0) >= 1:
            # Modifies the linemask with the additional balmer lines.
            append_balmers(makestruct_dict, balmer0, balmer_st, balmer_en)

    # Resetting back to normal line list for further runs.
    makestruct_dict['line_list'] = backup_linelist
    # Not balmer run any more so we can overwrite the files with sme output 
    # during the next run
    makestruct_dict['balmer_run'] = False
    return makestruct_dict


# Dedicated function to add the created balmer lines to the linemask for pysme 
# to recognise as "important" lines. We load the linemask data frame, convert 
# to array, add the new linemask data, then convert back to dataframe (Pandas)
def append_balmers(makestruct_dict, balmer0, balmer_st, balmer_en):
    """Input:
        makestruct_dict: contains names of files needed
        balmer0: Center of the balmer line
        balmer_st: Start of the balmer line
        balmer_en: End of the balmer line"""
    # We load the large data release of dr2  and then produce one specific to 
    # the star with the additional balmer lines. If we made one before, we try 
    # to load that first.
    try:
        linemask_data = pd.read_csv(
            makestruct_dict['line_mask'], delim_whitespace=True, header=None, 
            engine='python', comment=';',
            names=["Sim_Wavelength_Peak", 
                "Sim_Wavelength_Start", "Sim_Wavelength_End", "Atomic_Number"])
    except KeyError:
        linemask_data = pd.read_csv(
#            "GALAH/DATA/" + makestruct_dict['setup_for_obs'] + '_Sp.dat', 
#                    delim_whitespace=True, header=None,
            "DATA/LINELIST/" + makestruct_dict['line_mask'] + '_Sp.dat', 
                    delim_whitespace=True, header=None,
            engine='python', comment=';',
            names=["Sim_Wavelength_Peak", "Sim_Wavelength_Start", 
                        "Sim_Wavelength_End", "Atomic_Number"])

    # An array for atomic number, hich is unimportant so we just have 1s. We 
    # take our atomic data from the linelists themselves, not our line masks.
    balmer_atom = np.ones(len(balmer0))
    # Dictionary to contain the soon to be dataframe information.
    extra = {}
    # Extend lists, then convert to np.array
    temp_data_list = []
    # For each column in the linemask, it has its own dictionary key with the 
    # data transfered.
    for column in linemask_data:
        # For each data value in the column. (e.g row)
        for data_value in linemask_data[column]:
            temp_data_list.append(data_value)
        extra[column] = temp_data_list
        temp_data_list = []
    extra['Sim_Wavelength_Peak'].extend(balmer0)
    extra['Sim_Wavelength_Start'].extend(balmer_st)
    extra['Sim_Wavelength_End'].extend(balmer_en)
    extra['Atomic_Number'].extend(balmer_atom)
    # Convert from dictionary to pandas dataframe.
    linemask_data = pd.DataFrame.from_dict(extra)

    # Line masks for log(g) sensitive lines, i.e. Fe,Ti,Sc. Changing to the 
    # obs_name, to make an individual new line mask for each star rather than 
    # the entire datarealse dr2 or 3 that we had originally for the first run.
    makestruct_dict['line_mask'] = r"OUTPUT/LINELIST/" \
                + makestruct_dict['obs_name'] + ".dat"
    # Remove the titles before saving
    linemask_data.columns = linemask_data.iloc[0]

    # And now save the data frame to be loaded later. If desired, could also 
    # return it and carry it forward, reducing the need for line lists being s
    # saved.
    pd.DataFrame.to_csv(linemask_data, makestruct_dict['line_mask'], 
            header=False, index=False, sep=" ")


# We read the text from the input field to determine what type of run we want 
# (lbol/seis). Only run during solve/parameter optimisation runs and not 
# synth/normalisation runs.
def determine_free_parameters(makestruct_dict, reduction_variable_dict):
    """
    Input:
        makestruct_dict: 
            Information on whether lbol or seis, and adjusts the 
            free parameters
        reduction_variable_dict: 
            Update logg variables to print out.
    """
    if makestruct_dict['field_end'] == 'lbol':
        makestruct_dict["global_free_parameters"] = ['TEFF', 'MONH', 'VRAD']
        # Bit silly seeing as we actually do this all printing and stuff in part 3 anyway.
        print('Running in LBOL mode, glob_free are TEF, MONH, VRAD, (VSINI), GRAV adjusted from TEFF, KMAG, PLX')
        print("KMAG =", str(reduction_variable_dict['k_mag']),
              "mag, DISTANCE =", str(reduction_variable_dict['dist']),
              ",PLX =", str(reduction_variable_dict['plx']),
              "mass, E_PLX/PLX =", str(reduction_variable_dict['e_plx'] /
                                       reduction_variable_dict['plx']))

    elif makestruct_dict['field_end'] == 'seis':
        makestruct_dict["global_free_parameters"] = ['TEFF', 'MONH', 'VRAD']
        # Vsini is rotational velocity.
        print("Running in SEISMIC mode, glob_free are TEF, MONH, VRAD, (VSINI), GRAV adjusted from TEFF, NU_MAX")
        print("NU_MAX =", str(reduction_variable_dict['nu_max']),
              ", E_NU_MAX/NU_MAX =", str(reduction_variable_dict['e_nnu_max'] /
                                         reduction_variable_dict['nu_max']))

#    # Fixed because we only use the linemask values during sme solve.
#    elif 'fixed' in makestruct_dict['field_for_obs']:
##        makestruct_dict["global_free_parameters"] = ['VRAD']
##        makestruct_dict["global_free_parameters"] = ['VRAD', 'TEFF', 'MONH']
#        if ((makestruct_dict['vrad_flag'] == 'whole') |
#                (makestruct_dict['vrad_flag'] == 'each')):
#            makestruct_dict["global_free_parameters"] = ['TEFF', 'MONH', 'VRAD']
#            print("Running in modified FIXED mode, glob_Free are " \
#                    + "TEFF, MONH, VRAD, (VSINI)")
#        elif ((makestruct_dict['vrad_flag'] == 'fix') |
#                (makestruct_dict['vrad_flag'] == 'none')):
#            makestruct_dict["global_free_parameters"] = ['TEFF', 'MONH']
#            print("Running in modified FIXED mode, glob_Free are " \
#                    + "TEFF, MONH, (VSINI)")
##        makestruct_dict["global_free_parameters"] = ['TEFF', 'MONH']
##        print("Running in FIXED mode, glob_Free are VRAD, (VSINI)")
##        print("Running in modifired FIXED mode, glob_Free are VRAD, TEFF, MONH, (VSINI)")

    elif makestruct_dict['field_end'] == 'free':
        if ((makestruct_dict['vrad_flag'] == 'whole') |
                (makestruct_dict['vrad_flag'] == 'each')):
            makestruct_dict["global_free_parameters"] = ['TEFF', 'GRAV', 'MONH', 'VRAD', 'VSINI']
            print("Running in FREE mode, glob_Free are TEFF, GRAV, MONH, VRAD, (VSINI).")
        elif ((makestruct_dict['vrad_flag'] == 'fix') |
                (makestruct_dict['vrad_flag'] == 'none')):
            makestruct_dict["global_free_parameters"] = ['TEFF', 'GRAV', 'MONH', 'VSINI']
            print("Running in FREE mode, glob_Free are TEFF, GRAV, MONH, (VSINI).")

    return makestruct_dict

# A corrected version of the abundances in a format PySME accepts.
def element_abundance():
    """
    Input:
        None, always the same
    Output:
        abundances: Abundance of elements
    """
    abundances = np.array([  0.92067066  ,-1.1058957 , -10.9858957 , -10.6558957  , -9.3358957,
         -3.6458957  , -4.2558957  , -3.3758957  , -7.4758957  , -4.1958957,
         -5.8658957 ,  -4.5058957  , -5.6658957  , -4.5258957  , -6.6758957,
         -4.8958957  , -6.5358957  , -5.8558957  , -6.9558957  , -5.7258957,
         -8.8658957 ,  -7.1358957  , -8.0358957  , -6.3958957  , -6.6458957,
         -4.5858957  , -7.1158957  , -5.8058957  , -7.8258957  , -7.4358957,
         -9.1558957   ,-8.4558957  , -9.7458957  , -8.7058957  , -9.4758957,
         -8.7858957   ,-9.4358957  , -9.1158957  , -9.8258957  , -9.4558957,
         -10.6158957,  -10.1158957 , -20.0358957 , -10.1958957 , -10.9158957,
         -10.3758957 , -11.0958957 , -10.2658957 , -10.4358957 , -10.0358957,
         -11.0358957 ,  -9.8458957 , -10.5258957 ,  -9.7958957 , -10.9658957,
         -9.8658957  ,-10.9058957  ,-10.3358957  ,-11.4558957  ,-10.5858957,
         -20.0358957 , -11.0358957 , -11.5158957 , -10.9258957 , -11.7558957,
         -10.8958957 , -11.5258957 , -11.1058957 , -12.0358957 , -10.9558957,
         -11.9758957 , -11.1558957 , -12.2058957 , -10.9258957 , -11.8058957,
         -10.7858957 , -10.6558957 , -10.3958957 , -11.0258957 , -10.9058957,
         -11.1358957 , -10.0358957 , -11.3858957 , -20.0358957 , -20.0358957,
         -20.0358957 , -20.0358957 , -20.0358957 , -20.0358957 , -11.9758957,
         -4.0358957  ,-12.5558957  ,-20.0358957  ,-20.0358957  ,-20.0358957,
         -20.0358957 , -20.0358957 , -20.0358957 , -20.0358957 ])

    return abundances

