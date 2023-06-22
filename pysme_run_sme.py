""" Runs pysme either as solve or syntehsize, then output the results in an 
output folder. Synthesize changes the spectra, and solve varies the parameters.
"""
import tqdm
import os

# Removes the progress bar. Must come before imports.
class _TQDM(tqdm.tqdm):
    def __init__(self, *argv, **kwargs):
        kwargs['disable'] = True
        if kwargs.get('disable_override', 'def') != 'def':
            kwargs['disable'] = kwargs['disable_override']
        super().__init__(*argv, **kwargs)

    def _time(x):
        pass


tqdm.tqdm = _TQDM
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, r"/media/sf_SME-master/src")

import pickle
import numpy as np

from pysme.gui import plot_plotly
from pysme import sme as SME
from pysme import util
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum
from pysme.abund import Abund
from pysme.linelist.linelist import LineList

def start_sme(temp_sme, cut_line_list, run):
    """
    Only function in the file, runs sme after setting up its input parameters 
    from our input.
                  Currently supported file formats:
                * ".npy": Numpy save file of an SME_Struct
                * ".sav", ".inp", ".out": IDL save file with an sme structure
                * ".ech": Echelle file from (Py)REDUCE
    So we cannot load with SME_structure.load(), we must instead set each 
    parameter individually.

    Input:
        temp_sme: dictionary 
            Makestruct_dict in the other files, this contains all the 
            information required to run Pysme. It is given in dictionary 
            form so must be set manually.
        linelist: pandas data_frame
            Pysme requires the linelist both in the regular naming 
            convention and also a seperate class
        run: 
            Used to determine whether we run Synth or Solve
    Output:
        None, although we save the result as seperate files in the OUTPUT 
        folder.
    """
    obs_name = temp_sme['obs_name']

    # Load your existing SME structure or create your own. We create our own.
    sme = SME.SME_Structure()
    pickle.dump(temp_sme, open("Latest input to sme.pkl", "wb"))

    # Set all sme class properties as the names of the input dict, skipping 
    # those that don't exist or are set later.
    for key in temp_sme:
        try:
            # We set those in deeper class properties
            if key != 'abund' and key != 'atmo' and key != 'maxiter':
                setattr(sme, key, temp_sme[key])
        except AttributeError:
            print(AttributeError, key)
    element_abund1 = np.array([  
          0.92067066  ,-1.1058957 , -10.9858957 , -10.6558957  , -9.3358957,
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

    # We set abundance out of the loop due to SME class differences
    sme.abund = Abund(sme.monh, element_abund1)
    sme.atmo.source = temp_sme["atmo"]["source"]
    sme.atmo.method = temp_sme["atmo"]["method"]
    sme.atmo.depth = temp_sme["atmo"]["depth"]
    sme.atmo.interp = temp_sme["atmo"]["interp"]
    sme.atmo.geom = temp_sme["atmo"]["geom"]

    #no nlte for now
#    sme.nlte.set_nlte("H", temp_sme['nlte_abund'][0])
#    sme.nlte.set_nlte("Fe", temp_sme['nlte_abund'][1])

    # set the linelist in the sme object to the processed line list loaded 
    # earlier in pysme_makestruct.py
    sme.linelist = cut_line_list 
#    sme.linelist = LineList(linedata=linelist, lineformat="long", 
#        medium="vac")
    print('######################## %s ###################' %run)
    print('######################## %d ###################' %temp_sme['maxiter'])

    sme.fitresults.maxiter = temp_sme['maxiter']
    sme.fitparameters = temp_sme["fitparameters"]

    print("\n\n\nmetallicity and grav and teff BEFORE sme", \
                sme.monh, sme.logg, sme.teff, "\n\n\n")

    sme.h2broad = True
    sme.specific_intensities_only = False
    sme.normalize_by_continuum = True
    sme.vrad_flag = temp_sme['vrad_flag']

    print("len here", len(sme.wave))

    print("temp sme cscale type is", temp_sme['cscale_type'])
    print("temp sme cscale flag is", temp_sme['cscale_flag'])
    print("temp sme vrad flag is", temp_sme['vrad_flag'])

    # Start the logging to the file

    # Start SME solver
    print("Running in ", run, "mode")
    print("maxiter is %d" %sme.fitresults.maxiter)
    sme.save("smeinput", sme)
    pickle.dump(sme, open("Starting sme.pkl", "wb"))
    util.start_logging('run_logs/' + temp_sme['obs_name']+'_log')

    if run == "Synthesise":
        print("Starting sme_synth")
        sme = synthesize_spectrum(sme)
    elif run == "Solve":
        print("Starting sme_solve")
        # Fitparameters (globfree) come fom the sme structure.
        sme = solve(sme, param_names=sme.fitparameters)
#        obs_name = 'conv_' + obs_name
    else:
        print("Run is neither Synthesise or Solve, that's a big ERROR. Exiting.")
        exit()


    # Applies the fix that we perform on synth to spec, the observed spectra.
    for seg in range(len(sme.wave)):
        x = sme.wave[seg] - sme.wave[seg][0]
        cont = np.polyval(sme.cscale[seg], x)
        sme.spec[seg] = sme.spec[seg] / cont

    print(sme.citation())
#    print("Cscale after a run is:", sme.cscale)
    print("Sme accwi and accrt:", sme.accwi, sme.accrt)
    print("AFter a", run," run, sme vsini, nmu, mu is ", sme.vsini, sme.nmu, sme.mu)
    print("len of synth right after this run (", temp_sme['balmer_run'], "is balmer?", len(sme.synth))
    print("Finished current sme loop, starting afresh.")
    print("VRAD AFTER SME IS", sme.vrad)
    print("sme ipres is:", sme.ipres)
    print("Fit parameters are still", sme.fitparameters)
    print("\n\n\nmetallicity and grav and teff AFTER sme", sme.monh, sme.logg, sme.teff, "\n\n\n")
    # We create dicts to save out as mentioned below. Only some are saved, the 
    # ones that are unused and set again each run are not bothered with.
    # Only happens when not running a balmer line run with a diff fit file.
    print("After, fitresults are", sme.fitresults)
    print("and maxiter is ", sme.fitresults.maxiter)
    print("iptype is ", sme.iptype)

    spectra_dict = {"wave": sme.wave,   'flux': sme.spec,
                    "error": sme.uncs,  'mask': sme.mask,
                    "synth": sme.synth}

    variables_dict = {"effective_temperature": sme.teff,    
                      "gravity": sme.logg,
                      "metallicity": sme.monh,
                      "radial_velocity_global": sme.vrad,
                      "abundances": sme.abund,
                      "microturbulence_velocity": sme.vmic,
                      "macroturbulence_velocity": sme.vmac,
                      "rotational_velocity": sme.vsini,
                      "field_end": temp_sme['field_end'],
                      "load_file": True,
                      'segment_begin_end': sme.wran,
                      "ipres": sme.ipres,
                      "vsini": sme.vsini,
                      "vmac": sme.vmac,
                      "depth": sme.depth,
                      "fit_params": sme.fitparameters,
                      "fit_results": sme.fitresults,
                      "CNAME": temp_sme['object']
                      }

    print("Balmer run is", temp_sme['balmer_run'])
    if not temp_sme['balmer_run']:
        # Convert to pickle to open elsewhere, because opening SME classes 
        # outside of linux is v hard. This could be changed once all code is on 
        # the same platform perhaps. But I like using non-niche stuff.
        # Overwrites the previous sme input, which in turn is recreated and 
        # overwrites this output. Save results
        pickle.dump(variables_dict, 
                        open(r"OUTPUT/VARIABLES/" + obs_name 
                                        + "_SME_variables.pkl", "wb"))
        pickle.dump(spectra_dict, open(r"OUTPUT/SPECTRA/" + obs_name 
                                        + "_SME_spectra.pkl", "wb"))


        print("Saved SME output in OUTPUT/SPECTRA and OUTPUT/VARIABLES")
        # For testing in one big file, we currently don't use it.
        np.save(r"OUTPUT/FullOutput/SME_Output", sme)

    # IF it is a balmer run we don't ovewrite our good data, and we don't 
    # update logg.
    elif temp_sme['balmer_run']:
        print("Finishing balmer run, saving separate files.")

        pickle.dump(variables_dict, open(r"OUTPUT/VARIABLES/" + obs_name 
                + "_Balmer_variables.pkl", "wb"))
        pickle.dump(spectra_dict, open(r"OUTPUT/SPECTRA/" + obs_name 
                + "_Balmer_spectra.pkl", "wb"))
        a = pickle.load(open(r"OUTPUT/SPECTRA/" + obs_name 
                + "_Balmer_spectra.pkl", "rb"))

        np.save(r"OUTPUT/FullOutput/SME_Balmer_Output", sme)

    # return the sme output object
    return sme
