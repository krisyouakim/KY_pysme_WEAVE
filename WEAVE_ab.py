#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd
import pickle
import pysme_makestruct
import time

from astropy.io import fits

import pysme_exec
import pysme_galah4
import params
###########################################################################
#
# This is the major program of the GALAH pipeline for step 2 (of 2):
# Elemental Abundance estimation
#
# It consists of 5 main parts: 
#
# 1) Initialise setup (variables, atmospheres etc.) 
# 2) Read in spectra + resolution (+ eventually correct telluric & skyline)
# 3) Read in stellar parameters
# 4) Optimisation of elemental abundances for
#      a) only one element with keyword mode='ZZ'
#      b) an element line with keyword mode='ZZ1234'
#      c) all elements with keyword mode='all'  
# 4.1) Segment selection, normalisation, synthesis with all elements
# 4.2) Synthesis only with element of choice (fixed segments & normalisation)
# 4.3) Iterative optimisation (fixed segments & normalisation)     
# 5) Clean up / diagnostic plots
#
########################################################################### 

# History
#
# 17/07/05 SB Starting from file used for reduction version 5.1
# 17/07/06 SB Adding comments to get better overview of (main) parts
# 17/07/06 SB Switch off telluric correction for IRAF reduction dr5.2
# 18/02/15 SB Set Li line.depth=0.99, because DEPTH files run /w sol. Li
# 18/02/28 SB Go to IRAF reduction 5.3 
# 10/01/22 KY Adapting the python version to take WEAVE data as input


# ### Part 1 (Get results from stellar parameters run)

mypath = '/proj/snic2020-16-23/shared/KY_pysme_WEAVE/'
file_name = \
  'conv_whole_733_WVE_09513992+0802006_WEAVE_HR_G_R_gums_229_SME_'

variables = pickle.load(open(mypath + 'OUTPUT/VARIABLES/%s' %file_name + \
                                'variables.pkl', 'rb'))
spectra = pickle.load(open(mypath + 'OUTPUT/SPECTRA/%s' %file_name + \
                                'spectra.pkl', 'rb'))

###########################################################################
#
# MAIN PART 1/5:
# Initialise setup
# 
###########################################################################

# Print & remember begin of calculations
start_time = time.time()
print("WEAVE_ab started") 

######################################
# Initialize variables for later use
######################################

# Create makestruct_dict by collecting the parameters from the input files
makestruct_dict, reduction_variable_dict = pysme_exec.collect_data()

#add some entries to makestruct_dict
makestruct_dict['continuum_scale_flag'] = "linear"                  
makestruct_dict['continuum_scale_type'] = 'mask' 
makestruct_dict['abundances'] = pysme_galah4.element_abundance()

print(makestruct_dict.keys())
print(makestruct_dict['field_end'])

# don't have this block of parameters in WEAVE #
version        = 1
vrad_flag      = -1 #-1: global vrad, 0: separate vrad for each wavelen segment
cscale_flag    = 1
auto_alpha     = 0 # galah
broad_lines    = \
    np.array([4861.323, 6562.797, 5167.3216, 5172.6843, 5183.6042]) 
    #Hbeta, Halpha, Mg Ib triplet
ps             = None
cannon         = None

field  = makestruct_dict['field_end']
objct  = makestruct_dict['obj_for_obs']
setup  = makestruct_dict['setup_for_obs']
mode   = makestruct_dict['res_type']

#field  = variables['field_end']
#objct  = variables['CNAME']
#setup  = 
#mode   =

#vsini          = makestruct_dict.rotational_vel.values[0]
#vradglob       = makestruct_dict.vrad_global.values[0]
#ab_free        = makestruct_dict.atomic_abundances_free_parameters.values[0]
#obs_name       = makestruct_dict.obs_name.values[0]
#obs_file       = obs_name+'.dat'
#depthmin       = makestruct_dict.depthmin.values[0]
#object_pivot   = objct
#LBOL           = np.nan
#glob_free      = makestruct_dict.global_free_parameters.values[0]
#cont_mask      = setup+'_Cont.dat'        
#segm_mask      = makestruct_dict.segment_mask.values[0]
#line_mask      = makestruct_dict.line_mask.values[0]
#line_cores     = makestruct_dict.line_cores.values[0]
#iterations     = makestruct_dict.iterations.values[0]

vsini          = variables['vsini']
vradglob       = variables['radial_velocity_global']
ab_free        = np.zeros(99)
obs_name       = makestruct_dict['obs_name']
obs_file       = obs_name+'.dat'
depthmin       = 0.0
object_pivot   = objct
LBOL           = np.nan
glob_free      = variables['fit_params']
cont_mask      = params.cont_mask
segm_mask      = makestruct_dict['segment_mask']
line_mask      = makestruct_dict['line_mask']
line_cores     = makestruct_dict['line_cores']
iterations     = makestruct_dict['iterations']

print ('Using minimum depth: %f' %depthmin)

elstr = ['H','He','Li','Be','B','C','N','O','F',
         'Ne','Na','Mg','Al','Si','P','S','Cl',
         'Ar','K','Ca','Sc','Ti','V','Cr','Mn',
         'Fe','Co','Ni','Cu','Zn','Ga','Ge','As',
         'Se','Br','Kr','Rb','Sr','Y','Zr','Nb',
         'Mo','Tc','Ru','Rh','Pd','Ag','Cd','In',
         'Sn','Sb','Te','I','Xe','Cs','Ba','La',
         'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb',
         'Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta',
         'W','Re','Os','Ir','Pt','Au','Hg','Tl',
         'Pb','Bi','Po','At','Rn','Fr','Ra','Ac',
         'Th','Pa','U','Np','Pu','Am','Cm','Bk','Cs','Es']

######################################
# Choose atmosphere & linelist
######################################
atmogrid_file = makestruct_dict['atmosphere_grid_file']
line_list_loc = makestruct_dict['line_list_location']
line_list     = makestruct_dict['original_line_list']

#atmogrid_file = params.atmosphere_grid_file
#line_list_loc = params.path + 'LINELIST/'
#line_list     = params.line_list

print('Using atmosphere grid: ', atmogrid_file)
print('Using linelist:        ', line_list)


# ### Part 2

###########################################################################
#
# MAIN PART 2/5:
# Read in spectra & possibly perform skyline & tellurics
# correction
#
###########################################################################

#######################################
## Read in 4 band spectra (ext. 0)
#######################################
#
##hdulist  = fits.open("%s/stacked_1002046_1002045.aps.fits" %mypath)
#hdulist  = fits.open(params.path + params.spec_file_green)
#
## getting data, column names and extra information from fits file
#ii = 5
#tbdata    = hdulist[ii].data     # getting everything
#cols      = hdulist[ii].columns  # description of the columns
#col_names = cols.names          # column names
#head      = hdulist[ii].header   # fits file discription
#
#cols_names = []  # create empty list where we will store column names
#cols_ab    = []  # create empty list where we will store data from columns
#
#for i in range(len(col_names)):            # loop through column names
#       cols_names.append(tbdata.names[i])  # append a name of a column to a list
#       cols_ab.append((tbdata.field(i)).T) # append data from corresponding column to a list
#    
#cols_names   = np.asarray(cols_names).flatten().T 
## .flatten() flattens array of column names to one dimension
## .T tranponse an array (we want a row instead of a column)
#
#our_table         = pd.DataFrame(data=cols_ab).T # create pd.DataFrame (table) and transponse it
#our_table.columns = cols_names                   # add column names to the DataFrame

###########################################################################
# Read linelist file into listlist_f dataframe #
###########################################################################

#Create dictionary to pass to the load_linelist function. This is done to make
#it compatible with the function that usually takes the makestruct dictionary
# as input
#line_list_dict = {"original_location": "DATA/LINELIST/", 
#                        "line_list": params.line_list}

line_list, line_list_type = pysme_makestruct.load_linelist(makestruct_dict)

        
linelist_f         = pd.DataFrame(line_list)    # create pd.DataFrame (table) and transponse it


###########################################################################
# Read mode file into mode_f dataframe #
mode_f = pd.read_csv('%sDATA/LINELIST/mode_DR3' %mypath, delim_whitespace=True, 
                        comment=';')

###########################################################################
# Read segmentation file into segm_f dataframe #
#segm_f = pd.read_fwf('%s/%s' %(mypath, makestruct_dict.segment_mask.values[0]), 
#                     header=None,
#                     delim_whitespace=True, 
#                     comment=';')
#segm_f.columns = ['line_st', 'line_en', 'res', 'com1', 'com2']
#
#segm_f = pd.read_fwf('%s/%s' %(mypath, makestruct_dict.segment_mask.values[0]), 
#                     header=None,
#                     delim_whitespace=True, 
#                     comment=';')

segm_f = pd.read_csv('%sDATA/LINELIST/%s' %(mypath, params.segment_mask), 
                delim_whitespace=True, header=None,                                                            
                names=["Wavelength_Start", "Wavelength_End", "Resolution", 
                            "comment", "overflow"],                                                
                engine='python', skipinitialspace=True, comment=';',                                                            
                usecols=["Wavelength_Start", "Wavelength_End", "Resolution"])

######################################
# 2nd try SME output & collected FITS
###################################### 
# read makestructu_dict
print ('Initial values according to SME SETUP')

# vradglob = inp[ind].vel
# turb     = vmic_vmac(teff,grav,feh)

teff     = variables['effective_temperature']
grav     = variables['gravity']
feh      = variables['metallicity']
vmic     = variables['microturbulence_velocity']
vsini    = variables['vsini']
vmac     = variables['vmac']

# marcs_abund,abund,fehmod=feh
# Increase initial Li abund
# abund[3-1] = abund[3-1] - feh + 1.25

###########################################################################
#
# 4) MAIN PART 4/5, ELEMENT ABUNDANCE OPTIMISATION
#    This can happen for
#    a) only one element with keyword mode='ZZ'
#    b) an element line with keyword mode='ZZ1234'
#    c) all elements with keyword mode='all'
#
###########################################################################

print(np.unique(linelist_f.species.values))
print(np.unique(linelist_f.columns))

# from mode file
iterr = [1, 1, 20]

#modeun   = np.unique(mode_f.elem.values)       # unique element
modeun   = ['Fe']      # unique element

for m in range(0, len(modeun)):

    mode = modeun[m]
    print('Running element: ', mode)

    if (len(mode) > 2): 
        elem = mode[:len(mode)-4]
    else:
        elem = mode   
         
    atom = np.asarray(np.where(np.asarray(elstr) == elem)).flatten() + 1 #Chose element from list
    if (len(atom) == 0):
        print(elem+' not found in mode_'+setup)
    else:
        line0    = mode_f[mode_f.elem == mode]['lambda'].values       # central line length
        line_st  = mode_f[mode_f.elem == mode]['maskst'].values       # line start 
        line_en  = mode_f[mode_f.elem == mode]['maskend'].values      # line end
        segm_st  = mode_f[mode_f.elem == mode]['segst'].values        # segment start
        segm_en  = mode_f[mode_f.elem == mode]['segend'].values       # segment end
        ipres    = mode_f[mode_f.elem == mode]['res'].values          # The resolution of the instrument to simulate
#        line_lm  = linelist_f.LAMBDA.values                           # line lambda
#        line_nm  = linelist_f.NAME_0.values                           # line name has three columns, chararray convert to np
        line_lm  = linelist_f.wlcent.values                           # line lambda
#        line_nm  = linelist_f.species.values                           # line name has three columns, chararray convert to np
        line_nm  = np.array(list(map(lambda x: x.split()[0], linelist_f.species.values)))                           # line name has three columns, chararray convert to np

#        print(line_lm)
#        print(line_nm[line_lm == 7799.9957])
#        print(line_lm[(line_lm > 7795) & (line_lm < 7805)])
#        raise
#        print(linelist_f.species.values[:20])
#        print(np.array(list(map(lambda x: x.split()[0], linelist_f.species.values[:20]))))

        #################################################
        ### Increase depth & adjust log_gf for Si7800 ###
        #search for line with a specific wavelength and element name
        # Commented out because WEAVE does not have this line
#        si_7800 = np.where((line_lm == 7799.9957) & 
#                           (line_nm == 'Si')) 
#        si_7800 = np.asarray(list(si_7800)).flatten()[0]            #convert index to np array and then int
#        
#        linelist_f.DEPTH[si_7800]  = 0.99 
#        linelist_f.LOG_GF[si_7800] = -0.75
        
#        line_lm  = linelist_f.LAMBDA.values                         # line lambda
#        line_nm  = linelist_f.NAME_0.values                         # line name has three columns, chararray convert to np
        
        # Increase depth for Li6707
#        li_6707 = np.where((line_lm == 6707.7635) & 
#                           (line_nm == 'Li'))                        #name has three columns, chararray convert to np
#        li_6707 = np.asarray(list(li_6707)).flatten()[0]             #convert index to np array and then int
#        
#        linelist_f.DEPTH[li_6707]  = 0.99 
        #################################################
        
        
        line_dp = linelist_f.depth.values                           # line depth
        jcall   = np.zeros(len(line0))
        
        for i in range(0, len(line0)):
            depth_check = np.where((np.abs(line_lm - line0[i]) < 1e-5) & 
                                   (line_nm == elem))  
            depth_check = np.asarray(list(depth_check))
            
            
            j  = np.where((np.abs(line_lm - line0[i]) < 1e-5) & 
                          (line_nm == elem) &
                          (line_dp > depthmin))
            j  = np.asarray(list(j)).flatten()
            jc = len(j)
            jcall[i] = jc
            
        if (np.sum(jcall) == 0):
            print ('Predicted line(s) too weak, exiting')
        else:
            #Print new line & segment mask with visible lines
            i       = np.where(jcall != 0)
            i       = np.asarray(list(i)).flatten() 
            
            full  = []
            clean = []
            for i in range(0, len(line0)):
                j  = np.where((segm_st < line0[i]) &
                              (segm_en > line0[i]))
                j  = np.asarray(list(j)).flatten()
                
                k  = np.where((line_lm > segm_st[i]) &
                              (line_lm < segm_en[i]) &
                              (line_dp > depthmin))
                k  = np.asarray(list(k)).flatten()
                if (len(k) > 0):
                    full.append(k)
                    
                k  = np.where((line_lm > segm_st[i]) &
                              (line_lm < segm_en[i]) &
                              (line_nm == elem) &
                              (line_dp > depthmin)) 
                k  = np.asarray(list(k)).flatten()
                if (len(k) > 0):
                   clean.append(k)
        
            for i in range(0, len(broad_lines)):
                j = np.where(line_lm == broad_lines[i])
                j = np.asarray(list(j)).flatten()
                if (len(j) > 0):
                   clean.append(j)
                if (len(j) > 0):
                   full.append(j)                
            
            if (len(clean) > 0):
               clean = np.concatenate(clean).ravel()
               clean = np.unique(clean)
               clean = np.sort(clean)
            
            if (len(full) > 0):    
               full  = np.concatenate(full).ravel()
               full  = np.unique(full)
               full  = np.sort(full)

            linelist_f_cut = linelist_f.iloc[full]
            linelist_f_cut = linelist_f_cut.reset_index(drop=True)
            linelist_f_cut.to_csv('%s/linelist_cut_ab.csv' %mypath, index=False, sep=',') 
            linelist_f_cut = pd.read_csv('%s/linelist_cut_ab.csv' %mypath, sep=',')

            nlte = 1 # set to 0 for AVATAR runs
            nlte_elem_flags = np.empty(99)
            nlte_grids      = np.empty(99)
            nltee = ['Li','C','O','Na','Mg','Al','Si','K','Ca','Mn','Ba']
            nltez = [  3 , 6 , 8 , 11 , 12,  13 , 14 , 19, 20 , 25 , 56 ]
            inlte = np.where(nltee == elem)
            inlte = np.asarray(inlte).flatten()

            if ((nlte == 1) & (len(inlte) != 0)):
               print ('NLTE on for '+nltee[inlte])
            
               nlte_elem_flags[nltez[inlte]] = 1 
               nlte_grids[nltez[inlte]]      = 'Amarsi19_' + nltee[inlte] + '.grd'
            
               if ((nltee[inlte] == 'Li') & (teff < 5750)):
                   nlte_grids[nltez[inlte]] = 'Amarsi19_' + nltee[inlte] + '_t3800_5750.grd'
                
               if ((nltee[inlte] == 'Li') & (teff >= 5750)): 
                   nlte_grids[nltez[inlte]] = 'Amarsi19_' + nltee[inlte] + '_t5750_8000.grd'
            else:
               print ('LTE only')

            for k in range(0, len(iterr)):
                if (k == 0): 
                    print (' STARTING LOOP 1 -> Normalization full synthesis')
                if (k == 1): 
                    print (' STARTING LOOP 2 -> Element synthesis')
                if (k == 2): 
                    print (' STARTING LOOP 3 -> Element abundance optimisation (max. 20 iterations)')
      
                #These two lines do the same thing, except maxiter is 
                #defined inside the pysme call (from makestruct_dict)
                maxiter = iterr[k]
                makestruct_dict['current_iterations'] = iterr[k]

                # Print info for the run
                #print (field + '_' + objct, teff,grav,feh,vmac,vmic,vsini,vradglob,k,mode)
                print (field + '_' + objct, teff,grav,feh,vmac,vmic,vsini,k,mode) # don't print vrad
        
                ###########################################################################
                #
                # LOOP STEP 4.1: SEGMENT SELECTION AND NORMALISATION WITH FULL SYNTHESIS
                #
                ###########################################################################   
    
                if (k == 0) :
                    cscale_flag   = 1
                    norm          = 1
                    ab_free[atom] = 0
         
                    if (mode == 'H') & (iterr[k] <= 1): 
                        glob_free = '-1'
                    else: 
                        ab_free[atom] = 0
    
                    # Synthesize run with all lines within segments
                    makestruct_dict, sme_out = \
                        pysme_makestruct.create_structure(makestruct_dict,
                            reduction_variable_dict,
                            normalise_flag=\
                                    makestruct_dict['normalise_flag'],
                            sel_elem="all",
                            run="Synthesise")

                    if (ps is not None):
                        os.system('cp -f OUTPUT/' + obs_name + '_SME.out ' + obs_name + '_SME_full.out')
                        #!!! inspect,field,objct,setup,mode,/ps,/norm,yr=[-0.1,1.1],label=3
                        os.system('mv -f '+obs_name+'.ps '+obs_name+'_0.ps')    
                    else:
         
                         ###########################################################################
                         #
                         # LOOP STEP 4.2: SYNTHESIS WITH ONLY ELEMENT OF MODE
                         #
                         ###########################################################################
         
                         if (k == 1):
                             vradglob      = 0
                             cscale_flag   = -3
                             norm          = 0
                             ab_free[atom] = 0

                             depthmin = 0.1 
            
                             #Restore full synthesis from k=0
#                             restore, 'OUTPUT/' + obs_name + '_SME.out'
                             sme_full      = sme_out
                             sme_full_mob  = sme_full.mob
                             sme_full_wave = sme_full.wave
                             sme_full_smod = sme_full.smod
    
                             obs_file = obs_name + '.dat'
            
                             #Run with list only containing the element
                             #mwrfits,line[clean],'LINELIST/'+line_list,/create
    
#                             if (version > 4.00):
#                                 print ('using SME pipeline version ' + str(version))
#                                 run    = 2
#                                 nlte   = nlte
#                                 frames = [run, nlte]
#                                 df_st  = pd.DataFrame(frames).T
#                                 df_st.columns = ['run', 'nlte'] #!!! /new_sme
#                             else:
#                                 print ('using SME pipeline version ' + str(version))
#                                 run    = 2
#                                 nlte   = nlte
#                                 frames = [run, nlte]
#                                 df_st  = pd.DataFrame(frames).T
#                                 df_st.columns = ['run', 'nlte']

                             # Change min depth now that we have synthesized 
                             # depth values for our spectrum
                             makestruct_dict['depthmin'] = 0.1
                             print('Using minimum depth: %f' %makestruct_dict['depthmin'])
                             # Synthesize run with only list containing 
                             # lines only from the current element lines
                             makestruct_dict, sme_out = \
                                 pysme_makestruct.create_structure(
                                         makestruct_dict,
                                         reduction_variable_dict,
                                         normalise_flag=\
                                         makestruct_dict['normalise_flag'],
                                         sel_elem=modeun[0],
                                         run="Synthesise")
#
#                             if (ps is not None):
#                                 #!!! inspect,field,object,setup,mode,/ps,/norm,yr=[-0.1,1.1],label=3
#                                 os.system('mv -f '+obs_name+'.ps '+obs_name+'_1.ps')
                                
                             #Restore element synthesis
                             #restore,'OUTPUT/'+obs_name+'_SME.out'
                             sme_clean      = sme_out #!!!
                             sme_clean_mob  = sme_clean.mob
                             sme_clean_wave = sme_clean.wave
                             sme_clean_smod = sme_clean.smod
    
                             if (ps is not None):
                                 os.system('cp -f OUTPUT/'+obs_name+'_SME.out '+obs_name+'_SME_clean.out')
                                         
                             #Double-check that the wavelength scales are identical
                             if (np.max(np.abs(sme_full.wave-sme_clean.wave)) > 0.0001):
                                 npix = np.min([len(sme_full.wave),len(sme_clean.wave)])
    #                              from scipy import interpolate
    #                              interpfunc = interpolate.interp1d(b,a, kind='linear')
    #                              x=interpfuc(c)
                                 sme_full.smod[0:npix] = interpol(sme_full.smod, sme_full.wave, sme_clean.wave[0:npix-1])
                                 sme_full.wave[0:npix] = sme_clean.wave[0:npix]
                             
                             #Build up new line mask by comparing synthetic spectra
                             dum1    = []  
                             dum2    = [] 
                             dum3    = []
                             chimax  = 0.005
                             eps     = 1 #Buffer
            
                             for i in range(0, len(line0)):
                                 mob  = np.where((sme_clean.wave >= line_st[i]) &
                                                 (sme_clean.wave <= line_en[i]) &
                                                ((sme_full.smod-sme_clean.smod)**2 < chimaxmobc))
                                 mob  = np.asarray(list(mob)).flatten()
                                 mobc = len(mob)
                                
                                 print ('Clean MOB: ' + str(mobc) + ' with chimax ' + str(chimax))
                                 print ((sme_full.smod[mob] - sme_clean.smod[mob])**2)
                                 if (mobc >= 4):
                                     for j in range(0, mobc):
                                         if (j == 0):
                                            dum1 = dum1.append(sme_clean.wave[mob[j]] - eps)
                                            dum2 = dum2.append(sme_clean.wave[mob[j]] + eps)
                                            dum3 = dum3.append(line0[i])
                                         else:
                                            if (mob[j] == mob[j-1]+1):
                                               dum2.append(sme_clean.wave[mob[j]] + eps)
                                            else:
                                                dum1 = dum1.append(sme_clean.wave[mob[j]] - eps)
                                                dum2 = dum2.append(sme_clean.wave[mob[j]] + eps)
                                                dum3 = dum3.append(line0[i])
                                                    
                                 else:
                                     print ('Less than 5 MOB left: ')     
    
                                 #No clean regions left
                                 if (len(dum3) <= 1):
                                     print ('No clean regions left')
                                 else:   
                                     #Print new line-mask for k=2 & revert to full list
                                     print('DATA/'+obs_name+'.dat',dum3[1:,],dum1[1:,],dum2[1:,])
                                     linelist_f_cut = linelist_f.iloc[full]
                                     linelist_f_cut = linelist_f_cut.reset_index(drop=True)
                                     linelist_f_cut.to_csv('%s/linelist_cut_ab.csv' %mypath, index=False, sep=',') 
                                     linelist_f_cut = pd.read_csv('%s/linelist_cut_ab.csv' %mypath, sep=',')
    
                                                  
                         ###########################################################################
                         #
                         # LOOP STEP 4.3: OPTIMISATION OF ABUNDANCE OF ELEMENT OF MODE
                         #
                         ###########################################################################
    
                         if (k == 2):
    
                            if ((mode == 'Fe') & (cannon is not None)):
                               glob_free     = ['VRAD','VSINI']
                               ab_free[atom] = 1
                                    
                            if (mode == 'Ba5854'):
                                glob_free = ['VRAD']
                            if ((mode == 'Ca6718') or (mode == 'Ca5868')): 
                                glob_free = ['VRAD']
                            if (mode == 'Si6722'):
                                glob_free = ['VRAD']
                                                            
                            cscale_flag     = -3
                            norm            = 0
                            ab_free[atom-1] = 1
            
#                            if (version > 4.00):
#                                print ('using SME pipeline version ' + str(version))
#                                run    = 1
#                                nlte   = nlte
#                                frames = [run, nlte]
#                                df_st  = pd.DataFrame(frames).T
#                                df_st.columns = ['run', 'nlte']
#                            else:
#                                print ('using SME pipeline version ' + str(version))
#                                run    = 1
#                                nlte   = nlte
#                                frames = [run, nlte]
#                                df_st  = pd.DataFrame(frames).T
#                                df_st.columns = ['run', 'nlte']


                            # Solve run with only list containing 
                            # lines only from the current element lines
                            makestruct_dict, sme_out = \
                                pysme_makestruct.create_structure(
                                        makestruct_dict,
                                        reduction_variable_dict,
                                        normalise_flag=\
                                        makestruct_dict['normalise_flag'],
                                        sel_elem=modeun[0],
                                        run="Solve")
    
                           #Add sme.full to access spectrum used for
                           #normalisation & the continuum masks
                           #!!! cmsave,file='OUTPUT/'+obs_name+'_SME.out',sme_full_mob,sme_full_wave,sme_full_smod, sme_clean_mob, sme_clean_wave, sme_clean_smod,/append
    
                         if (ps is None):
                             #!!! inspect,field,object,setup,mode,/ps,/norm,yr=[-0.1,1.1],label=3
                             os.system('mv -f '+obs_name+'.ps '+obs_name+'_2.ps')    
        
   
            ###########################################################################
            #
            # 4) MAIN PART 5/5, CLEAN UP AND DIAGNOSTICS
            #
            ###########################################################################    

            print("WEAVE_ab run time: %f minutes" % ((time.time() - start_time)/60.))
    
            #!!! finishline:
    
            os.system('rm -f DATA/'+obs_name+'.dat')
            os.system('rm -f DATA/'+obs_name+'_Segm.dat')
            os.system('rm -f LINELIST/'+obs_name+'.fits')
            os.system('rm -f OUTPUT/'+obs_name+'_SME.inp')
            os.system('rm -f SPECTRA/'+obs_file)
    
            if (m != len(modeun)):
                print('Continuing with next element/line')

print("WEAVE_ab run time: %f minutes" % ((time.time() - start_time)/60.))

# Output SME structure
np.save('OUTPUT/' + obs_name + '.npy', sme_out)

# If running in VM-mode, save the log-file
os.system('mv -f idl_'+obs_name+'.log OUTPUT_'+field+'/')
os.system('mv -f OUTPUT/'+obs_name+'_SME.out OUTPUT_'+field+'/')
