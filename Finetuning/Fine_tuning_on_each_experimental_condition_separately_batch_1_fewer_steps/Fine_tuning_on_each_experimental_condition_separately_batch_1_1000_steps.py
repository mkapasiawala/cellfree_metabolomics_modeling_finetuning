# IMPORTS
# import functions for crn building and inference
import biocrnpyler as bcrn
from biocrnpyler import *
from bioscrape.types import import_sbml
import bioscrape as bioscr
from bioscrape.inference import py_inference
from bioscrape.simulator import py_simulate_model

# import standard python data analysis and plotting functions
import bokeh.io
import bokeh.plotting
import pandas as pd 
import bokeh.palettes
import warnings
from scipy.integrate import ODEintWarning
import pickle
import numpy as np

# import custom scripts
# import custom scripts
import sys
sys.path.insert(1, '../')
from useful_functions import *

# import packages to set up webdriver for Bokeh image export
import selenium
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.webdriver import WebDriver as Firefox
service = Service()
options = Options()
options.add_argument("--headless")
options.set_preference('layout.css.devPixelsPerPx', f'{1}')
webdriver = Firefox(service=service, options=options)

# import computing packages
from multiprocessing import Pool
import os
os.environ["OMP_NUM_THREADS"] = "1"

""

# LOAD INPUT FILES

# Load TX-TL data
master_df = pd.read_csv('../../Data/tidy_data_calibrated_full.csv')

# get relevant degfp background subtraction data
degfp_bkgd_subt_dict = get_dict_of_background_subtract_functions(master_df)

# Load parameter dictionary
with open('../../Data/top_params_dict.pkl', 'rb') as f:
    loaded_param_dict = pickle.load(f)


""
# FORMAT EXPERIMENTAL DATA AND INITIAL CONDITIONS

# get dataframe that contains relevant data
temp_df = master_df.drop(index=master_df[((master_df['Fuel'] !='Succinate') & (master_df['Fuel'] !='Pyruvate')  & (master_df['Fuel'] !='Maltose')  & (master_df['Fuel'] !='3PGA')) |
                                                 ((master_df['Extract'] !='AR_ZJ') & (master_df['Extract'] !='DAJ_MK_MY')) |
                                                 (master_df['Channel'] != 'deGFP') |
                                                 (master_df['DNA (nM)'] != 5) |
                                                 (master_df['Plasmid construct'] != 'pOR1OR2-MGapt-deGFP') | 
                                                 (master_df['mRNA (uM)'] != 0) | 
                                                 (master_df['DNA/mRNA batch'] >= 5)
                                                ].index).groupby(['Extract','Fuel', 'DNA (nM)', 'Fuel (mM)', 'Mg (mM)', 'Plasmid construct'])


# create list to store data
exp_data = []
ordered_groups_of_interest = []

# iterate through groups
for name, group in temp_df:

    # add group to order list
    ordered_groups_of_interest.append(name)
            
    # get relevant information
    extract = name[0]
    fuel = name[1]
    dna_conc = name[2]
    fluor_mol = name[3]
    
    # background subtraction
    bkgd_sub_fn = degfp_bkgd_subt_dict[list(group['Extract'])[0]]
    group['Measurement'] = group['Measurement'] - bkgd_sub_fn(group['Time (hr)'])

    # average over replicates
    cols = list(group.columns)
    cols.remove('Measurement')
    cols.remove('Replicate')
    cols.remove('Well')
    new_group = group.copy()
    new_group = new_group.groupby(cols).apply(lambda group2: np.mean(group2['Measurement'])*10**9, include_groups=False).reset_index().ffill()
    new_group = pd.DataFrame(new_group).rename({0:'GFP'}, axis=1)
    
    # drop all but relevant columns
    irrelevant_cols = list(new_group.columns)
    for i in ['Time (sec)', 'GFP']:
        irrelevant_cols.remove(i)
    new_group = new_group.drop(columns=irrelevant_cols)
    
    # add dataframe 
    exp_data.append(new_group)


# experimental conditions list
initial_conditions = []

# iterate through random conditions, create corresponding initial conditions for each
for this_group in ordered_groups_of_interest:
    fuel_conc = this_group[3]
    dna_conc = this_group[2]
    this_dict = {}
    this_dict['G'] = dna_conc*10**6
    this_dict['F'] = fuel_conc*10**12
    initial_conditions.append(this_dict)

""
# CREATE A STANDARD PRIOR FOR ALL FINE-TUNING

# params to estimate
params_to_estimate = ['kex__', 'Kex__', 'KW', 'nW']

# initialize prior dicitonary
prior = {}

# create initial seed array in shape of (nwalkers, nparams)
nwalkers = 100
nparams = len(params_to_estimate)
init_seed = np.zeros((nwalkers, nparams))

# iterate through params, create prior and init_seed
for index, p in enumerate(params_to_estimate):

    # get np array of all param values
    x = [i[p] for k, i in loaded_param_dict.items()]
    x = np.array(x)

    # normalize some params
    concentration_multiplier = 10**9
    if p in ['Kex__', 'KW']:
        x = x*concentration_multiplier

    # construct prior
    if p in ['kex__']:
        prior[p] = ['gaussian', np.median(x), 0.5, 'positive']
    elif p in ['nW']:
        prior[p] = ['gaussian', np.median(x), 2, 'positive']
    else:
        prior[p] = ['gaussian', np.median(x), np.std(x)*3, 'positive']

    # add values to init_seed
    init_seed[:, index] = x


""
# CREATE MODEL WHERE PARAMS ARE MEAN VALUE

# iterate through param sets, construct dict of mean values
crn_params = {}
argmax_params = loaded_param_dict[list(loaded_param_dict.keys())[0]]
for param in argmax_params.keys():
    x = [i[param] for k, i in loaded_param_dict.items()]
    x = np.array(x)
    crn_params[param] = np.median(x)

# create a CRN from that set of parameters
this_CRN = create_model_for_this_param_set(crn_params,
                                        waste_tl_inhibition=False, 
                                        waste_chelation=False,  
                                        W_B_binding_stoich=1,
                                        )

# format CRN and make every parameter a global parameter
filename = 'temp_model3.xml'
this_CRN.write_sbml_file(filename)

# update parameter names, write new xml file
update_model_local_to_global_params(filename)

# define model 
filename2 = filename.split('.')[0] + '_updated.xml'
M = import_sbml(filename2)



""
# RUN INFERENCE

# initialize dictionary to store mase values
mase_dict = {}

# iterate through experimental conditions, do fine-tuning, plot and save results
nsteps = 1000
for this_index, this_name in enumerate(ordered_groups_of_interest):

    # # select randomly a set of parameters from the list of top 100 parameter sets
    # param_ind = random.randrange(0,100)
    # argmax_params = loaded_param_dict[list(loaded_param_dict.keys())[param_ind]]

    # # create a CRN from that set of parameters
    # this_CRN = create_model_for_this_param_set(argmax_params,
    #                                         waste_tl_inhibition=False, 
    #                                         waste_chelation=False,  
    #                                         W_B_binding_stoich=1,
    #                                         )

    # # format CRN and make every parameter a global parameter
    # filename = 'temp_model.xml'
    # this_CRN.write_sbml_file(filename)
    
    # # update parameter names, write new xml file
    # update_model_local_to_global_params(filename)
    
    # # define model 
    # filename2 = filename.split('.')[0] + '_updated.xml'
    # M = import_sbml(filename2)
        
    # # create a prior using the relevant parameter values
    # concentration_multiplier = 10**9
    # prior = {
    #             'kex__' : ['gaussian', argmax_params['kex__'], 0.5, 'positive'],
    #             'Kex__' : ['gaussian', argmax_params["Kex__"]*concentration_multiplier, argmax_params["Kex__"]*concentration_multiplier*3, 'positive'],
    #             'KW' : ['gaussian', argmax_params["KW"]*concentration_multiplier, argmax_params["KW"]*concentration_multiplier*3, 'positive'],
    #             'nW' : ['gaussian', argmax_params['nW'], 2, 'positive']
    #         }

    # # extract information from prior
    # init_seed, params_to_estimate = take_prior_give_info(prior)

    # run inference
    sampler, pid = py_inference(Model = M,
                                exp_data = exp_data[this_index], 
                                measurements = ['GFP'], 
                                time_column = ['Time (sec)'],
                                initial_conditions = initial_conditions[this_index],
                                nwalkers = nwalkers, 
                                nsteps = nsteps, 
                                init_seed = init_seed,
                                sim_type = 'deterministic',
                                params_to_estimate = params_to_estimate,
                                prior = prior,
                                plot_show = False,
                                convergence_check = False, 
                                parallel=True
                               )

    # file names for export
    extract = this_name[0]
    fuel = this_name[1]
    dna_conc = this_name[2]
    fuel_conc = this_name[3]
    mg_conc = this_name[4]
    promoter = this_name[5].split('-')[0]
    filename_for_export = f'{extract}_{fuel}_{dna_conc}nM_DNA_{fuel_conc}mM_fuel_{mg_conc}mM_Mg_{promoter}_plasmid_{nsteps}'
    filename_for_sampler_pid_export = f'../../Inference_Results/Sampler_and_PID_Objects/Expt_Batch_1/' + filename_for_export
    filename_for_image_export = f'../../Inference_Results/Plots/Expt_Batch_1/' + filename_for_export + '.png'
    
    # save inference results to respective files
    with open(filename_for_sampler_pid_export + f'_sampler.pickle', 'wb') as f:
        pickle.dump(sampler, f)
    with open(filename_for_sampler_pid_export + f'_pid.pickle', 'wb') as f:
        pickle.dump(pid, f)

    # plot and save results
    mase_values = plot_and_save_mcmc_simulation_single_initial_condition(M, 
                                                                         filename_for_image_export,
                                                                         pid, 
                                                                         sampler, 
                                                                         initial_conditions[this_index], 
                                                                         exp_data[this_index],
                                                                         discard=nsteps-1, 
                                                                         )

    # save mase values
    mase_dict[filename_for_export] = mase_values

# save dictionary of mase values
filename_for_mase_dict_export = f'../../Inference_Results/MASE_Values/mase_dict_expt_batch_1_{nsteps}.pickle'
with open(filename_for_mase_dict_export, 'wb') as f:
        pickle.dump(mase_dict, f)
