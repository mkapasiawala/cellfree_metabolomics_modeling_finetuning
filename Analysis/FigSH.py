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
import scipy.stats

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



##################################################

# LOAD INPUT FILES

# Load TX-TL data
master_df = pd.read_csv('../Finetuning/Data/tidy_data_calibrated_full.csv')

# get relevant degfp background subtraction data
degfp_bkgd_subt_dict = get_dict_of_background_subtract_functions(master_df)

# Load parameter dictionary
with open('../Finetuning/Data/top_params_dict.pkl', 'rb') as f:
    loaded_param_dict = pickle.load(f)


##################################################

# FORMAT EXPERIMENTAL DATA AND INITIAL CONDITIONS

# get dataframe that contains relevant data
temp_df = master_df.drop(index=master_df[(master_df['Fuel'] !='3PGA') |
                                         ((master_df['Extract'] !='AR_ZJ') & (master_df['Extract'] !='DAJ_MK_MY')) |
                                         (master_df['Channel'] != 'deGFP') |
                                         (master_df['DNA (nM)'] != 5) |
                                         (master_df['Plasmid construct'] != 'pOR1OR2-MGapt-deGFP') | 
                                         (master_df['mRNA (uM)'] != 0) | 
                                         (master_df['DNA/mRNA batch'] >= 5)
                                        ].index).groupby(['Extract','Fuel', 'DNA (nM)', 'Fuel (mM)', 'Plasmid construct', 'Mg (mM)'])


# create list to store data
exp_data = []
ordered_groups_of_interest = []

# iterate through groups
for name, group in temp_df:

    # add group to order list
    ordered_groups_of_interest.append(name)
    
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
    mg_conc = this_group[5]
    this_dict = {}
    this_dict['G'] = dna_conc*10**6
    this_dict['F'] = fuel_conc*10**12
    this_dict['B'] = mg_conc*10**12
    initial_conditions.append(this_dict)



##################################################

# GROUP TOGETHER DATA CORRESPONDING TO SAME DNA CONC

# create lists to store groups of groups
groups_of_initial_conditions = []
groups_of_expt_data = []
groups_of_ordered_groups = []

# add groups to lists
count = 0
for index, this_group in enumerate(ordered_groups_of_interest):
    if count == 0:
        this_data_list = []
        this_ic_list = []
        this_group_name_list = []
    this_data_list.append(exp_data[index])
    this_ic_list.append(initial_conditions[index])
    this_group_name_list.append(this_group)
    count += 1
    if count == 6:
        groups_of_initial_conditions.append(this_ic_list)
        groups_of_expt_data.append(this_data_list)
        groups_of_ordered_groups.append(this_group_name_list)
        count = 0



##################################################

# CREATE A STANDARD PRIOR FOR ALL FINE-TUNING

# params to estimate
params_to_estimate = ['kex', 'Kex', 'KW', 'nW', 'KB', 'nB', 'log_k_buffer_b', 'k_buffer_u']

# initialize prior dicitonary
prior = {}

# create initial seed array in shape of (nwalkers, nparams)
nwalkers = 100
nparams = len(params_to_estimate)
init_seed = np.zeros((nwalkers, nparams))

# iterate through params, create prior and init_seed
for index, p in enumerate(params_to_estimate[:4]):

    # get np array of all param values
    if p in ['kex', 'Kex']:
        p2 = f'{p}__'
        x = [i[p2] for k, i in loaded_param_dict.items()]
    else:
        x = [i[p] for k, i in loaded_param_dict.items()]
    x = np.array(x)

    # normalize some params
    concentration_multiplier = 10**9
    if p in ['Kex', 'KW']:
        x = x*concentration_multiplier

    # construct prior
    if p in ['kex']:
        prior[p] = ['gaussian', np.median(x), 0.5, 'positive']
    elif p in ['nW']:
        prior[p] = ['uniform', 0.001, 8, 'positive']
    else:
        prior[p] = ['gaussian', np.median(x), np.std(x)*3, 'positive']

    # add values to init_seed
    init_seed[:, index] = x

# add mg params
W_B_binding_stoich = 4
prior['KB'] = ['gaussian', 4*10**12, 3*10**12, 'positive']
prior['nB'] = ['uniform', 0.001, 8, 'positive']
prior['log_k_buffer_b'] = ['gaussian', -46, 1]
prior['k_buffer_u'] = ['gaussian', .1, 1, 'positive']
# add values to init_seed
init_seed[:, 4] = 4*10**12*np.abs(scipy.stats.norm.rvs(loc=1, scale=1, size=nwalkers, random_state=21))
init_seed[:, 5] = np.abs(scipy.stats.norm.rvs(loc=2, scale=1, size=nwalkers, random_state=22))
init_seed[:, 6] = -46*np.abs(scipy.stats.norm.rvs(loc=1, scale=0.2, size=nwalkers, random_state=23))
init_seed[:, 7] = scipy.stats.norm.rvs(loc=.1, scale=1, size=nwalkers, random_state=24)


##################################################

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
                                           waste_tl_inhibition=True, 
                                           waste_chelation=True,  
                                           W_B_binding_stoich=W_B_binding_stoich,
                                           nB_val=2.001,
                                           unbinding_rxn=True
                                            )

# format CRN and make every parameter a global parameter
filename = 'temp_model7.xml'
this_CRN.write_sbml_file(filename)

# update parameter names, write new xml file
update_model_local_to_global_params(filename)

# define model 
filename2 = filename.split('.')[0] + '_updated.xml'
M = import_sbml(filename2)



##################################################

# RUN INFERENCE

# initialize dictionary to store mase values
mase_dict = {}

# iterate through experimental conditions, do fine-tuning, plot and save results
nsteps = 50000
this_index = 0
this_group_of_names = groups_of_ordered_groups[this_index]

# subset data for this extract
exp_data = exp_data[:36]
initial_conditions = initial_conditions[:36]

# plot and save results
for i in range(0,6):
    this_index = i
    this_group_of_names = groups_of_ordered_groups[this_index]
    this_particular_name = this_group_of_names[0]

    # file names for export
    extract = this_particular_name[0]
    fuel = this_particular_name[1]
    dna_conc = this_particular_name[2]
    fuel_conc = this_particular_name[3]
    mg_conc = this_particular_name[5]
    promoter = this_particular_name[4].split('-')[0]
    filename_for_export = f'{extract}_{fuel}_{fuel_conc}mM_fuel'
    filename_for_sampler_pid_export = f'../../Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs6/' + filename_for_export
    filename_for_image_export = f'../../Inference_Results/Plots/Expt_Batch_4_all_mg_concs6/' + filename_for_export + '.png'
    
    # save inference results to respective files
    if i==0:
        with open(f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/{extract}_sampler.pickle', 'rb') as f:
            sampler = pickle.load(f)
        with open(f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/{extract}_pid.pickle', 'rb') as f:
            pid = pickle.load(f)

    # plot data
    img_save = f'../Figures/FigSH/{filename_for_export}.png' 
        
    # plot and save results
    timepoints = np.array(groups_of_expt_data[this_index][0]['Time (sec)'])
    plot_and_save_mcmc_simulation_multiple_mg_initial_conditions(M, 
                                                                 img_save,
                                                                 pid, 
                                                                 sampler, 
                                                                 groups_of_initial_conditions[this_index], 
                                                                 groups_of_expt_data[this_index],
                                                                 discard=nsteps-1, 
                                                                 timepoints_correction=True,
                                                                 )
