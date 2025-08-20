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
master_df = pd.read_csv('../Finetuning/Data/tidy_data_calibrated_full.csv')

# get relevant degfp background subtraction data
dajmkmy_bkgd_samples = master_df.loc[(master_df['DNA (nM)']==0) &
                      (master_df['DNA/mRNA batch'] == 6) &
                      (master_df['Extract']=='DAJ_MK_MY') & 
                      (master_df['Fuel (mM)']==0) & 
                      (master_df['Mg (mM)']==0) & 
                      (master_df['Plasmid construct']=='pOR1OR2-MGapt-deGFP') & 
                      (master_df['mRNA (uM)'] == 0) & 
                      (master_df['Channel'] == 'deGFP') &
                      (master_df['Tetratcycline (ug/mL)'] == 0)
                    ]
x = dajmkmy_bkgd_samples['Time (hr)']
y = dajmkmy_bkgd_samples['Measurement']
fn = np.polyfit(x, y, 5)
dajmkmy_degfp_fn = np.poly1d(fn)


# Load parameter dictionary
with open('../Finetuning/Data/top_params_dict.pkl', 'rb') as f:
    loaded_param_dict = pickle.load(f)



""
# FORMAT EXPERIMENTAL DATA AND INITIAL CONDITIONS

# get dataframe that contains relevant data
temp_df = master_df.drop(index=master_df[(master_df['DNA/mRNA batch'] != 6) |
                                         (master_df['Extract'] != 'DAJ_MK_MY') |
                                         (master_df['Tetratcycline (ug/mL)'] != 0) |
                                         (master_df['Extract fraction'] != 0.33) |
                                         (master_df['DNA (nM)'] == 0) |
                                         (master_df['Fuel (mM)'] == 0) |
                                         (master_df['Channel'] != 'deGFP')
                                        ].index).groupby(['Extract', 'Fuel', 'Fuel (mM)', 'Mg (mM)', 'Plasmid construct', 'DNA (nM)'])

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
    bkgd_sub_fn = dajmkmy_degfp_fn
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
    fuel_conc = this_group[2]
    dna_conc = this_group[5]
    this_dict = {}
    this_dict['G'] = dna_conc*10**6
    this_dict['F'] = fuel_conc*10**12
    initial_conditions.append(this_dict)



""
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
    if count == 5:
        groups_of_initial_conditions.append(this_ic_list)
        groups_of_expt_data.append(this_data_list)
        groups_of_ordered_groups.append(this_group_name_list)
        count = 0



""
# CREATE A STANDARD PRIOR FOR ALL FINE-TUNING

# params to estimate
params_to_estimate = ['kex', 'Kex', 'KW', 'nW', 'Gmin', 'Gmax', 'nmin', 'nmax']

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
        prior[p] = ['gaussian', np.median(x)*10**6, 0.5*10**6, 'positive']
    elif p in ['nW']:
        prior[p] = ['gaussian', np.median(x), 2, 'positive']
    else:
        prior[p] = ['gaussian', np.median(x), np.std(x)*3, 'positive']

    # add values to init_seed
    init_seed[:, index] = x

# add Gmin, Gmax
prior['Gmin'] = ['gaussian', 5*10**6, 1*10**6, 'positive']
prior['Gmax'] = ['gaussian', 7*10**6, 1*10**6, 'positive']
prior['nmin'] = ['gaussian', 5, 1, 'positive']
prior['nmax'] = ['gaussian', 2, 1, 'positive']
# add values to init_seed
init_seed[:, 4] = 5*10**6*scipy.stats.norm.rvs(loc=1, scale=1, size=nwalkers, random_state=47)
init_seed[:, 5] = 7*10**6*scipy.stats.norm.rvs(loc=1, scale=1, size=nwalkers, random_state=53)
init_seed[:, 6] = scipy.stats.norm.rvs(loc=5, scale=1, size=nwalkers, random_state=61)
init_seed[:, 7] = scipy.stats.norm.rvs(loc=2, scale=1, size=nwalkers, random_state=33)



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
this_CRN = create_model_for_this_param_set_dna_saturation2(crn_params,
                                                        waste_tl_inhibition=False, 
                                                        waste_chelation=False,  
                                                        W_B_binding_stoich=1,
                                                        )

# format CRN and make every parameter a global parameter
filename = 'temp_model5.xml'
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
nsteps = 50000
for this_index, this_group_of_names in enumerate(groups_of_ordered_groups):
    
    # subset data and initial conditions for inference
    subset_exp_data = [groups_of_expt_data[this_index][0],
                       groups_of_expt_data[this_index][1],
                       groups_of_expt_data[this_index][3],
                       groups_of_expt_data[this_index][4]]
    subset_initial_conditions = [groups_of_initial_conditions[this_index][0],
                                   groups_of_initial_conditions[this_index][1],
                                   groups_of_initial_conditions[this_index][3],
                                   groups_of_initial_conditions[this_index][4]]

    # plot and save results
    this_particular_name = this_group_of_names[0]
    
    # file names for export
    extract = this_particular_name[0]
    fuel = this_particular_name[1]
    dna_conc = this_particular_name[5]
    fuel_conc = this_particular_name[2]
    mg_conc = this_particular_name[3]
    promoter = this_particular_name[4].split('-')[0]
    filename_for_export = f'{extract}_{fuel}_{fuel_conc}mM_fuel_{mg_conc}mM_Mg'
    filename_for_sampler_pid_export = f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Expt_Batch_3_only_4_dna_concs/' + filename_for_export
    filename_for_image_export = f'../Finetuning/Inference_Results/Plots/Expt_Batch_3_only_4_dna_concs/' + filename_for_export + '.png'
    
    # open sampler and pid
    with open(filename_for_sampler_pid_export + f'_sampler.pickle', 'rb') as f:
        sampler = pickle.load(f)
    with open(filename_for_sampler_pid_export + f'_pid.pickle', 'rb') as f:
        pid = pickle.load(f)

    # plot data
    img_save = f'../Figures/FigSE/{filename_for_export}.png' 
    
    # plot and save results
    plot_and_save_mcmc_simulation_multiple_dna_initial_conditions(M, 
                                                                 img_save,
                                                                 pid, 
                                                                 sampler, 
                                                                 groups_of_initial_conditions[this_index], 
                                                                 groups_of_expt_data[this_index],
                                                                 discard=nsteps-1, 
                                                                 timepoints_correction=True
                                                                 )


    
