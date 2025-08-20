# IMPORTS

# import standard python data analysis and plotting functions
import bokeh.io
import bokeh.plotting
from bokeh.models import Label, Div
from bokeh.layouts import gridplot, layout, row, column, Spacer
import pandas as pd 
import bokeh.palettes
import warnings
from scipy.integrate import ODEintWarning
import pickle
import numpy as np
import scipy.stats
import pandas as pd
from base64 import b64encode 

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

import holoviews as hv
hv.extension('bokeh')


from bokeh.themes import Theme 

theme = Theme(json={'attrs': {
    
# apply defaults to Figure properties
'figure': {
    'toolbar_location': None,
    'outline_line_color': None,
    'min_border_right': 10,
    'height':1000,
    'width':1200,
},    
    
# apply defaults to Grid properties
'Grid': {
    'grid_line_color': None,
},
    
# apply defaults to Title properties
'Title': {
    'text_font_size': '30pt',
    'align': 'center'
},
    
# # apply defaults to Plot properties
# 'Plot': {
#     'renderers': {'add_layout': {'legend[0]':'right'}},
# },
    
# apply defaults to Axis properties
'Axis': {
    'major_label_text_font_size': '50pt',
    'axis_label_text_font_size': '55pt',
    'axis_label_text_font_style': 'normal',
    'axis_label_standoff':30
},

# apply defaults to Legend properties
'Legend': {
    'background_fill_alpha': 0.8,
    'location': 'top_right',
    "label_text_font_size": '15pt',
    "click_policy": 'hide',
    "title_text_font_style": 'normal',
    "title_text_font_size": '18pt'
}}})

bokeh.io.curdoc().theme = theme
hv.renderer('bokeh').theme = theme


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


# iterate through some conditions of interest
# extract, fuel, dna_conc, fuel_conc, mg_conc, promoter
conds_of_interest = [('DAJ_MK_MY', '3PGA', 5, 0, 'pOR1OR2'),
                     ('DAJ_MK_MY', '3PGA', 20, 6, 'pOR1OR2'),
                     ('DAJ_MK_MY', '3PGA', 45, 0, 'pOR1OR2'),
                    ]
indices_of_interest = [0, 21, 30]

# load some inference data for conditions to plot
for index, this_name in enumerate(conds_of_interest):
    
    # file names for export
    extract = this_name[0]
    fuel = this_name[1]
    fuel_conc = this_name[2]
    mg_conc = this_name[3]
    promoter = this_name[4]
    filename_for_export = f'{extract}_{fuel}_{fuel_conc}mM_fuel_{mg_conc}mM_Mg'
    filename_for_sampler_pid_export = f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Selected_Samplers_and_PIDs/' + filename_for_export

    # open sampler and pid
    with open(filename_for_sampler_pid_export + f'_sampler.pickle', 'rb') as f:
        sampler = pickle.load(f)
    with open(filename_for_sampler_pid_export + f'_pid.pickle', 'rb') as f:
        pid = pickle.load(f)

    # get exp data and initial conditions
    this_index = indices_of_interest[index]
    this_init_cond = groups_of_initial_conditions[this_index]
    this_exp_data = groups_of_expt_data[this_index]

    # plot data
    img_save = f'../Figures/{filename_for_export}.png'

    # plot and save results
    nsteps = 50000
    mase_values = plot_and_save_mcmc_simulation_multiple_dna_initial_conditions(M, 
                                                                                img_save,
                                                                                pid,
                                                                                sampler,
                                                                                this_init_cond,
                                                                                this_exp_data,
                                                                                discard=nsteps-1, 
                                                                                timepoints_correction=True
                                                                               )

