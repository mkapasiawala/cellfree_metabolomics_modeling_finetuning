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
import holoviews as hv
from bokeh.themes import Theme
hv.extension("bokeh")

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

# get relevant degfp background subtraction data for each lysate
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
arzj_bkgd_samples = master_df.loc[(master_df['DNA (nM)']==0) &
                      (master_df['DNA/mRNA batch'] < 5) &
                      (master_df['Extract']=='AR_ZJ') & 
                      (master_df['Fuel (mM)']==0) & 
                      (master_df['Mg (mM)']==0) & 
                      (master_df['Plasmid construct']=='pOR1OR2-MGapt-deGFP') & 
                      (master_df['mRNA (uM)'] == 0) & 
                      (master_df['Channel'] == 'deGFP') &
                      (master_df['Tetratcycline (ug/mL)'] == 0)
                    ]
x = arzj_bkgd_samples['Time (hr)']
y = arzj_bkgd_samples['Measurement']
fn = np.polyfit(x, y, 5)
arzj_degfp_fn = np.poly1d(fn)

# create dictionary to store functions
degfp_bkgd_subt_dict = {'AR_ZJ': arzj_degfp_fn,
                        'DAJ_MK_MY': dajmkmy_degfp_fn
                           }



####################################################################

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


# PLOT EXPERIMENTAL DATA HEATMAPS

# plot heatmaps
def create_heatmaps(combined_max_degfp_df, fuel, dna_conc, extract):

    # define rounding
    value_dimension = hv.Dimension('Maximum deGFP (μM)', value_format=lambda x: '%.2f' % x)    
    
    # plot range max
    new_group = combined_max_degfp_df.copy()
    new_group['Fuel (mM)'] = pd.Categorical(new_group['Fuel (mM)'],categories=['0', '5', '10', '15', '20', '30', '45'])
    new_group['Mg (mM)'] = pd.Categorical(new_group['Mg (mM)'],categories=['0', '2', '4', '6', '8', '10'])
    new_group = new_group.sort_values(['Fuel (mM)', "Mg (mM)"])
    new_group = new_group.reset_index().drop(labels=['index'], axis=1)
    this_max = np.max(new_group['Maximum deGFP (μM)'])
    
    # create holoviews heatmap
    hv_fig = hv.HeatMap(data=new_group,
                        kdims=[('Fuel (mM)', f'Added {fuel} (mM)'), ('Mg (mM)', 'Added Mg²⁺ (mM)') ],
                        vdims=hv.Dimension('Maximum deGFP (μM)', range=(0, this_max))
    # ).aggregate(function=np.max
    ).opts(
        frame_height=500,
        frame_width=500,
        colorbar=True,
        show_grid=False,
        cmap='Viridis',
        # title = f'Maximum deGFP (μM)',
        fontsize={'labels': 35, 'xticks': 30, 'yticks': 30, 'ticks':30}
    )
    hv_fig = hv_fig * hv.Labels(hv_fig, vdims=value_dimension).opts(text_color='white', text_font_size='20px',fontsize={'ticks':20})
    
    # render as bokeh figure, format
    hv_fig2 = hv.render(hv_fig)
    hv_fig2.toolbar_location = None
    hv_fig2.title.align = 'center'
    
    # # add plot title
    # if extract == 'AR_ZJ':
    #     title = Div(text = f'<p style="color: #444444; font-size: 75px; margin:0;"> Lysate 1, {dna_conc}nM DNA </p>', align='center')
    # else:
    #     title = Div(text = f'<p style="color: #444444; font-size: 75px; margin:0;"> Lysate 2, {dna_conc}nM DNA </p>', align='center')
    # p = column(title, hv_fig1, Spacer(height=50), hv_fig2)

    return(hv_fig2)

# function to extract data
def create_max_degfp_integrated_mgapt_df(master_df, fuel, extract, dna_conc):

    # get dataframe for particular fuel, extract, and dna conc
    subset_df2 = master_df.drop(index=master_df[(master_df['Fuel'] !=fuel) |
                                             (master_df['Extract'] !=extract) |
                                             (master_df['DNA (nM)'] != dna_conc) |
                                             (master_df['Plasmid construct'] != 'pOR1OR2-MGapt-deGFP') | 
                                             (master_df['mRNA (uM)'] != 0) | 
                                             (master_df['DNA/mRNA batch'] >= 5) |
                                                (master_df['Channel'] != 'deGFP')
                                            ].index)
    combined_df = subset_df2.copy()
    temp_df = combined_df.groupby(['Extract','Fuel', 'DNA (nM)', 'Channel', 'NTPs (mM)', 'Added GTP (mM)', 'Added ATP (mM)'], observed=False)

    # create list to store dataframes
    integrated_mgapt_dfs = []
    max_degfp_dfs = []
    
    # iterate through groups
    for name, group in temp_df:
                    
        # get relevant information
        extract = name[0]
        fuel = name[1]
        dna_conc = name[2]
        fluor_mol = name[3]
        
        # background subtraction
        if fluor_mol == 'MGaptamer':
            bkgd_sub_fn = mgaptamer_bkgd_subt_dict[list(group['Extract'])[0]]
        elif fluor_mol == 'deGFP':
            bkgd_sub_fn = degfp_bkgd_subt_dict[list(group['Extract'])[0]]
        group['Measurement'] = group['Measurement'] - bkgd_sub_fn(group['Time (hr)'])    
        
        # drop unrelated columns, create new df
        group = group.drop(columns=['Well'])
        cols = list(group.columns)
        cols.remove('Replicate')
        cols.remove('Measurement')
        new_group = group.groupby(cols, observed=False).apply(lambda group2: np.mean(group2['Measurement'])).reset_index().ffill()#fillna(method="ffill")
        new_group = pd.DataFrame(new_group).rename({0:'Measurement'}, axis=1)
    
        if fuel=='None':
            print(new_group.head())
            
        # perform an integration
        if fluor_mol == 'MGaptamer':
            cols2 = list(new_group.columns)
            cols2.remove('Measurement')
            cols2.remove('Time (hr)')
            cols2.remove('Time (sec)')
            # new_group = new_group.groupby(cols2).apply(lambda group2: np.trapz(y=group2['Measurement'])).reset_index().ffill()#fillna(method="ffill")
            timepoints = sorted(list(set(new_group['Time (hr)'])))
            dx = timepoints[2] - timepoints[1]
            new_group = new_group.groupby(cols2, observed=False).apply(lambda group2: np.trapz(y=group2['Measurement'], dx=dx)).reset_index().ffill()
            new_group = pd.DataFrame(new_group).rename({0:'Measurement'}, axis=1)
        elif fluor_mol == 'deGFP':
            new_group = new_group.drop(columns=['Time (sec)'])
            cols = list(new_group.columns)
            cols.remove('Measurement')
            cols.remove('Time (hr)')
            new_group = new_group.groupby(cols, observed=False).apply(lambda group2: np.max(group2['Measurement'])).reset_index().ffill()#fillna(method="ffill")
            new_group = pd.DataFrame(new_group).rename({0:'Measurement'}, axis=1)
            
        # axis labels
        new_group['Fuel (mM)'] = new_group['Fuel (mM)'].astype('str')
        new_group['Mg (mM)'] = new_group['Mg (mM)'].astype('str')
        
        # drop all but relevant columns
        irrelevant_cols = list(new_group.columns)
        for i in ['Extract', 'DNA (nM)', 'Fuel (mM)', 'Mg (mM)', 'Fuel', 'NTPs (mM)', 'Added ATP (mM)', 'Added GTP (mM)', 'Measurement']:
            irrelevant_cols.remove(i)
        new_group = new_group.drop(columns=irrelevant_cols)
        
        # rename measurement column, add relevant data to relevant list of dataframes
        if fluor_mol == 'MGaptamer':
            new_group = pd.DataFrame(new_group).rename({'Measurement':'Integrated MGaptamer (μM-hr)'}, axis=1)
            integrated_mgapt_dfs.append(new_group)
        elif fluor_mol == 'deGFP':
            new_group = pd.DataFrame(new_group).rename({'Measurement':'Maximum deGFP (μM)'}, axis=1)
            max_degfp_dfs.append(new_group)

    # combine list of dataframes into single dataframe for each channel
    # combined_integrated_mgapt_df = pd.concat(integrated_mgapt_dfs)
    combined_max_degfp_df = pd.concat(max_degfp_dfs)

    return(combined_max_degfp_df)



# # iterate through fuels
# for index, fuel in enumerate(['3PGA']):
    
#     # iterate through extracts
#     for extract in ['AR_ZJ', 'DAJ_MK_MY']:
        
#         dna_conc = 5
#         combined_max_degfp_df = create_max_degfp_integrated_mgapt_df(master_df, fuel, extract, dna_conc)
        
#         # give information to plotting function
#         p = create_heatmaps(combined_max_degfp_df, fuel, dna_conc, extract)

#         # save image
#         file_save = f'../Figures/FigS5B_{extract}_expt.png'
#         bokeh.io.reset_output()
#         bokeh.io.curdoc().theme = theme
#         bokeh.io.export_png(p, filename = file_save)    







###############################################################

# CREATE MODEL SIMULATION HEATMAPS

###############################################################


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
df_list = []
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
    filename_for_sampler_pid_export = f'../../Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/' + filename_for_export
    filename_for_image_export = f'../../Inference_Results/Plots/Expt_Batch_4_all_mg_concs5/' + filename_for_export + '.png'
    
    # save inference results to respective files
    if i==0:
        with open(f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/{extract}_sampler.pickle', 'rb') as f:
            sampler = pickle.load(f)
        with open(f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/{extract}_pid.pickle', 'rb') as f:
            pid = pickle.load(f)
        
    # plot and save results
    timepoints = np.array(groups_of_expt_data[this_index][0]['Time (sec)'])
    df = summarize_degfp_endpoint(M, 
                                 filename_for_image_export,
                                 pid, 
                                 sampler, 
                                 groups_of_initial_conditions[this_index], 
                                 groups_of_expt_data[this_index],
                                 discard=nsteps-1, 
                                 timepoints_correction=True,
                                 )
    df_list.append(df)

# concat dfs
full_df = pd.concat(df_list, axis=0)

# save mase values
cols = list(full_df.columns)
cols.remove('Maximum deGFP (μM)')
new_group = full_df.groupby(cols, observed=False).apply(lambda group2: np.mean(group2['Maximum deGFP (μM)'])).reset_index().ffill()#fillna(method="ffill")
new_group = pd.DataFrame(new_group).rename({0:'Maximum deGFP (μM)'}, axis=1)
new_group['Fuel (mM)'] = new_group['Fuel (mM)'].astype(int).astype('str')
new_group['Mg (mM)'] = new_group['Mg (mM)'].astype(int).astype('str')

# give information to plotting function
extract = 'AR_ZJ'
p = create_heatmaps(new_group, '3PGA', 5, extract)

# save image
file_save = f'../Figures/FigS5B_{extract}_model_50000.png'
bokeh.io.reset_output()
bokeh.io.curdoc().theme = theme
bokeh.io.export_png(p, filename = file_save)   


##################################################

# RUN INFERENCE

# initialize dictionary to store mase values
mase_dict = {}

# iterate through experimental conditions, do fine-tuning, plot and save results
nsteps = 50000
this_index = 0
this_group_of_names = groups_of_ordered_groups[this_index]

# subset data for this extract
exp_data = exp_data[36:]
initial_conditions = initial_conditions[36:]

# plot and save results
df_list = []
for i in range(6,12):
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
    filename_for_sampler_pid_export = f'../../Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/' + filename_for_export
    filename_for_image_export = f'../../Inference_Results/Plots/Expt_Batch_4_all_mg_concs5/' + filename_for_export + '.png'
    
    # save inference results to respective files
    if i==6:
        with open(f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/{extract}_sampler.pickle', 'rb') as f:
            sampler = pickle.load(f)
        with open(f'../Finetuning/Inference_Results/Sampler_and_PID_Objects/Expt_Batch_4_all_mg_concs5/{extract}_pid.pickle', 'rb') as f:
            pid = pickle.load(f)
        
    # plot and save results
    timepoints = np.array(groups_of_expt_data[this_index][0]['Time (sec)'])
    df = summarize_degfp_endpoint(M, 
                                 filename_for_image_export,
                                 pid, 
                                 sampler, 
                                 groups_of_initial_conditions[this_index], 
                                 groups_of_expt_data[this_index],
                                 discard=nsteps-1, 
                                 timepoints_correction=True,
                                 )
    df_list.append(df)

# concat dfs
full_df = pd.concat(df_list, axis=0)

# save mase values
cols = list(full_df.columns)
cols.remove('Maximum deGFP (μM)')
new_group = full_df.groupby(cols, observed=False).apply(lambda group2: np.mean(group2['Maximum deGFP (μM)'])).reset_index().ffill()#fillna(method="ffill")
new_group = pd.DataFrame(new_group).rename({0:'Maximum deGFP (μM)'}, axis=1)
new_group['Fuel (mM)'] = new_group['Fuel (mM)'].astype(int).astype('str')
new_group['Mg (mM)'] = new_group['Mg (mM)'].astype(int).astype('str')


# give information to plotting function
extract = 'DAJ_MK_MY'
p = create_heatmaps(new_group, '3PGA', 5, extract)

# save image
file_save = f'../Figures/FigS5B_{extract}_model_50000.png'
bokeh.io.reset_output()
bokeh.io.curdoc().theme = theme
bokeh.io.export_png(p, filename = file_save)   