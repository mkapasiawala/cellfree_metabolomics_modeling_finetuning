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





def update_model_local_to_global_params(filename):

    ''' rename parameters in sbml file so that they have same ID'''
    
    # load your old SBML file
    sbml_doc = libsbml.readSBMLFromFile(filename)
    
    # iterate through reactions, add parameters to dictionary
    param_dict = {}
    sbml_model = sbml_doc.getModel()
    for rxn in sbml_model.getListOfReactions():
        kl = rxn.getKineticLaw()
        for param in kl.getListOfLocalParameters():
            param_dict[param.getValue()] = param.getId()
                
    # maintain list of unique parameters
    added_params = []
    changelog = {}
    count = 0
    # iterate through parameter dictionary
    for p_val, p in param_dict.items():
        sbml_param = sbml_model.createParameter()
        # if parameter is already present
        if p in added_params:
            p_new = p+str(count)
            changelog[p_val] = p_new
            sbml_param.setId(p_new)
            added_params.append(p_new)
        else:
            sbml_param.setId(p)
            added_params.append(p)
        count += 1
        sbml_param.setValue(p_val)
        sbml_param.setConstant(False)
    
    # replace local parameters with global parameters
    for rxn in sbml_model.getListOfReactions():
        kl = rxn.getKineticLaw()
        for param in list(kl.getListOfLocalParameters()):
            if param.getValue() in changelog:
                print('renaming {0} with {1}'.format(param.getId(), changelog[param.getValue()]))
                kl.setFormula(kl.getFormula().replace(param.getId(), changelog[param.getValue()]))
            kl.removeLocalParameter(param.getId())

    # update file name
    updated_filname = filename.split('.xml')[0] + '_updated.xml'
    
    # write new sbml file with global parameters
    libsbml.writeSBML(sbml_doc, updated_filname)





def take_prior_give_info(prior):
    
    # init seed
    init_seed = []
    for i in prior.items():
        init_seed.append(i[1][1])

    # params to estimate
    params_to_estimate = []
    for i in prior.items():
        params_to_estimate.append(str(i[0]))
    
    return(init_seed, params_to_estimate)



















# create function that takes in parameter set to create sbml model
def create_model_for_this_param_set(argmax_params, 
                                    waste_tl_inhibition=False, 
                                    waste_chelation=False, 
                                    KB_val=4000, 
                                    nB_val = 2.0, 
                                    W_B_binding_stoich = 1, 
                                    nW_val=None, 
                                    KW_val=None,
                                    unbinding_rxn = True
                                   ):

    # create species
    G = bcrn.Species("G")       # Gene
    X = bcrn.Species("GFP")     # Protein
    E_s = bcrn.Species("E_s")   # NTPs
    E_d = bcrn.Species("E_d")   # NDPs
    F = bcrn.Species("F")       # 3PGA
    W = bcrn.Species('W')       # Waste

    # create additional species
    W_B_binding_stoich = W_B_binding_stoich
    B = bcrn.Species("B")       # Buffer
    BW = bcrn.Complex([B] + [W]*W_B_binding_stoich)   # Buffer Waste Complex
    species = [G, X, E_s, E_d, F, W, B, BW]

    # define initial concentrations
    G0 = 2.3*10**6                        # 10^-14 M; 48ng/uL = 2.3 nM
    E_s_0 = 5*10**12                      # 10^-14 M; previous ntp0
    F0 = 30*10**12                        # 10^-14 M; previous pga0
    W0 = 0
    concentration_multiplier = 10**9

    # create lists to store expression reactions and energy/waste reactions
    reactions_ex = []
    reactions_E_W = []

    # TX-TL reaction
    kex = bcrn.ParameterEntry("kex", argmax_params["kex__"])
    Kex = bcrn.ParameterEntry("Kex", argmax_params["Kex__"]*concentration_multiplier)
    nexp = bcrn.ParameterEntry("nexp", argmax_params["nexp__"])
    prop_ex = bcrn.ProportionalHillPositive(k = kex, s1 = E_s, d = G, n = nexp, K = Kex)
    # optional modification to add waste inhibition of TXTL
    if waste_tl_inhibition == True:
        # Bmin = bcrn.ParameterEntry('Bmin', 0*10**12)
        # Bmax = bcrn.ParameterEntry('Bmax', 6.000001*10**12)
        # nmin = bcrn.ParameterEntry('nmin', 2)
        # nmax = bcrn.ParameterEntry('nmax', 3)
        # prop_ex = bcrn.GeneralPropensity(f"kex * {G} * ({B}^nmin + 1)/({B}^nmin + Bmin^nmin + 1) * Bmax^nmax/({B}^nmax + Bmax^nmax) * {E_s}^nexp/({E_s}^nexp + Kex^nexp)",
        #                                 propensity_species=[G, E_s, B],
        #                                 propensity_parameters=[nexp, kex, Kex, Bmin, Bmax, nmin, nmax])
        
        KB = bcrn.ParameterEntry("KB", KB_val*concentration_multiplier)
        nB = bcrn.ParameterEntry("nB", nB_val)
        prop_ex = bcrn.GeneralPropensity(f"kex * {G} * {E_s}^nexp/(Kex^nexp+{E_s}^nexp) * 1/(1+(({B})/KB)^nB)",
                                        propensity_species=[E_s, G, B],
                                        propensity_parameters=[kex, Kex, nexp, KB, nB])
    reactions_ex.append(bcrn.Reaction(inputs=[G], outputs=[G, X], propensity_type = prop_ex))


    # ATP regeneration reaction
    vmaxE = bcrn.ParameterEntry("vmaxE", argmax_params["vmaxATP"]*concentration_multiplier) #Previously vmaxATP
    KF= bcrn.ParameterEntry('KF', argmax_params["Kpga"]*concentration_multiplier) #Previously Kpga
    K_Ed = bcrn.ParameterEntry("K_Ed", argmax_params["Kndp"]*concentration_multiplier) #previously Kndp
    if KW_val is None:
        KW = bcrn.ParameterEntry("KW", argmax_params["KW"]*concentration_multiplier)
    else:
        KW = bcrn.ParameterEntry("KW", KW_val*concentration_multiplier)
    # optional different nW value
    if nW_val is None:
        nW = bcrn.ParameterEntry("nW", argmax_params["nW"])
    else:
        nW = bcrn.ParameterEntry("nW", nW_val)
    prop = bcrn.GeneralPropensity(f"vmaxE * {F}/(KF+{F}) * {E_d}/(K_Ed+{E_d}) * 1/(1+({W}/KW)^nW)", 
                            propensity_species = [E_d, F, W], 
                            propensity_parameters = [vmaxE, KF, nW, K_Ed, KW])
    reactions_E_W.append(bcrn.Reaction(inputs=[F]+3*[E_d], outputs=[W]+3*[E_s], propensity_type = prop))

    # 3PGA leak reaction
    KFW = bcrn.ParameterEntry("KFW", argmax_params["KpgaW__"]*concentration_multiplier) #Previously KpgaW
    vmaxW = bcrn.ParameterEntry("vmaxW", argmax_params["vmaxW__"]*concentration_multiplier) 
    nW_leak = bcrn.ParameterEntry("nW_leak", argmax_params["nW_leak__"])
    prop = bcrn.HillPositive(k = vmaxW, s1 = F, K = KFW, n = nW_leak)
    reactions_E_W.append(bcrn.Reaction([F], [W], propensity_type = prop))

    # ATP degredation reaction
    vmaxDEG = bcrn.ParameterEntry("vmaxDeg", argmax_params["vmaxDeg__"]*concentration_multiplier)
    KEdeg = bcrn.ParameterEntry("KEdeg", argmax_params["Kntpdeg__"]*concentration_multiplier) #previously Kntpdeg
    prop = bcrn.HillPositive(k = vmaxDEG, s1 = E_s, K = KEdeg, n = 1)
    reactions_E_W.append(bcrn.Reaction([E_s], [E_d], propensity_type = prop))

    # waste chelation reaction
    if waste_chelation == True:

        # new params
        rate_adjustment = concentration_multiplier**W_B_binding_stoich
        log_k_buffer_b_val = np.log10(.1/rate_adjustment)
        log_k_buffer_b = bcrn.ParameterEntry('log_k_buffer_b', np.log10(.1/rate_adjustment))
        # k_buffer_b = bcrn.ParameterEntry("k_buffer_b", 10**(log_k_buffer_b.value))
        k_buffer_u = bcrn.ParameterEntry("k_buffer_u", .1)
        binding_stoich = bcrn.ParameterEntry('binding_stoich', W_B_binding_stoich)

        # forward reaction
        prop = bcrn.GeneralPropensity(f'10^log_k_buffer_b * {B} * {W}^binding_stoich',
                                      propensity_species = [W, B], 
                                      propensity_parameters = [log_k_buffer_b, binding_stoich])
        reactions_E_W.append(bcrn.Reaction(inputs = [B] + [W]*W_B_binding_stoich,
                                           outputs = [BW],
                                           propensity_type = prop))
        
        # reverse reaction
        if unbinding_rxn == True:
            prop = bcrn.GeneralPropensity(f'k_buffer_u * {BW}',
                                          propensity_species = [BW], 
                                          propensity_parameters = [k_buffer_u])
            reactions_E_W.append(bcrn.Reaction(inputs = [BW], 
                                               outputs = [B] + [W]*W_B_binding_stoich,
                                               propensity_type = prop))
        
        # reactions_E_W.append(bcrn.Reaction.from_massaction([B] + [W]*W_B_binding_stoich, [BW], k_forward = k_buffer_b, k_reverse = k_buffer_u))

    # combine reaction lists
    reactions = reactions_ex + reactions_E_W

    # define initial concentrations
    initial_concentration = {G:G0, 
                            E_s:E_s_0, 
                            F:F0, 
                            W:W0, 
                            B:10*10**12}

    # create chemical reaction network
    CRN = bcrn.ChemicalReactionNetwork(species = species, reactions = reactions, initial_concentration_dict = initial_concentration)

    return(CRN)















# create function that takes in parameter set to create sbml model
def create_model_for_this_param_set_dna_saturation(argmax_params, 
                                                    waste_tl_inhibition=False, 
                                                    waste_chelation=False, 
                                                    KB_val=4000, 
                                                    nB_val = 2.0, 
                                                    W_B_binding_stoich = 1, 
                                                    nW_val=None, 
                                                    KW_val=None):

    # create species
    G = bcrn.Species("G")       # Gene
    X = bcrn.Species("GFP")     # Protein
    E_s = bcrn.Species("E_s")   # NTPs
    E_d = bcrn.Species("E_d")   # NDPs
    F = bcrn.Species("F")       # 3PGA
    W = bcrn.Species('W')       # Waste

    # create additional species
    W_B_binding_stoich = W_B_binding_stoich
    B = bcrn.Species("B")       # Buffer
    BW = bcrn.Complex([B] + [W]*W_B_binding_stoich)   # Buffer Waste Complex
    species = [G, X, E_s, E_d, F, W, B, BW]

    # define initial concentrations
    G0 = 2.3*10**6                        # 10^-14 M; 48ng/uL = 2.3 nM
    E_s_0 = 5*10**12                      # 10^-14 M; previous ntp0
    F0 = 30*10**12                        # 10^-14 M; previous pga0
    W0 = 0
    concentration_multiplier = 10**9

    # create lists to store expression reactions and energy/waste reactions
    reactions_ex = []
    reactions_E_W = []

    # TX-TL reaction
    kex = bcrn.ParameterEntry("kex", argmax_params["kex__"]*10**6)
    Kex = bcrn.ParameterEntry("Kex", argmax_params["Kex__"]*concentration_multiplier)
    nexp = bcrn.ParameterEntry("nexp", argmax_params["nexp__"])
    Gmin = bcrn.ParameterEntry('Gmin', 5.00001*10**6)
    Gmax = bcrn.ParameterEntry('Gmax', 7.000001*10**6)
    prop_ex = bcrn.GeneralPropensity(f"kex * {G}/({G} + Gmin) * Gmax/({G} + Gmax) * {E_s}^nexp/({E_s}^nexp + Kex^nexp)",
                                    propensity_species=[G, E_s],
                                    propensity_parameters=[nexp, kex, Kex, Gmin, Gmax])
    # optional modification to add waste inhibition of TXTL
    if waste_tl_inhibition == True:
        KB = bcrn.ParameterEntry("KB", KB_val*concentration_multiplier)
        nB = bcrn.ParameterEntry("nB", nB_val)
        prop_ex = bcrn.GeneralPropensity(f"kex * {G}/({G} + Gmin) * Gmax/({G} + Gmax) * {E_s}^nexp/({E_s}^nexp + Kex^nexp) * 1/(1+(({B})/KB)^nB)",
                                        propensity_species=[G, E_s, B],
                                        propensity_parameters=[nexp, kex, Kex, KB, nB, Gmin, Gmax])
    reactions_ex.append(bcrn.Reaction(inputs=[G], outputs=[G, X], propensity_type = prop_ex))

    # ATP regeneration reaction
    vmaxE = bcrn.ParameterEntry("vmaxE", argmax_params["vmaxATP"]*concentration_multiplier) #Previously vmaxATP
    KF= bcrn.ParameterEntry('KF', argmax_params["Kpga"]*concentration_multiplier) #Previously Kpga
    K_Ed = bcrn.ParameterEntry("K_Ed", argmax_params["Kndp"]*concentration_multiplier) #previously Kndp
    if KW_val is None:
        KW = bcrn.ParameterEntry("KW", argmax_params["KW"]*concentration_multiplier)
    else:
        KW = bcrn.ParameterEntry("KW", KW_val*concentration_multiplier)
    # optional different nW value
    if nW_val is None:
        nW = bcrn.ParameterEntry("nW", argmax_params["nW"])
    else:
        nW = bcrn.ParameterEntry("nW", nW_val)
    prop = bcrn.GeneralPropensity(f"vmaxE * {F}/(KF+{F}) * {E_d}/(K_Ed+{E_d}) * 1/(1+({W}/KW)^nW)", 
                            propensity_species = [E_d, F, W], 
                            propensity_parameters = [vmaxE, KF, nW, K_Ed, KW])
    reactions_E_W.append(bcrn.Reaction(inputs=[F]+3*[E_d], outputs=[W]+3*[E_s], propensity_type = prop))

    # 3PGA leak reaction
    KFW = bcrn.ParameterEntry("KFW", argmax_params["KpgaW__"]*concentration_multiplier) #Previously KpgaW
    vmaxW = bcrn.ParameterEntry("vmaxW", argmax_params["vmaxW__"]*concentration_multiplier) 
    nW_leak = bcrn.ParameterEntry("nW_leak", argmax_params["nW_leak__"])
    prop = bcrn.HillPositive(k = vmaxW, s1 = F, K = KFW, n = nW_leak)
    reactions_E_W.append(bcrn.Reaction([F], [W], propensity_type = prop))

    # ATP degredation reaction
    vmaxDEG = bcrn.ParameterEntry("vmaxDeg", argmax_params["vmaxDeg__"]*concentration_multiplier)
    KEdeg = bcrn.ParameterEntry("KEdeg", argmax_params["Kntpdeg__"]*concentration_multiplier) #previously Kntpdeg
    prop = bcrn.HillPositive(k = vmaxDEG, s1 = E_s, K = KEdeg, n = 1)
    reactions_E_W.append(bcrn.Reaction([E_s], [E_d], propensity_type = prop))

    # waste chelation reaction
    if waste_chelation == True:
        rate_adjustment = concentration_multiplier#**W_B_binding_stoich
        k_buffer_b = bcrn.ParameterEntry("k_buffer_b", .1/rate_adjustment)
        k_buffer_u = bcrn.ParameterEntry("k_buffer_u", .1)
        reactions_E_W.append(bcrn.Reaction.from_massaction([B] + [W]*W_B_binding_stoich, [BW], k_forward = k_buffer_b, k_reverse = k_buffer_u))

    # combine reaction lists
    reactions = reactions_ex + reactions_E_W

    # define initial concentrations
    initial_concentration = {G:G0, 
                            E_s:E_s_0, 
                            F:F0, 
                            W:W0, 
                            B:10*10**12}

    # create chemical reaction network
    CRN = bcrn.ChemicalReactionNetwork(species = species, reactions = reactions, initial_concentration_dict = initial_concentration)

    return(CRN)
















# create function that takes in parameter set to create sbml model
def create_model_for_this_param_set_dna_saturation2(argmax_params, 
                                                    waste_tl_inhibition=False, 
                                                    waste_chelation=False, 
                                                    KB_val=4000, 
                                                    nB_val = 2.0, 
                                                    W_B_binding_stoich = 1, 
                                                    nW_val=None, 
                                                    KW_val=None):

    # create species
    G = bcrn.Species("G")       # Gene
    X = bcrn.Species("GFP")     # Protein
    E_s = bcrn.Species("E_s")   # NTPs
    E_d = bcrn.Species("E_d")   # NDPs
    F = bcrn.Species("F")       # 3PGA
    W = bcrn.Species('W')       # Waste

    # create additional species
    W_B_binding_stoich = W_B_binding_stoich
    B = bcrn.Species("B")       # Buffer
    BW = bcrn.Complex([B] + [W]*W_B_binding_stoich)   # Buffer Waste Complex
    species = [G, X, E_s, E_d, F, W, B, BW]

    # define initial concentrations
    G0 = 2.3*10**6                        # 10^-14 M; 48ng/uL = 2.3 nM
    E_s_0 = 5*10**12                      # 10^-14 M; previous ntp0
    F0 = 30*10**12                        # 10^-14 M; previous pga0
    W0 = 0
    concentration_multiplier = 10**9

    # create lists to store expression reactions and energy/waste reactions
    reactions_ex = []
    reactions_E_W = []

    # TX-TL reaction
    kex = bcrn.ParameterEntry("kex", argmax_params["kex__"]*10**6)
    Kex = bcrn.ParameterEntry("Kex", argmax_params["Kex__"]*concentration_multiplier)
    nexp = bcrn.ParameterEntry("nexp", argmax_params["nexp__"])
    Gmin = bcrn.ParameterEntry('Gmin', 5.00001*10**6)
    Gmax = bcrn.ParameterEntry('Gmax', 7.000001*10**6)
    nmin = bcrn.ParameterEntry('nmin', 3)
    nmax = bcrn.ParameterEntry('nmax', 2)
    prop_ex = bcrn.GeneralPropensity(f"kex * {G}^nmin/({G}^nmin + Gmin^nmin) * Gmax^nmax/({G}^nmax + Gmax^nmax) * {E_s}^nexp/({E_s}^nexp + Kex^nexp)",
                                    propensity_species=[G, E_s],
                                    propensity_parameters=[nexp, kex, Kex, Gmin, Gmax, nmin, nmax])
    # optional modification to add waste inhibition of TXTL
    if waste_tl_inhibition == True:
        KB = bcrn.ParameterEntry("KB", KB_val*concentration_multiplier)
        nB = bcrn.ParameterEntry("nB", nB_val)
        prop_ex = bcrn.GeneralPropensity(f"kex * {G}^nmin/({G}^nmin + Gmin^nmin) * Gmax^nmax/({G}^nmax + Gmax^nmax) * {E_s}^nexp/({E_s}^nexp + Kex^nexp) * 1/(1+(({B})/KB)^nB)",
                                        propensity_species=[G, E_s, B],
                                        propensity_parameters=[nexp, kex, Kex, KB, nB, Gmin, Gmax, nmin, nmax])
    reactions_ex.append(bcrn.Reaction(inputs=[G], outputs=[G, X], propensity_type = prop_ex))

    # ATP regeneration reaction
    vmaxE = bcrn.ParameterEntry("vmaxE", argmax_params["vmaxATP"]*concentration_multiplier) #Previously vmaxATP
    KF= bcrn.ParameterEntry('KF', argmax_params["Kpga"]*concentration_multiplier) #Previously Kpga
    K_Ed = bcrn.ParameterEntry("K_Ed", argmax_params["Kndp"]*concentration_multiplier) #previously Kndp
    if KW_val is None:
        KW = bcrn.ParameterEntry("KW", argmax_params["KW"]*concentration_multiplier)
    else:
        KW = bcrn.ParameterEntry("KW", KW_val*concentration_multiplier)
    # optional different nW value
    if nW_val is None:
        nW = bcrn.ParameterEntry("nW", argmax_params["nW"])
    else:
        nW = bcrn.ParameterEntry("nW", nW_val)
    prop = bcrn.GeneralPropensity(f"vmaxE * {F}/(KF+{F}) * {E_d}/(K_Ed+{E_d}) * 1/(1+({W}/KW)^nW)", 
                            propensity_species = [E_d, F, W], 
                            propensity_parameters = [vmaxE, KF, nW, K_Ed, KW])
    reactions_E_W.append(bcrn.Reaction(inputs=[F]+3*[E_d], outputs=[W]+3*[E_s], propensity_type = prop))

    # 3PGA leak reaction
    KFW = bcrn.ParameterEntry("KFW", argmax_params["KpgaW__"]*concentration_multiplier) #Previously KpgaW
    vmaxW = bcrn.ParameterEntry("vmaxW", argmax_params["vmaxW__"]*concentration_multiplier) 
    nW_leak = bcrn.ParameterEntry("nW_leak", argmax_params["nW_leak__"])
    prop = bcrn.HillPositive(k = vmaxW, s1 = F, K = KFW, n = nW_leak)
    reactions_E_W.append(bcrn.Reaction([F], [W], propensity_type = prop))

    # ATP degredation reaction
    vmaxDEG = bcrn.ParameterEntry("vmaxDeg", argmax_params["vmaxDeg__"]*concentration_multiplier)
    KEdeg = bcrn.ParameterEntry("KEdeg", argmax_params["Kntpdeg__"]*concentration_multiplier) #previously Kntpdeg
    prop = bcrn.HillPositive(k = vmaxDEG, s1 = E_s, K = KEdeg, n = 1)
    reactions_E_W.append(bcrn.Reaction([E_s], [E_d], propensity_type = prop))

    # waste chelation reaction
    if waste_chelation == True:
        rate_adjustment = concentration_multiplier#**W_B_binding_stoich
        k_buffer_b = bcrn.ParameterEntry("k_buffer_b", .1/rate_adjustment)
        k_buffer_u = bcrn.ParameterEntry("k_buffer_u", .1)
        reactions_E_W.append(bcrn.Reaction.from_massaction([B] + [W]*W_B_binding_stoich, [BW], k_forward = k_buffer_b, k_reverse = k_buffer_u))

    # combine reaction lists
    reactions = reactions_ex + reactions_E_W

    # define initial concentrations
    initial_concentration = {G:G0, 
                            E_s:E_s_0, 
                            F:F0, 
                            W:W0, 
                            B:10*10**12}

    # create chemical reaction network
    CRN = bcrn.ChemicalReactionNetwork(species = species, reactions = reactions, initial_concentration_dict = initial_concentration)

    return(CRN)




# We create a new function to plot and save plot
def compute_naive_mase_value_single_initial_condition(model,  
                                                        this_init_cond, 
                                                        this_exp_data,
                                                        ):

    # load model
    M_fit = model

    # format experimental data
    timepoints = np.array(this_exp_data['Time (sec)'])
    exp_data_y = this_exp_data['GFP']/10**9

    # set initial conditions
    M_fit.set_species(this_init_cond)
    
    # simulate model
    R = py_simulate_model(timepoints, Model= M_fit)
    pred_y = R['GFP']/10**9

    # compute error metric
    this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)

    return(this_mase)


# We create a new function to plot and save plot
def compute_naive_mase_values_multiple_dna_initial_conditions(model, 
                                                                init_cond_list, 
                                                                exp_data_list,
                                                                ):

    # load model 
    M_fit = model

    # iterate through 5 dna concs
    mase_values = []
    for j in range(0,5):

        # get initial condition, expt data
        this_init_cond = init_cond_list[j]
        this_exp_data = exp_data_list[j]
    
        # format experimental data
        timepoints = np.array(this_exp_data['Time (sec)'])
        exp_data_y = this_exp_data['GFP']/10**9
    
        # set initial conditions
        M_fit.set_species(this_init_cond)
            
        # plot trajectory
        R = py_simulate_model(timepoints, Model= M_fit)
        pred_y = R['GFP']/10**9

        # compute error metric
        this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)
        mase_values.append(this_mase)

    return(mase_values)




# We create a new function to plot and save plot
def plot_and_save_mcmc_simulation_single_initial_condition(model, 
                                                            filename_for_export,
                                                            pid, 
                                                            sampler, 
                                                            this_init_cond, 
                                                            this_exp_data,
                                                            discard=500, 
                                                            timepoints_correction=False
                                                            ):

    # load model and parameter sets
    M_fit = model
    if timepoints_correction==True:
        timepoints=pid.timepoints[0]
    else:
        timepoints = pid.timepoints
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    # inds = np.random.randint(len(flat_samples), size=100)
    inds = np.arange(0,100)

    # initialize bokeh plot
    # p = bokeh.plotting.figure(width=1000, height=1000)
    p = bokeh.plotting.figure(width=1050, height=900)

    # initialize MASE list
    mase_values = []

    # format experimental data
    exp_data_x = this_exp_data['Time (sec)']/3600
    exp_data_y = this_exp_data['GFP']/10**9
    
    # set model initial conditions
    M_fit.set_species(this_init_cond)
    
    # plot trajectory for base model
    R = py_simulate_model(timepoints, Model= M_fit)
    pred_x = timepoints/3600
    pred_y = R['GFP']/10**9
    p.line(pred_x, 
            pred_y, 
            alpha=1, 
            line_width=36,
            legend_label = 'Model (base)',
           line_dash = 'dashed',
            color='#9ecae1'
            )

    # iterate through specific initial conditions, do simulation, compute mase values
    for ind in inds:
        sample = flat_samples[ind]
        
        # set parameters for this model
        for pi, pi_val in zip(pid.params_to_estimate, sample):
            M_fit.set_parameter(pi, pi_val)
        
        # plot trajectory
        R = py_simulate_model(timepoints, Model= M_fit)
        pred_x = timepoints/3600
        pred_y = R['GFP']/10**9
        # p.line(pred_x, 
        #         pred_y, 
        #         alpha=0.3, 
        #         line_width=6,
        #         legend_label = 'simulation',
        #         color='#9ecae1'
        #         )

        # compute error metric
        this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)
        mase_values.append(this_mase)

    # determine which trajectories do not have an insanely high error, plot those simulations
    mase_array = np.array(mase_values)
    indx_to_plot = indices = [ind for ind, val in enumerate(mase_array) if val < np.median(mase_array)*1.2]
    for this_ind in indx_to_plot:
        
        sample = flat_samples[this_ind]
        
        # set parameters for this model, initial conditions
        for pi, pi_val in zip(pid.params_to_estimate, sample):
            M_fit.set_parameter(pi, pi_val)
        M_fit.set_species(this_init_cond)
        
        # plot trajectory
        R = py_simulate_model(timepoints, Model= M_fit)
        pred_x = timepoints/3600
        pred_y = R['GFP']/10**9
        p.line(pred_x, 
                pred_y, 
                alpha=0.3, 
                line_width=36,
                legend_label = 'Model (fine-tuned)',
                color='#9ecae1'
                )

    # plot experimental data
    p.scatter(exp_data_x,
         exp_data_y,
         color='#084594',#'orange',
         size=15,
         # fill_alpha=0.5,
         # line_alpha=0.5,
         legend_label = 'Experimental'
        )


    # customize plots
    p.yaxis.axis_label = 'GFP (uM)'
    p.xaxis.axis_label = 'Time (hrs)'

    # additional customizations
    p.xaxis.major_label_text_font_size = '50pt'
    p.xaxis.axis_label_text_font_size = '50pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.axis_label_standoff = 45

    p.yaxis.major_label_text_font_size = '50pt'
    p.yaxis.axis_label_text_font_size = '50pt'
    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.axis_label_standoff = 45

    p.legend.label_text_font_size = '30pt'
    p.legend.title_text_font_size = '36pt'
    p.legend.title_text_font_style = 'normal'

    # # p.add_layout(p.legend[0], 'right')
    p.legend.location = 'bottom_right'

    # # additional customizations
    # p.xaxis.major_label_text_font_size = '33pt'
    # p.xaxis.axis_label_text_font_size = '33pt'
    # p.xaxis.axis_label_text_font_style = 'normal'
    # p.xaxis.axis_label_standoff = 30

    # p.yaxis.major_label_text_font_size = '33pt'
    # p.yaxis.axis_label_text_font_size = '33pt'
    # p.yaxis.axis_label_text_font_style = 'normal'
    # p.yaxis.axis_label_standoff = 30

    # p.legend.label_text_font_size = '30pt'
    # p.legend.title_text_font_size = '33pt'
    # p.legend.title_text_font_style = 'normal'
    # p.legend.padding = 20
    # p.legend.spacing = 6
    # p.legend.glyph_height = 30
    # p.legend.glyph_width = 30

    # p.add_layout(p.legend[0], 'right')
    
    p.legend.title = 'Data'
    p.toolbar_location = None 

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None


    bokeh.io.export_png(p, filename=filename_for_export, webdriver=webdriver)

    # return(mase_values)




# We create a new function to plot and save plot
def plot_and_save_mcmc_simulation_multiple_dna_initial_conditions(model, 
                                                                filename_for_export,
                                                                pid, 
                                                                sampler, 
                                                                init_cond_list, 
                                                                exp_data_list,
                                                                discard=500, 
                                                                timepoints_correction=False
                                                                ):

    # load model and parameter sets
    M_fit = model
    if timepoints_correction==True:
        timepoints=pid.timepoints[0]
    else:
        timepoints = pid.timepoints
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    # inds = np.random.randint(len(flat_samples), size=100)
    inds = np.arange(0,100)

    # initialize bokeh plot
    if (init_cond_list[0]['F']==45*10**12):
        p = bokeh.plotting.figure(width=1050, height=715)
    else:
        p = bokeh.plotting.figure(width=800, height=715)
    # colors = bokeh.palettes.Viridis5
    colors = [
          bokeh.palettes.Colorblind8[6],
          bokeh.palettes.Colorblind8[0], 
          bokeh.palettes.Colorblind8[1], 
          bokeh.palettes.Colorblind8[3], 
          bokeh.palettes.Colorblind8[5], 
         ]

    # iterate through 5 dna concs
    mase_values = []
    for j in range(0,5):

        # get initial condition, expt data
        this_init_cond = init_cond_list[j]
        this_exp_data = exp_data_list[j]

        # initialize MASE list
        temp_mase_values = []
    
        # format experimental data
        exp_data_x = this_exp_data['Time (sec)']/3600
        exp_data_y = this_exp_data['GFP']/10**9
    
        # iterate through specific initial conditions, do simulation, compute mase values
        for ind in inds:
            sample = flat_samples[ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            
            # plot trajectory
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R['GFP']/10**9
            # p.line(pred_x, 
            #         pred_y, 
            #         alpha=0.3, 
            #         line_width=6,
            #         legend_label = 'simulation',
            #         color='#9ecae1'
            #         )
    
            # compute error metric
            this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)
            temp_mase_values.append(this_mase)
            mase_values.append(this_mase)
    
        # determine which trajectories do not have an insanely high error, plot those simulations
        mase_array = np.array(temp_mase_values)
        indx_to_plot = indices = [ind for ind, val in enumerate(mase_array) if val < np.median(mase_array)*1.2]
        for this_ind in indx_to_plot:
            
            sample = flat_samples[this_ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            
            # plot trajectory
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R['GFP']/10**9
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=16,
                    # legend_label = 'simulation',
                    color=colors[j]
                    )

    # now plot experimental data
    for j in range(0,5):
    
        # get initial condition, expt data
        this_init_cond = init_cond_list[j]
        this_exp_data = exp_data_list[j]

        # format experimental data
        exp_data_x = this_exp_data['Time (sec)']/3600
        exp_data_y = this_exp_data['GFP']/10**9

        # plot experimental data
        dna_conc = this_init_cond['G']/10**6
        if (this_init_cond['F']==45*10**12):
            p.scatter(exp_data_x,
                      exp_data_y,
                      color=colors[j],
                      size=10,
                 # fill_alpha=0.5,
                 # line_alpha=0.5,
                 legend_label = f'{dna_conc}'
                )
        else:
            p.scatter(exp_data_x,
                      exp_data_y,
                      color=colors[j],
                      size=10,
                 # fill_alpha=0.5,
                 # line_alpha=0.5,
                 # legend_label = f'{dna_conc}'
                )         
    
    # customize plots
    p.yaxis.axis_label = 'GFP (uM)'
    p.xaxis.axis_label = 'Time (hrs)'

    # additional customizations
    # p.xaxis.major_label_text_font_size = '33pt'
    # p.xaxis.axis_label_text_font_size = '33pt'
    # p.xaxis.axis_label_text_font_style = 'normal'
    # p.xaxis.axis_label_standoff = 30

    # p.yaxis.major_label_text_font_size = '33pt'
    # p.yaxis.axis_label_text_font_size = '33pt'
    # p.yaxis.axis_label_text_font_style = 'normal'
    # p.yaxis.axis_label_standoff = 30

    # p.legend.label_text_font_size = '30pt'
    # p.legend.title_text_font_size = '33pt'
    # p.legend.title_text_font_style = 'normal'
    
    p.xaxis.major_label_text_font_size = '50pt'
    p.xaxis.axis_label_text_font_size = '50pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.axis_label_standoff = 45

    p.yaxis.major_label_text_font_size = '50pt'
    p.yaxis.axis_label_text_font_size = '50pt'
    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.axis_label_standoff = 45

    if (this_init_cond['F']==45*10**12):
        p.legend.label_text_font_size = '30pt'
        p.legend.title_text_font_size = '36pt'
        p.legend.title_text_font_style = 'normal'
    
        
        p.legend.padding = 20
        p.legend.spacing = 6
        p.legend.glyph_height = 30
        p.legend.glyph_width = 30
    
        p.add_layout(p.legend[0], 'right')
        p.legend.title = 'DNA (nM)'
    p.toolbar_location = None 

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    bokeh.io.export_png(p, filename=filename_for_export, webdriver=webdriver)



# We create a new function to plot and save plot
def plot_and_save_mcmc_simulation_multiple_dna_initial_conditions_other_species(model, 
                                                                                filename_for_export,
                                                                                pid, 
                                                                                sampler, 
                                                                                init_cond_list, 
                                                                                exp_data_list,
                                                                                species,
                                                                                discard=500, 
                                                                                timepoints_correction=False,
                                                                                ):

    # load model and parameter sets
    M_fit = model
    if timepoints_correction==True:
        timepoints=pid.timepoints[0]
    else:
        timepoints = pid.timepoints
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    # inds = np.random.randint(len(flat_samples), size=100)
    inds = np.arange(0,100)

    # initialize bokeh plot
    p = bokeh.plotting.figure(width=1050, height=715)
    # colors = bokeh.palettes.Viridis5
    colors = [
          bokeh.palettes.Colorblind8[6],
          bokeh.palettes.Colorblind8[0], 
          bokeh.palettes.Colorblind8[1], 
          bokeh.palettes.Colorblind8[3], 
          bokeh.palettes.Colorblind8[5], 
         ]

    # iterate through 5 dna concs
    mase_values = []
    for j in range(0,5):

        # get initial condition, expt data
        this_init_cond = init_cond_list[j]
        this_exp_data = exp_data_list[j]

        # initialize MASE list
        temp_mase_values = []
    
        # format experimental data
        exp_data_x = this_exp_data['Time (sec)']/3600
        exp_data_y = this_exp_data['GFP']/10**9
    
        # iterate through specific initial conditions, do simulation, compute mase values
        for ind in inds:
            sample = flat_samples[ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            
            # plot trajectory
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R['GFP']/10**9
    
            # compute error metric
            this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)
            temp_mase_values.append(this_mase)
            mase_values.append(this_mase)
    
        # determine which trajectories do not have an insanely high error, plot those simulations
        mase_array = np.array(temp_mase_values)
        indx_to_plot = indices = [ind for ind, val in enumerate(mase_array) if val < np.median(mase_array)*1.2]
        for this_ind in indx_to_plot:
            
            sample = flat_samples[this_ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            
            # plot trajectory
            dna_conc = this_init_cond['G']/10**6
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R[species]/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=16,
                    color=colors[j],
                    legend_label = f'{dna_conc}'
                    )

    
    # customize plots
    if species == 'F':
        species_name = 'Fuel'
    elif species == 'W':
        species_name = 'Waste'
    elif species == 'E_s':
        species_name = 'Energy'
    p.yaxis.axis_label = f'{species_name} (mM)'
    p.xaxis.axis_label = 'Time (hrs)'

    # additional customizations
    p.xaxis.major_label_text_font_size = '33pt'
    p.xaxis.axis_label_text_font_size = '33pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.axis_label_standoff = 30

    p.yaxis.major_label_text_font_size = '33pt'
    p.yaxis.axis_label_text_font_size = '33pt'
    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.axis_label_standoff = 30

    p.legend.label_text_font_size = '30pt'
    p.legend.title_text_font_size = '33pt'
    p.legend.title_text_font_style = 'normal'
    p.legend.padding = 20
    p.legend.spacing = 6
    p.legend.glyph_height = 30
    p.legend.glyph_width = 30

    p.add_layout(p.legend[0], 'right')
    p.legend.title = 'DNA (nM)'
    p.toolbar_location = None 

    bokeh.io.export_png(p, filename=filename_for_export, webdriver=webdriver)




# We create a new function to plot and save plot
def plot_and_save_mcmc_simulation_multiple_mg_initial_conditions(model, 
                                                                filename_for_export,
                                                                pid, 
                                                                sampler, 
                                                                init_cond_list, 
                                                                exp_data_list,
                                                                discard=500, 
                                                                timepoints_correction=False
                                                                ):

    # load model and parameter sets
    M_fit = model
    if timepoints_correction==True:
        timepoints=pid.timepoints[0]
    else:
        timepoints = pid.timepoints
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    # inds = np.random.randint(len(flat_samples), size=100)
    inds = np.arange(0,100)

    # initialize bokeh plot
    p = bokeh.plotting.figure(width=1050, height=715)
    # colors = bokeh.palettes.Viridis5
    colors = [
          bokeh.palettes.Colorblind8[6],
          bokeh.palettes.Colorblind8[0], 
          bokeh.palettes.Colorblind8[1], 
          bokeh.palettes.Colorblind8[3], 
          bokeh.palettes.Colorblind8[5], 
          bokeh.palettes.Colorblind8[4] 
         ]

    # iterate through 5 dna concs
    mase_values = []
    for j in range(0,6):

        # get initial condition, expt data
        this_init_cond = init_cond_list[j]
        this_exp_data = exp_data_list[j]

        # initialize MASE list
        temp_mase_values = []
    
        # format experimental data
        exp_data_x = this_exp_data['Time (sec)']/3600
        exp_data_y = this_exp_data['GFP']/10**9
    
        # iterate through specific initial conditions, do simulation, compute mase values
        for ind in inds:
            sample = flat_samples[ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            
            # plot trajectory
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R['GFP']/10**9
    
            # compute error metric
            this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)
            temp_mase_values.append(this_mase)
            mase_values.append(this_mase)
    
        # determine which trajectories do not have an insanely high error, plot those simulations
        mase_array = np.array(temp_mase_values)
        indx_to_plot = [ind for ind, val in enumerate(mase_array) if val < np.median(mase_array)*1.2]
        for this_ind in indx_to_plot:
            
            sample = flat_samples[this_ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            
            # plot trajectory
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R['GFP']/10**9
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[j]
                    )

    # now plot experimental data
    for j in range(0,6):
    
        # get initial condition, expt data
        this_init_cond = init_cond_list[j]
        this_exp_data = exp_data_list[j]

        # format experimental data
        exp_data_x = this_exp_data['Time (sec)']/3600
        exp_data_y = this_exp_data['GFP']/10**9

        # plot experimental data
        mg_conc = this_init_cond['B']/10**12
        p.scatter(exp_data_x,
             exp_data_y,
             color=colors[j],
             size=5,
             legend_label = f'{mg_conc}'
            )

    # customize plots
    p.yaxis.axis_label = 'GFP (uM)'
    p.xaxis.axis_label = 'Time (hrs)'

    # additional customizations
    p.xaxis.major_label_text_font_size = '33pt'
    p.xaxis.axis_label_text_font_size = '33pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.axis_label_standoff = 30

    p.yaxis.major_label_text_font_size = '33pt'
    p.yaxis.axis_label_text_font_size = '33pt'
    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.axis_label_standoff = 30

    p.legend.label_text_font_size = '30pt'
    p.legend.title_text_font_size = '33pt'
    p.legend.title_text_font_style = 'normal'
    p.legend.padding = 20
    p.legend.spacing = 6
    p.legend.glyph_height = 30
    p.legend.glyph_width = 30

    p.add_layout(p.legend[0], 'right')
    p.legend.title = 'Mg (mM)'
    p.toolbar_location = None 

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    bokeh.io.export_png(p, filename=filename_for_export, webdriver=webdriver)




# We create a new function to plot and save plot
def plot_and_save_mcmc_simulation_single_mg_initial_condition_FWE(model, 
                                                                filename_for_export,
                                                                pid, 
                                                                sampler, 
                                                                this_init_cond, 
                                                                this_exp_data,
                                                                discard=500, 
                                                                timepoints_correction=False
                                                                ):

    # load model and parameter sets
    M_fit = model
    if timepoints_correction==True:
        timepoints=pid.timepoints[0]
    else:
        timepoints = pid.timepoints
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    # inds = np.random.randint(len(flat_samples), size=100)
    inds = np.arange(0,100)

    # initialize bokeh plot
    if this_init_cond['F']==45*10**12:
        p = bokeh.plotting.figure(width=1050, height=715, y_range=(-2,46))
    else:
        p = bokeh.plotting.figure(width=800, height=715, y_range=(-2,46))
    # colors = bokeh.palettes.Viridis5
    colors = [
          bokeh.palettes.Colorblind8[6],
          bokeh.palettes.Colorblind8[0], 
          bokeh.palettes.Colorblind8[1], 
          bokeh.palettes.Colorblind8[3], 
          bokeh.palettes.Colorblind8[5], 
          bokeh.palettes.Colorblind8[4] 
         ]

    # iterate through 5 dna concs
    mase_values = []

    # initialize MASE list
    temp_mase_values = []

    # format experimental data
    exp_data_x = this_exp_data['Time (sec)']/3600
    exp_data_y = this_exp_data['GFP']/10**9

    # iterate through specific initial conditions, do simulation, compute mase values
    for ind in inds:
        sample = flat_samples[ind]
        
        # set parameters for this model, initial conditions
        for pi, pi_val in zip(pid.params_to_estimate, sample):
            M_fit.set_parameter(pi, pi_val)
        M_fit.set_species(this_init_cond)
        
        # plot trajectory
        R = py_simulate_model(timepoints, Model= M_fit)
        pred_x = timepoints/3600
        pred_y = R['GFP']/10**9

        # compute error metric
        this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)
        temp_mase_values.append(this_mase)
        mase_values.append(this_mase)

    # determine which trajectories do not have an insanely high error, plot those simulations
    mase_array = np.array(temp_mase_values)
    indx_to_plot = [ind for ind, val in enumerate(mase_array)]# if val < np.median(mase_array)*5] #1.2
    for this_ind in indx_to_plot:
        
        sample = flat_samples[this_ind]
        
        # set parameters for this model, initial conditions
        for pi, pi_val in zip(pid.params_to_estimate, sample):
            M_fit.set_parameter(pi, pi_val)
        M_fit.set_species(this_init_cond)
        
        # plot trajectory
        R = py_simulate_model(timepoints, Model= M_fit)
        pred_x = timepoints/3600
        pred_y = R['F']/10**12

        if this_init_cond['F']==45*10**12:
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[0],
                   legend_label = 'Fuel'
                    )
            pred_y = R['W']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[1],
                   legend_label = 'Waste'
                    )
            pred_y = R['E_s']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[2],
                   legend_label = 'Energy'
                    )
            pred_y = R['complex_B_W_4x_']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[3],
                   legend_label = 'Mg_W_complex'
                    )
            pred_y = R['B']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[4],
                   legend_label = 'Mg'
                    )
        else:
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[0],
                   # legend_label = 'Fuel'
                    )
            pred_y = R['W']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[1],
                   # legend_label = 'Waste'
                    )
            pred_y = R['E_s']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[2],
                   # legend_label = 'Energy'
                    )
            pred_y = R['complex_B_W_4x_']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[3],
                   # legend_label = 'Mg_W_complex'
                    )
            pred_y = R['B']/10**12
            p.line(pred_x, 
                    pred_y, 
                    alpha=0.01, 
                    line_width=12,
                    color=colors[4],
                   # legend_label = 'Mg'
                    )        

    # customize plots
    p.yaxis.axis_label = 'Concentration (μM)'
    p.xaxis.axis_label = 'Time (hrs)'

    # additional customizations
    p.xaxis.major_label_text_font_size = '33pt'
    p.xaxis.axis_label_text_font_size = '33pt'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.axis_label_standoff = 30

    p.yaxis.major_label_text_font_size = '33pt'
    p.yaxis.axis_label_text_font_size = '33pt'
    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.axis_label_standoff = 30

    if (this_init_cond['F']==45*10**12):

        p.legend.label_text_font_size = '30pt'
        p.legend.title_text_font_size = '33pt'
        p.legend.title_text_font_style = 'normal'
        p.legend.padding = 20
        p.legend.spacing = 6
        p.legend.glyph_height = 30
        p.legend.glyph_width = 30
    
        p.add_layout(p.legend[0], 'right')
        p.legend.title = 'Species'
        
    p.toolbar_location = None 

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    bokeh.io.export_png(p, filename=filename_for_export, webdriver=webdriver)



# We create a new function to plot and save plot
def summarize_degfp_endpoint(model, 
                            filename_for_export,
                            pid, 
                            sampler, 
                            init_cond_list, 
                            exp_data_list,
                            discard=500, 
                            timepoints_correction=False
                            ):

    # initialize lists to hold info
    mg_conc_list = []
    fuel_conc_list = []
    max_degfp = []

    # load model and parameter sets
    M_fit = model
    if timepoints_correction==True:
        timepoints=pid.timepoints[0]
    else:
        timepoints = pid.timepoints
    flat_samples = sampler.get_chain(discard=discard, flat=True)
    # inds = np.random.randint(len(flat_samples), size=100)
    inds = np.arange(0,100)

    # initialize bokeh plot
    p = bokeh.plotting.figure(width=700, height=500)
    # colors = bokeh.palettes.Viridis5
    colors = [
          bokeh.palettes.Colorblind8[6],
          bokeh.palettes.Colorblind8[0], 
          bokeh.palettes.Colorblind8[1], 
          bokeh.palettes.Colorblind8[3], 
          bokeh.palettes.Colorblind8[5], 
          bokeh.palettes.Colorblind8[4] 
         ]

    # iterate through 5 dna concs
    mase_values = []
    for j in range(0,6):

        # get initial condition, expt data
        this_init_cond = init_cond_list[j]
        this_exp_data = exp_data_list[j]

        # initialize MASE list
        temp_mase_values = []
    
        # format experimental data
        exp_data_x = this_exp_data['Time (sec)']/3600
        exp_data_y = this_exp_data['GFP']/10**9
    
        # iterate through specific initial conditions, do simulation, compute mase values
        for ind in inds:
            sample = flat_samples[ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            
            # simulate trajectory
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R['GFP']/10**9
    
            # compute error metric
            this_mase = mean_absolute_scaled_error(exp_data_y, pred_y)
            temp_mase_values.append(this_mase)
            mase_values.append(this_mase)
    
        # determine which trajectories do not have an insanely high error, plot those simulations
        mase_array = np.array(temp_mase_values)
        indx_to_plot = [ind for ind, val in enumerate(mase_array)] #S if val < np.median(mase_array)*12]
        for this_ind in indx_to_plot:
            
            sample = flat_samples[this_ind]
            
            # set parameters for this model, initial conditions
            for pi, pi_val in zip(pid.params_to_estimate, sample):
                M_fit.set_parameter(pi, pi_val)
            M_fit.set_species(this_init_cond)
            fuel_conc_list.append(this_init_cond['F']/10**12)
            mg_conc_list.append(this_init_cond['B']/10**12)
            
            # plot trajectory
            R = py_simulate_model(timepoints, Model= M_fit)
            pred_x = timepoints/3600
            pred_y = R['GFP']/10**9
            max_degfp.append(np.max(pred_y))


    # create df to summarize data
    data = {'Fuel (mM)': fuel_conc_list,
            'Mg (mM)': mg_conc_list,
            'Maximum deGFP (μM)': max_degfp}
    df = pd.DataFrame(data)
    
    return(df)








def get_dict_of_background_subtract_functions(master_df):

    # get relevant degfp background subtraction data
    # ar_zj lysate
    arzj_bkgd_samples = master_df.loc[(master_df['DNA (nM)']==0) &
                          (master_df['DNA/mRNA batch'] < 5) &
                          (master_df['Extract']=='AR_ZJ') & 
                          (master_df['Fuel (mM)']==0) & 
                          (master_df['Mg (mM)']==0) & 
                          (master_df['Plasmid construct']=='pOR1OR2-MGapt-deGFP') & 
                          (master_df['mRNA (uM)'] == 0) & 
                          (master_df['Channel'] == 'deGFP')
                        ]
    x = arzj_bkgd_samples['Time (hr)']
    y = arzj_bkgd_samples['Measurement']
    fn = np.polyfit(x, y, 5)
    arzj_degfp_fn = np.poly1d(fn)
    
    # daj_mk_my lysate
    dajmkmy_bkgd_samples = master_df.loc[(master_df['DNA (nM)']==0) &
                          (master_df['DNA/mRNA batch'] < 5) &
                          (master_df['Extract']=='DAJ_MK_MY') & 
                          (master_df['Fuel (mM)']==0) & 
                          (master_df['Mg (mM)']==0) & 
                          (master_df['Plasmid construct']=='pOR1OR2-MGapt-deGFP') & 
                          (master_df['mRNA (uM)'] == 0) & 
                          (master_df['Channel'] == 'deGFP')
                        ]
    x = dajmkmy_bkgd_samples['Time (hr)']
    y = dajmkmy_bkgd_samples['Measurement']
    fn = np.polyfit(x, y, 5)
    dajmkmy_degfp_fn = np.poly1d(fn)
            
    # create dictionary to store functions
    degfp_bkgd_subt_dict = {'AR_ZJ': arzj_degfp_fn,
                            'DAJ_MK_MY': dajmkmy_degfp_fn
                            }

    return(degfp_bkgd_subt_dict)




def mean_absolute_scaled_error(y_true, y_pred):
    
    # Ensure both inputs are numpy arrays for easy manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the numerator (Mean Absolute Error between predictions and true values)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate the mean of the true values (base forecast)
    mean_y = np.mean(y_true)
    
    # Calculate the denominator (Mean Absolute Error of the true values from the mean)
    scale = np.mean(np.abs(y_true - mean_y))
    
    # MASE calculation
    if scale == 0:
        return np.nan  # Return NaN if scale is zero to avoid division by zero
        
    return mae / scale
