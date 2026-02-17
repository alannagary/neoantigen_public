import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from neoantigen_functions import *
import pandas as pd
import dill
import seaborn as sns
from scipy import stats
from statannot import add_stat_annotation
from cycler import cycler
from scipy.stats import binomtest
import os

# Variables you need to set:
path_to_ARdata = 'C:/Users/Alanna/Desktop/Research_Code/Desktop_research/AxR_data.txt' # Path to AxR_data.txt
path_to_MMRD_data = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_oct_24/MMRD/'
path_to_MMRP_data = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_oct_24/MMRP/'
path_to_clin_data = 'C:/Users/Alanna/Desktop/Research_Code/Desktop_research/crc_neoant_clinical_validation/'
path_to_source_data = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/source_data/'
path_base = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_oct_24/' # Path for cohort plots and therapydata
model_type = 'monoclonal' # model type of T cell responses. Acceptable to run 'polyclonal' as well, but it may be slow.
debug = False # Do you want to manually re-simulate immunotherapy for all 10,000 tumors? This is slow if set to True!
make_plots = False # Do you want to plot the results? Default set to False for speed
export_source_data = True # Do you want to export source data into a source data directory?


# Load in AxR data for later use
AxR = []
with open(path_to_ARdata, 'r') as f:
    content = f.readlines()
    for i in content:
        num = float(''.join(list(i)[:-1]))
        if num>=1:
            AxR.append(num)

# Permute AxR data (with seed) for later use
np.random.seed(757)
AxR = np.random.permutation(AxR)
AxR_ind = 0

# Define parameters for trees
paths_to_tumordata = [path_to_MMRD_data, path_to_MMRP_data] # ORDER IS VERY IMPORTANT!
all_deltas = [0.955, 0.955] # Speed of branching process (hard-coded)
ms_stat_list = ['MSI', 'MSS'] # Type of tumors, MMR-D (MSI-H) or MMR-P (MSS)
speed_list = ['fast', 'fast'] # Speed of branching process (hard-coded)
b = 0.25 # Birth rate for branching process (hard-coded)
maxruns = 5000

# Flag certain trajectories for source data export (used in paper figures):
source_data_ids = [1482, 2946, 922, 1671, 156, 2009, 2163, 68, 245, 135, 341, 3831, 4275, 3832]

# Simulate immunotherapy and collect response statistics
try:
    if debug==True:
        raise Exception('Manual re-simulation for debugging purposes.')
    therapydata = pd.read_pickle(open(path_base + model_type + "therapydata_" + str(maxruns) + "_pandas_df.dump", 'rb'))
    granulardata = dill.load(open(path_to_source_data + model_type + "_granular_immunotherapy_data_" + str(maxruns) + ".dump", 'rb'))
    print('Immunotherapy dataset with these parameters already created. Loading...')
except:
    print('Immunotherapy dataset with these parameters not already created or debug mode selected. Creating...')
    # Define parameters for tumor growth, model
    sigma = 10  # cells produced per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
    mu = 1e-2  # per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
    t0 = 0
    b_tumor = 1e-7  # dimesions: 1 / cells
    ici_type = 'Pembro'
    m_proportionality_constant = 1.757e-7 # 4.815e-8  # 2.154e-8# 1.259e-8
    k_proportionality_constant = 3.419e-7 # 2.839e-8  # 1e-10 # 5.627e-7
    a_rng = np.random.default_rng(757)

    # Set up outcome variables
    d = {'ms_stat': [],
         'speed': [],
         'is_clonal_neoant': [],
         'num_subclones': [],
         'starting_TMB': [],
         'ending_TMB': [],
         'dTMB': [],
         'maxNAquality': [],
         'unmut_frac_IC': [],
         'time_to_progression': [],
         'peak_effector_response': [],
         'peak_E_foldchange': [],
         'AUC_effector_response': [],
         'tree_index': [],
         'response_pseud': [],
         'response_tmax': [],
         'best_response': [],
         'tumor_growth_rate': []}
    therapydata = pd.DataFrame(data=d)
    TMB_timeseries = {}
    Effector_timeseries = {}
    Tumor_mats = {}
    AxR_vals_sd = {}
    all_num_solved = []
    ICskip = 0
    # Load in trees and begin work
    for i in range(len(all_deltas)):
        ms_stat = ms_stat_list[i]
        path = paths_to_tumordata[i]
        delta = all_deltas[i]
        speed = speed_list[i]
        tree_dicts = [f for f in os.listdir(path) if f.startswith('dilltree_' + speed + '_' + ms_stat)]
        kk = 0
        EOFflag = 0
        num_solved = 0
        # Load dilled trees from data_gen script
        for dilled_tree_dict in tree_dicts:
            kk += 1
            source_data_flag = False
            if kk > maxruns:
                break
            if kk in source_data_ids and ms_stat=='MSI':
                source_data_flag = True
            # Unpack and processs
            try:
                tree_dict = dill.load(open(path + '/' + dilled_tree_dict, "rb"))
            except EOFError:
                EOFflag += 1
                print('EOF Skip, kk = ' + '%0.f' % kk)
                continue
            # Convert dilled tree into Population object
            tree = unpack_root_dict_to_Population(tree_dict)

            # Sample tumor growth rate from lognormal distribution
            a = a_rng.lognormal(mean = -4.483, sigma = 0.828)  # sample a from distribution of doubling times for CRC. true mean is -6.448, what we were using: -4.483

            # Get clonal structure matrix
            newNewMat, num_subclones, is_there_a_clonal_neoant = get_subclones(path, tree, kk, min_size=10000, save_df=False) #save_df=True) #save df if you changed get_sublcones function
            totalTMB = getTMB(tree, depth=0, TMB=0)
            TMB_1perc = getTMB_threshold(tree, depth=0, TMB=0, threshold=0.01 * 1e5)
            TMB_10perc = getTMB_threshold(tree, depth=0, TMB=0, threshold=0.10 * 1e5)
            unique_muts = get_num_unique_mut(tree, num_mut=0)
            unique_muts_1perc = get_num_unique_mut_threshold(tree, num_mut=0, threshold=0.01 * 1e5)
            unique_muts_10perc = get_num_unique_mut_threshold(tree, num_mut=0, threshold=0.10 * 1e5)
            orig_num_subclones = np.copy(num_subclones)
            if orig_num_subclones > 1:
                max_NA_quality = max(AxR[AxR_ind:(AxR_ind + orig_num_subclones)])
            else:
                max_NA_quality = 0
            min_NA_quality = 0

            if is_there_a_clonal_neoant:
                idx = [AxR_ind + i for i in range(1,len(newNewMat)) if newNewMat[i,0] >= 1e5]
                clonal_neoant_quality = max(AxR[idx])
                num_clonal_neoant = len(idx)
            else:
                clonal_neoant_quality = 0
                num_clonal_neoant = 0

            # Assign m,k based on AxR data
            m, k, AxR_vals, new_AxR_ind = assign_k_and_m(AxR, AxR_ind, newNewMat, num_subclones, b_tumor, mu, a, sigma, m_proportionality_constant, k_proportionality_constant)

            if source_data_flag:
                Tumor_mats[str(kk)] = newNewMat
                AxR_vals_sd[str(kk)] = AxR_vals

            # Find initial value populations
            init_vals, m_trunc = assign_ICs(newNewMat)

            # If necessary (parental/founder population has 0 pop), truncate m and k to remove that from consideration
            m = m[m_trunc:]
            k = k[m_trunc:]
            AxR_vals = AxR_vals[m_trunc:]

            # Group together uniquely evolving populations based on (m,k) uniqueness
            starting_TMB = sum(init_vals)
            unique_m = list(set(m))  # use set only for # of unique values, and not for anything else!!!
            unique_k = list(set(k))
            if len(unique_m) != len(unique_k):
                print('The number of (m,k) pairs is strange -- check this.')
            if len(unique_m) != len(m): # if we have overlapping populations, reduce subclonal structure for final input.
                adj_IC = []
                adj_m = []
                adj_k = []
                adj_AxR = []
                ii = 0
                skip = 0
                while ii < (len(unique_m)+skip):  # for each m, preserving order, and not exceeding total number of unique m values!!!
                    if m[ii] in adj_m: #if we already covered this one,
                        skip += 1
                        ii += 1
                        continue
                    inds = np.array([jj for jj in range(len(m)) if m[jj] == m[ii]])  # find indices of all pops with same m
                    totpop = sum(np.array(init_vals)[inds])  # add together those populations
                    adj_IC.append(totpop)  # store this into the adjusted IC list (effective subclone number)
                    adj_m.append(m[ii])  # store this m into the adjusted m list (no repeats)
                    adj_k.append(k[ii]) # preserves the ordering of original m and k
                    adj_AxR.append(AxR_vals[ii])
                    ii += 1
                init_vals = adj_IC
                m = adj_m
                k = adj_k
                AxR_vals = adj_AxR
                if m[0] != adj_m[0]:
                    print('Something went wrong with reducing subclonal populations by (m,k) pairs; m got scrambled.')

                # Trim off any size-zero subclones before running
                adj_IC = []
                adj_m = []
                adj_k = []
                adj_AxR = []
                for ii in range(len(m)):
                    if init_vals[ii] == 0:
                        continue # skip this one!
                    else:
                        adj_IC.append(init_vals[ii])
                        adj_m.append(m[ii])
                        adj_k.append(k[ii])
                        adj_AxR.append(AxR_vals[ii])
                if sum(adj_IC)!= 1e5:
                    print('Error! Sum of initial conditions is not correct')
                init_vals = adj_IC
                m = adj_m
                k = adj_k
                AxR_vals = adj_AxR
                num_subclones = len(m)  # adjust number of effective subclones
            check_zero_remnants_vec = [iiii > 0 for iiii in init_vals]
            if sum(check_zero_remnants_vec)<len(check_zero_remnants_vec):
                m = [m[nn] for nn in range(len(init_vals)) if init_vals[nn]>0]
                k = [k[nn] for nn in range(len(init_vals)) if init_vals[nn]>0]
                AxR_vals = [AxR_vals[nn] for nn in range(len(init_vals)) if init_vals[nn]>0]
                init_vals = [init_vals[nn] for nn in range(len(init_vals)) if init_vals[nn]>0]
                if AxR_vals[0]==0:
                    num_subclones = sum(check_zero_remnants_vec) - 1
                else:
                    num_subclones = sum(check_zero_remnants_vec)
            if len(AxR_vals)>0:
                min_NA_quality = min(AxR_vals) # weakest subclone in the tumor, including the founder with a 0 score

            if len(init_vals) != len(m):
                print('ICs have different length than the number of (m,k) pairs.')
            AxR_ind = np.copy(new_AxR_ind)

            # Set up ODEfun based on number of subclones that are of the right size and model type
            if model_type == 'polyclonal':
                odefun, y0 = define_odefun_polyclonal(m, k, sigma, mu, a, b_tumor, init_vals)
            elif model_type == 'monoclonal':
                odefun, y0 = define_odefun_monoclonal(m, k, sigma, mu, a, b_tumor, init_vals)


            # Solve ODE using solve_IVP
            Tmax = round(55 * 30.437)  # x 100 to get to days. 84 days total = 12 weeks. Large Tmax = 5, small = 0.84 (12 weeks)
            dt = 1  # "sample" every 1 day.
            t_span = [t0, Tmax]
            t_eval = np.arange(t0, Tmax+dt, dt)
            sol = solve_ivp(odefun, t_span, y0, t_eval=t_eval)
            t = sol.t
            t_pseud_index = round((12*7) / dt + 1) # time to pseudoprogression is 5.7 weeks -- not 12 weeks. Add one for python weirdness
            if model_type == 'polyclonal':
                subclone_sol = sol.y[1:, ]
                effector_sol = sol.y[0, ]
            elif model_type == 'monoclonal':
                subclone_sol = sol.y[len(m):, ]
                effector_sol = sol.y[0:len(m), ]

            TMB_over_time = [sum(subclone_sol[:, iii]) for iii in range(len(t))]
            effector_over_time = [sum(effector_sol[:,iii]) for iii in range(len(t))]
            ending_num_subclones = sum(
                [1 for i in subclone_sol[:, -1] if i >= 1e-2])  # total the number of surviving clones at Tmax
            if AxR_vals[0] == 0 and subclone_sol[0, -1] >= 1e-2:
                ending_num_subclones -= 1  # remove the founder from that total number of subclones

            # Detect oscillations (alpha):
            does_it_oscillate = detectOscillations(TMB_over_time, t)

            # Check response every n weeks
            response_score_dict = {'CR': 0, 'PR': 1, 'SD': 2, 'PD': 3}
            response_dict = {}
            iw = 1
            best_response = 'PD'
            best_resp_score = 3
            time_to_PD = -1
            pdflag = False
            while round(12*iw*7 + 1) <= Tmax:
                if iw == 1:
                    t_index = round(12 * 7 + 1)  # Check at 12 weeks for the very first time point
                elif (iw*9*7+1) > 365: # past first year, check every 12 weeks.
                    t_index = round(12 * iw * 7 + 1)
                else:
                    t_index = round(9 * iw * 7 + 1)
                #resp_t_inds.append(t_index)  # append the indices for checking
                #resp_TMB_val_subset.append(TMB_over_time[t_index])  # values checked
                cur_resp = getResponse(TMB_over_time[:t_index], t[:t_index]) #getResponse(resp_TMB_val_subset, resp_t_inds)
                best_resp_score = min(response_score_dict[best_response], response_score_dict[cur_resp])
                if best_resp_score==0:
                    best_response = 'CR'
                elif best_resp_score==1:
                    best_response = 'PR'
                elif best_resp_score==2:
                    best_response = 'SD'
                elif best_resp_score==3:
                    best_response = 'PD'
                else:
                    print('Something went wrong with finding best response over time.')
                if iw==1:
                    response_12w = cur_resp
                if cur_resp == 'PD' and pdflag == False:  # once it's PD, no use in checking further ***
                    time_to_PD = t[t_index]
                    pdflag = True
                    response_tmax = cur_resp  # terminate therapy and use this as response_tmax
                    break
                iw += 1
            if pdflag==False:
                # Finally, check exactly at Tmax, if PD not achieved yet
                t_index = Tmax
                response_tmax = getResponse(TMB_over_time[:t_index], t[:t_index]) #getResponse(resp_TMB_val_subset, resp_t_inds)
                cur_resp = response_tmax
                if pdflag == False: # if not progressed yet, check if this could be our best score yet.
                    best_resp_score = min(response_score_dict[best_response], response_score_dict[cur_resp])
                    if best_resp_score == 0:
                        best_response = 'CR'
                    elif best_resp_score == 1:
                        best_response = 'PR'
                    elif best_resp_score == 2:
                        best_response = 'SD'
                    elif best_resp_score == 3:
                        best_response = 'PD'
                    time_to_PD = t[-1] # if not progressed, set PFS to tmax

            if time_to_PD > (24*7 + 1): # compute DCR as in Le et al. 2023
                diseaseControl = 1
            else:
                diseaseControl = 0

            # Check early disease behavior: 0, 12, 20 weeks.
            t0 = 0
            t1 = int(12*7 - 1)
            t2 = int(20*7 - 1)
            check_times = [t0, t1, t2]
            check_tmbs = [TMB_over_time[idx] for idx in check_times]
            wk12resp = getResponse(check_tmbs[:2], check_times[:2])
            wk20resp = getResponse(check_tmbs, check_times)
            if wk12resp=='SD':
                early_response = 'SD'
            elif wk12resp=='PD' or wk20resp=='PD':
                early_response = 'PD'
            else:
                early_response = wk20resp

            # Check immune-specific response criteria (imRECIST).
            [time_to_RECIST_PD, RECIST_pdflag, time_to_CR, time_to_imRECIST_iCPD, imRECIST_response_tmax] = check_imRECIST(TMB_over_time, t, min_increment = 4*7)

            # Check immune-specific response criteria (iRECIST)
            [time_to_RECIST_PD, RECIST_pdflag, time_to_iRECIST_iCPD, iRECIST_response_tmax] = check_iRECIST(TMB_over_time, t, min_increment = 4*7)

            # Check immune-specific response criteria (irRECIST)
            [time_to_RECIST_PD, RECIST_pdflag, time_to_irRECIST_iCPD, irRECIST_response_tmax] = check_irRECIST(TMB_over_time, t, min_increment = 4*7)

            # Simulate the structure in Colle et al. 2021 (Eur J Cancer)
            cellct_result = simulate_Colle_analysis(TMB_over_time, t)
            diam_result = simulate_Colle_analysis([estimate_lesion_diameter(i) for i in TMB_over_time], t)

            # Get long-term response
            LTR = getLongTermResponse(best_response, response_tmax)
            response_label_dict = {'CR': 'Complete Response', 'PR': 'Partial Response', 'SD': 'Stable Disease',
                                   'PD': 'Progressive Disease'}

            # Check for pseudoprogression (beta):
            pseud, minSLD_percent = getPseudoprogression_deprecated(TMB_over_time, t)

            # ### Plot timeseries of tumor and effector cell responses over time:
            if make_plots:
                if kk==1:
                    os.makedirs(path + 'tumor_plots/', exist_ok=True)
                plt.rcParams.update({'font.size': 14})
                plt.rcParams['font.family'] = ['Arial']
                plt.rc('legend', fontsize=11)

                plt.figure(figsize=(5, 4))
                plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
                if num_subclones>0:
                    if AxR_vals[0]==0:
                        plt.rc('axes', prop_cycle=(
                            cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo', 'aquamarine', 'steelblue', 'navy'])))
                        plt.plot(t/30.4368, subclone_sol.T / TMB_over_time[0])
                        if len(subclone_sol) > 1:
                            plt.plot(t/30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)
                        plt.legend(
                            ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol) - 1)] + [
                                'Total TB'])
                    else:
                        plt.rc('axes', prop_cycle=(cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo', 'aquamarine', 'steelblue', 'navy'])))
                        plt.plot(t/30.4368, subclone_sol.T / TMB_over_time[0])
                        if len(subclone_sol) > 1:
                            plt.plot(t/30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)
                        plt.legend(['Neoantigen ' + str(iii+1) for iii in range(len(subclone_sol))] + ['Total TB'])
                else:
                    plt.plot(t/30.4368, subclone_sol.T / TMB_over_time[0], 'fuchsia') # Fuchsia: non-neoantigen containing subclone (Founder)
                    plt.legend(['Total TB (no neoantigenic clones)'])
                plt.xlabel('Months since start of therapy')
                plt.ylabel('Relative tumor volume')
                plt.xticks([0, 10, 20, 30, 40, 50])
                plt.ylim([0, 3]) #max(2, max(TMB_over_time)/TMB_over_time[0])])
                plt.title(LTR)
                plt.savefig(path + '/tumor_plots/' + model_type + '_largeTmax_subclone_fig_' + str(kk) + '.png', format='png')
                plt.savefig(path + '/tumor_plots/' + model_type + '_largeTmax_subclone_fig_' + str(kk) + '.svg', format='svg')

                first_year_ind = int(365)
                fig, ax1 = plt.subplots(figsize=(5, 4))
                ax1.set_xlabel('Days since start of therapy')
                ax1.set_ylabel('Relative tumor volume')
                ax1.set_ylim([0, 2])
                ax1.plot(t[:first_year_ind], TMB_over_time[:first_year_ind] / TMB_over_time[0], 'k--', linewidth=2)

                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.plot(t[:first_year_ind], effector_over_time[:first_year_ind] / effector_over_time[0], color=color, linewidth=2)
                ax2.set_ylabel('Relative T cell population', color=color)
                ax2.set_ylim([0, 200])
                ax2.tick_params(axis='y', labelcolor=color)
                plt.title('')
                fig.tight_layout()
                plt.savefig(path + '/tumor_plots/' + model_type + '_smallTmax_effector_fig_' + str(kk) + '.png', format='png')
                plt.savefig(path + '/tumor_plots/' + model_type + '_smallTmax_effector_fig_' + str(kk) + '.svg', format='svg')


                fig, ax1 = plt.subplots(figsize=(5, 4))
                ax1.set_xlabel('Months since start of therapy')
                ax1.set_ylabel('Relative tumor volume')
                ax1.set_ylim([0, 2])
                ax1.plot(t/30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)

                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.plot(t/30.4368, effector_over_time / effector_over_time[0], color=color, linewidth=2)
                ax2.set_ylabel('Relative T cell population', color=color)
                ax2.set_ylim([0, 100])
                ax2.tick_params(axis='y', labelcolor=color)
                plt.title('')
                fig.tight_layout()
                plt.savefig(path + '/tumor_plots/' + model_type + '_largeTmax_effector_fig_' + str(kk) + '.png', format='png')
                plt.savefig(path + '/tumor_plots/' + model_type + '_largeTmax_effector_fig_' + str(kk) + '.svg', format='svg')

                plt.figure(figsize=(5, 4))
                plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
                if num_subclones>0:
                    if AxR_vals[0]==0:
                        plt.rc('axes', prop_cycle=(
                            cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo', 'aquamarine', 'steelblue', 'navy'])))
                        plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0])
                        if len(subclone_sol) > 1:
                            plt.plot(t[:t_pseud_index], TMB_over_time[:t_pseud_index] / TMB_over_time[0], 'k--', linewidth=2)
                            plt.legend(
                                ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol) - 1)] + [
                                    'Total TB'])
                    else:
                        plt.rc('axes', prop_cycle=(cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo', 'aquamarine', 'steelblue', 'navy'])))
                        plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0])
                        if len(subclone_sol) > 1:
                            plt.plot(t[:t_pseud_index], TMB_over_time[:t_pseud_index] / TMB_over_time[0], 'k--', linewidth=2)
                        plt.legend(['Neoantigen ' + str(iii+1) for iii in range(len(subclone_sol))] + ['Total TB'])
                else:
                    plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0], 'fuchsia') # Fuchsia: non-neoantigen containing subclone (Founder)
                    plt.legend(['Total TB (no neoantigenic clones)'])
                plt.xlabel('Days since start of therapy')
                plt.ylabel('Relative tumor volume')
                plt.ylim([0, 2])
                plt.title(response_label_dict[best_response])
                plt.savefig(path + '/tumor_plots/' + model_type + '_smallTmax_subclone_fig_' + str(kk) + '.png', format='png')
                plt.savefig(path + '/tumor_plots/' + model_type + '_smallTmax_subclone_fig_' + str(kk) + '.svg', format='svg')

                # Restore plot defaults
                plt.rcdefaults()
                plt.close('all')
            # Compute and append response variables
            ending_TMB = TMB_over_time[-1]
            # Unmutated fraction at start of sim:
            unmut_frac_IC = tree.size / tree.get_total_size()
            # Weighted antigenicity
            axr_weighted_anteginicity = sum([AxR_vals[i] * init_vals[i] for i in range(len(init_vals))]) / 1e5

            # Detect pseudoprogression overall:
            [early_events, late_events, num_pseudoprogression_events,
             initial_Tcell_slope, time_to_first_Tcell_peak, avg_delay, time_to_first_psp]  = getPseudoprogression(LTR, TMB_over_time, effector_over_time, t)

            # Subsample w/ 6 weeks
            subsampling_slice = 7 * 6  # if we reevaluate every 8 weeks for the duration of treatment, how many PsP incidents are there?
            [w6_subsampled_early_events, w6_subsampled_late_events, w6_subsampled_num_pseudoprogression_events,
             w6_subsampled_initial_Tcell_slope, w6_subsampled_time_to_first_Tcell_peak, w6_subsampled_avg_delay,
             w6_subsampled_time_to_first_psp] = getPseudoprogression(LTR, TMB_over_time[::subsampling_slice],
                                                                     effector_over_time[::subsampling_slice],
                                                                     t[::subsampling_slice])

            # Subsample w/ 8 weeks
            subsampling_slice = 7*8 # if we reevaluate every 8 weeks for the duration of treatment, how many PsP incidents are there?
            [w8_subsampled_early_events, w8_subsampled_late_events, w8_subsampled_num_pseudoprogression_events,
             w8_subsampled_initial_Tcell_slope, w8_subsampled_time_to_first_Tcell_peak, w8_subsampled_avg_delay, w8_subsampled_time_to_first_psp]  = getPseudoprogression(LTR, TMB_over_time[::subsampling_slice], effector_over_time[::subsampling_slice], t[::subsampling_slice])

            # Subsample w/ 10 weeks
            subsampling_slice = 7 * 10  # if we reevaluate every 8 weeks for the duration of treatment, how many PsP incidents are there?
            [w10_subsampled_early_events, w10_subsampled_late_events, w10_subsampled_num_pseudoprogression_events,
             w10_subsampled_initial_Tcell_slope, w10_subsampled_time_to_first_Tcell_peak, w10_subsampled_avg_delay,
             w10_subsampled_time_to_first_psp] = getPseudoprogression(LTR, TMB_over_time[::subsampling_slice],
                                                                     effector_over_time[::subsampling_slice],
                                                                     t[::subsampling_slice])

            # Subsample w/ 12 weeks
            subsampling_slice = 7 * 12  # if we reevaluate every 8 weeks for the duration of treatment, how many PsP incidents are there?
            [w12_subsampled_early_events, w12_subsampled_late_events, w12_subsampled_num_pseudoprogression_events,
             w12_subsampled_initial_Tcell_slope, w12_subsampled_time_to_first_Tcell_peak, w12_subsampled_avg_delay,
             w12_subsampled_time_to_first_psp] = getPseudoprogression(LTR, TMB_over_time[::subsampling_slice],
                                                                     effector_over_time[::subsampling_slice],
                                                                     t[::subsampling_slice])

            # Peak effector cell response
            if model_type == 'monoclonal':
                effector_sol = [sum(effector_sol[:, efs]) for efs in range(len(t_eval))]
            peak_E_response = max(effector_sol)
            peak_E_foldchange = peak_E_response / effector_sol[0]
            # AUC effector cell response
            AUC_E_response = np.trapz(effector_sol, x=t)
            dTMB = ending_TMB / starting_TMB
            tree_index = getTreeIndex(tree)
            if is_there_a_clonal_neoant == True:
                tumor_type = 'clonal'
            elif AxR_vals[0] != 0:
                tumor_type = 'subclonal_dominant'
            else:
                tumor_type = 'persistent_founder'
            change_in_num_subclones = num_subclones - ending_num_subclones
            saved_by_imRECIST = 0
            saved_by_iRECIST = 0
            saved_by_irRECIST = 0
            if time_to_RECIST_PD + 4 * 7 < Tmax:  # if the unconfirmed PD was not confirmed for at least 1 day,
                if imRECIST_response_tmax != 'iCPD':
                    saved_by_imRECIST = 1
                if iRECIST_response_tmax != 'iCPD':
                    saved_by_iRECIST = 1
                if irRECIST_response_tmax != 'iCPD':
                    saved_by_irRECIST = 1
                    print('Tumor saved by irRECIST, id = ' + str(kk))
            PFS_preserved_by_imRECIST = 0
            PFS_preserved_by_iRECIST = 0
            PFS_preserved_by_irRECIST = 0
            if saved_by_imRECIST == 1:
                PFS_preserved_by_imRECIST = time_to_imRECIST_iCPD - time_to_RECIST_PD
            if saved_by_iRECIST == 1:
                PFS_preserved_by_iRECIST = time_to_iRECIST_iCPD - time_to_RECIST_PD
            if saved_by_irRECIST == 1:
                PFS_preserved_by_irRECIST = time_to_irRECIST_iCPD - time_to_RECIST_PD
            tumor_dict = {'ms_stat': [ms_stat],
                          'speed': [speed],
                          'is_clonal_neoant': [is_there_a_clonal_neoant],
                          'clonal_neoant_quality': [clonal_neoant_quality],
                          'num_clonal_neoant': [num_clonal_neoant],
                          'num_subclones': [num_subclones],
                          'ending_num_subclones': [ending_num_subclones],
                          'change_in_num_subclones': [change_in_num_subclones],
                          'starting_TMB': [starting_TMB],
                          'ending_TMB': [ending_TMB],
                          'dTMB': [dTMB],
                          'pseudoprogression': [pseud],
                          'minSLD_percent': [minSLD_percent],
                          'maxNAquality': [max_NA_quality],
                          'minNAquality': [min_NA_quality],
                          'unmut_frac_IC': [unmut_frac_IC],
                          'time_to_progression': [time_to_PD],
                          'time_to_CR': [time_to_CR],
                          'time_to_RECIST_PD': [time_to_RECIST_PD],  # smoothly varying per day
                          'time_to_imRECIST_iCPD': [time_to_imRECIST_iCPD],
                          'time_to_iRECIST_iCPD': [time_to_iRECIST_iCPD],
                          'time_to_irRECIST_iCPD': [time_to_irRECIST_iCPD],
                          'saved_by_imRECIST': [saved_by_imRECIST],
                          'saved_by_iRECIST': [saved_by_iRECIST],
                          'saved_by_irRECIST': [saved_by_irRECIST],
                          'PFS_saved_by_imRECIST': [PFS_preserved_by_imRECIST],
                          'PFS_saved_by_iRECIST': [PFS_preserved_by_iRECIST],
                          'PFS_saved_by_irRECIST': [PFS_preserved_by_irRECIST],
                          'peak_effector_response': [peak_E_response],
                          'peak_E_foldchange': [peak_E_foldchange],
                          'AUC_effector_response': [AUC_E_response],
                          'tree_index': [tree_index],
                          'response_12w': [response_12w],
                          'response_tmax': [response_tmax],
                          'imRECIST_response_tmax': [imRECIST_response_tmax],
                          'iRECIST_response_tmax': [iRECIST_response_tmax],
                          'irRECIST_response_tmax': [irRECIST_response_tmax],
                          'best_response': [best_response],
                          'tumor_growth_rate': [a],
                          'axr_weighted_anteginicity': [axr_weighted_anteginicity],
                          'oscillation': [does_it_oscillate],
                          'LTR': [LTR],
                          'totalTMB': [totalTMB],
                          'TMB_1perc': [TMB_1perc],
                          'TMB_10perc': [TMB_10perc],
                          'unique_TMB': [unique_muts],
                          'unique_TMB_1perc': [unique_muts_1perc],
                          'unique_TMB_10perc': [unique_muts_10perc],
                          'diseaseControl': [diseaseControl],
                          'tumor_type': [tumor_type],
                          'early_response': [early_response],
                          'early_events': [early_events],
                          'late_events': [late_events],
                          'num_pseudoprogression_events': [num_pseudoprogression_events],
                          'initial_Tcell_slope': [initial_Tcell_slope],
                          'time_to_first_Tcell_peak': [time_to_first_Tcell_peak],
                          'avg_delay': [avg_delay],
                          'time_to_first_psp': [time_to_first_psp],
                          'w6_subsampled_early_events': [w6_subsampled_early_events], # 6 week subsampling
                          'w6_subsampled_late_events': [w6_subsampled_late_events],
                          'w6_subsampled_num_pseudoprogression_events': [w6_subsampled_num_pseudoprogression_events],
                          'w6_subsampled_initial_Tcell_slope': [w6_subsampled_initial_Tcell_slope],
                          'w6_subsampled_time_to_first_Tcell_peak': [w6_subsampled_time_to_first_Tcell_peak],
                          'w6_subsampled_avg_delay': [w6_subsampled_avg_delay],
                          'w6_subsampled_time_to_first_psp': [w6_subsampled_time_to_first_psp],
                          'w8_subsampled_early_events': [w8_subsampled_early_events], # 8 week subsampling
                          'w8_subsampled_late_events': [w8_subsampled_late_events],
                          'w8_subsampled_num_pseudoprogression_events': [w8_subsampled_num_pseudoprogression_events],
                          'w8_subsampled_initial_Tcell_slope': [w8_subsampled_initial_Tcell_slope],
                          'w8_subsampled_time_to_first_Tcell_peak': [w8_subsampled_time_to_first_Tcell_peak],
                          'w8_subsampled_avg_delay': [w8_subsampled_avg_delay],
                          'w8_subsampled_time_to_first_psp': [w8_subsampled_time_to_first_psp],
                          'w10_subsampled_early_events': [w10_subsampled_early_events], # 10 week subsampling
                          'w10_subsampled_late_events': [w10_subsampled_late_events],
                          'w10_subsampled_num_pseudoprogression_events': [w10_subsampled_num_pseudoprogression_events],
                          'w10_subsampled_initial_Tcell_slope': [w10_subsampled_initial_Tcell_slope],
                          'w10_subsampled_time_to_first_Tcell_peak': [w10_subsampled_time_to_first_Tcell_peak],
                          'w10_subsampled_avg_delay': [w10_subsampled_avg_delay],
                          'w10_subsampled_time_to_first_psp': [w10_subsampled_time_to_first_psp],
                          'w12_subsampled_early_events': [w12_subsampled_early_events],  # 12 week subsampling
                          'w12_subsampled_late_events': [w12_subsampled_late_events],
                          'w12_subsampled_num_pseudoprogression_events': [w12_subsampled_num_pseudoprogression_events],
                          'w12_subsampled_initial_Tcell_slope': [w12_subsampled_initial_Tcell_slope],
                          'w12_subsampled_time_to_first_Tcell_peak': [w12_subsampled_time_to_first_Tcell_peak],
                          'w12_subsampled_avg_delay': [w12_subsampled_avg_delay],
                          'w12_subsampled_time_to_first_psp': [w12_subsampled_time_to_first_psp],
                          'Colle_cellct': [cellct_result],
                          'Colle_diam': [diam_result]
            }
            cur_data = pd.DataFrame(tumor_dict)
            therapydata = pd.concat([therapydata, cur_data])
            if source_data_flag:
                TMB_timeseries[str(kk)] = subclone_sol
                Effector_timeseries[str(kk)] = effector_over_time
            num_solved += 1
            print('Simulated therapy on ' + ms_stat + ' tumor # ' + str(kk) + ' out of ' + str(maxruns) + ' tumors; best response: ' + best_response)
            all_num_solved.append(num_solved)
        therapydata.index=np.arange(1, len(therapydata)+1)
        granulardata = [Tumor_mats, AxR_vals_sd, TMB_timeseries, Effector_timeseries]
        dill.dump(granulardata, open(path_to_source_data + model_type + "_granular_immunotherapy_data_" + str(maxruns) + ".dump", 'wb'))
        dill.dump(therapydata, open(path_base + model_type + "therapydata_" + str(maxruns) + "_pandas_df.dump", 'wb'))
        print('Immunotherapy dataset with these parameters created. Saving to file...')
        print('Number of runs skipped due to nontrivial IC setup: ' + str(ICskip))


[Tumor_mats, AxR_vals_sd, TMB_timeseries, Effector_timeseries] = granulardata
MSIdata = therapydata[therapydata['ms_stat']=='MSI'] # Due to old naming convention, MSI data = MMR-D data (used to be labeled MSI-H)
MSSdata = therapydata[therapydata['ms_stat']=='MSS'] # and MSS data = MMR-P data

# Source data export:
if export_source_data:
    os.makedirs(path_to_source_data, exist_ok=True)  # create directory for plots of the entire simulated cohort

    # Source data for Figs 2, 4(a&b), and SF 2
    # Clonality/tree structure:
    for key in Tumor_mats.keys():
        arr = Tumor_mats[key]
        axrs_list = np.array(AxR_vals_sd[key]) # keys are same for Tumor_mats and AxR_vals_sd
        axrs = np.reshape(axrs_list, (len(axrs_list),1))
        num_unique_neoants = arr.shape[1] - 2
        colnames = ['total_pop', 'unique_pop'] + ["neoantigen_" + str(i) for i in range(1,num_unique_neoants+1)] + ["AxR_vals"]
        df = pd.DataFrame(np.hstack((arr, axrs)), columns=colnames)
        df.to_csv(path_to_source_data + 'clonal_structure_' + key + '_dataframe.csv', index=False)

    # Tumor/subclonal populations over time:
    for key in TMB_timeseries.keys():
        arr = TMB_timeseries[key]
        t = np.arange(1, arr.shape[1]+1, 1)
        tot = [sum(arr[:, iii]) for iii in range(len(t))]
        num_subclones = arr.shape[0]
        colnames = ["subclone_" + str(i) for i in range(1, num_subclones+1)]
        df = pd.DataFrame(np.transpose(arr), columns=colnames)
        df['total_tumor'] = tot
        df['days'] = t
        df.to_csv(path_to_source_data + 'tumor_longitudinal_' + key + '_dataframe.csv', index=False)

    # Effector populations over time (summed over monoclonal populations):
    for key in Effector_timeseries.keys():
        arr = TMB_timeseries[key]
        t = np.arange(1, arr.shape[1]+1, 1)
        tot = [sum(arr[:, iii]) for iii in range(len(t))]
        data = {"effector_pop": Effector_timeseries[key],
                "total_tumor_pop": tot,
                "days": np.arange(1, len(Effector_timeseries[key])+1, 1)}
        df = pd.DataFrame(data)
        df.to_csv(path_to_source_data + 'effector_longitudinal_' + key + '_dataframe.csv', index=False)

    # Source data for Fig 3
    df = MSIdata[['best_response',
                  'time_to_progression',
                  'response_12w']]
    df.to_csv(path_to_source_data + 'Fig3_MMRD.csv', index=False)
    df = MSSdata['response_12w']
    df.to_csv(path_to_source_data + 'Fig3_MMRP.csv', index=False)

    # Source data for Fig 4 (c, right; d, left)
    df = MSIdata[['LTR',
                  'best_response',
                  'response_12w',
                  'num_subclones']]
    df.to_csv(path_to_source_data + 'Fig4_MMRD.csv', index=False)
    df = MSSdata['LTR']
    df.to_csv(path_to_source_data + 'Fig4_MMRP.csv', index=False)

    # Source data for Fig 5
    df = MSIdata[['LTR',
                  'maxNAquality',
                  'axr_weighted_anteginicity',
                  'unique_TMB_10perc',
                  'tree_index',
                  'is_clonal_neoant',
                  'minNAquality']]
    df.to_csv(path_to_source_data + 'Fig5_MMRD.csv', index=False)

    # Source data for SF 3
    MSI_DRs = MSIdata[MSIdata['LTR'] == 'Durable Response']
    df = MSI_DRs[['maxNAquality',
                  'axr_weighted_anteginicity',
                  'num_subclones',
                  'tree_index',
                  'is_clonal_neoant']]
    df.to_csv(path_to_source_data + 'SF3_MMRD_DRs.csv', index=False)

    # Source data for SF 4 is output AT END OF FILE
    # Source data for SF 5 is AT END OF FILE
#
#
#
#
#
#
# Printed outputs
#
#
#
print('DCR = ' + str(sum(MSIdata['diseaseControl'])/len(MSIdata['diseaseControl'])))
ORR_count = (sum(MSIdata['best_response']=='PR')+sum(MSIdata['best_response']=='CR'))
n = len(MSIdata['best_response'])
ci = binomtest(ORR_count, n, ORR_count/n).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
yerr_min = ci.low * 100
yerr_max = ci.high * 100
print('ORR = ' + str(round(ORR_count/n*100, 3)) + '%')
print('ORR CI: (' + str(round(yerr_min, 3)) + ', ' + str(round(yerr_max, 3)) + '%)')

objective_response = []
for i in MSIdata['best_response']:
    if i=='PR' or i=='CR':
        objective_response.append('OR')
    else:
        objective_response.append('NOR')
MSIdata['objective_response'] = objective_response

ttpData = [i for i in MSIdata['time_to_progression'] if i>0]
print('Median PFS = ' + str(np.median(ttpData)/30.4368))
print('36-month PFS = ' + str(sum([i > 36*30.4368 for i in ttpData])/len(ttpData) * 100))
# np.savetxt(path_base + 'ttp.txt', ttpData, delimiter=' ')
dill.dump(ttpData, open(path_base + "ttp.dump", 'wb'))

print('MSI PD best responses: ' + str(sum(MSIdata['best_response']=='PD')/len(MSIdata['best_response']) * 100) + '%')
print('MSI SD best responses: ' + str(sum(MSIdata['best_response']=='SD')/len(MSIdata['best_response']) * 100) + '%')
print('MSI PR best responses: ' + str(sum(MSIdata['best_response']=='PR')/len(MSIdata['best_response']) * 100) + '%')
print('MSI CR best responses: ' + str(sum(MSIdata['best_response']=='CR')/len(MSIdata['best_response']) * 100) + '%')

print('MSI PD 12w responses: ' + str(sum(MSIdata['response_12w']=='PD')/len(MSIdata['response_12w']) * 100) + '%')
print('MSI SD 12w responses: ' + str(sum(MSIdata['response_12w']=='SD')/len(MSIdata['response_12w']) * 100) + '%')
print('MSI PR 12w responses: ' + str(sum(MSIdata['response_12w']=='PR')/len(MSIdata['response_12w']) * 100) + '%')
print('MSI CR 12w responses: ' + str(sum(MSIdata['response_12w']=='CR')/len(MSIdata['response_12w']) * 100) + '%')

print('MSS PD 12w responses: ' + str(sum(MSSdata['response_12w']=='PD')/len(MSSdata['response_12w']) * 100) + '%')
print('MSS SD 12w responses: ' + str(sum(MSSdata['response_12w']=='SD')/len(MSSdata['response_12w']) * 100) + '%')
print('MSS PR 12w responses: ' + str(sum(MSSdata['response_12w']=='PR')/len(MSSdata['response_12w']) * 100) + '%')
print('MSS CR 12w responses: ' + str(sum(MSSdata['response_12w']=='CR')/len(MSSdata['response_12w']) * 100) + '%')

MSI_DRs = MSIdata[MSIdata['LTR']=='Durable Response']
MSI_ARs = MSIdata[MSIdata['LTR']=='Acquired Resistance']
MSI_NRs = MSIdata[MSIdata['LTR']=='No Response']

print('MSI DR: ' + str(sum(MSIdata['LTR']=='Durable Response')/len(MSIdata['best_response']) * 100) + '%')
print('MSI AR: ' + str(sum(MSIdata['LTR']=='Acquired Resistance')/len(MSIdata['best_response']) * 100) + '%')
print('MSI NR: ' + str(sum(MSIdata['LTR']=='No Response')/len(MSIdata['best_response']) * 100) + '%')

print('MSS DR: ' + str(sum(MSSdata['LTR']=='Durable Response')/len(MSSdata['best_response']) * 100) + '%')
print('MSS AR: ' + str(sum(MSSdata['LTR']=='Acquired Resistance')/len(MSSdata['best_response']) * 100) + '%')
print('MSS NR: ' + str(sum(MSSdata['LTR']=='No Response')/len(MSSdata['best_response']) * 100) + '%')

print('DR tree index: ' + str(np.mean(MSI_DRs['tree_index'])) + ' +/- ' + str(np.std(MSI_DRs['tree_index'])))
print('AR tree index: ' + str(np.mean(MSI_ARs['tree_index'])) + ' +/- ' + str(np.std(MSI_ARs['tree_index'])))
print('NR tree index: ' + str(np.mean(MSI_NRs['tree_index'])) + ' +/- ' + str(np.std(MSI_NRs['tree_index'])))

# print('DR mean num mutations: ' + str(np.mean(MSI_DRs['totalTMB'])) + ' +/- ' + str(np.std(MSI_DRs['totalTMB'])))
# print('AR mean num mutations: ' + str(np.mean(MSI_ARs['totalTMB'])) + ' +/- ' + str(np.std(MSI_ARs['totalTMB'])))
# print('NR mean num mutations: ' + str(np.mean(MSI_NRs['totalTMB'])) + ' +/- ' + str(np.std(MSI_NRs['totalTMB'])))
# print('DR 95% CI: ' + str([i for i in stats.norm.interval(0.95, loc=np.mean(MSI_DRs['totalTMB']), scale=np.std(MSI_DRs['totalTMB'])/np.sqrt(len(MSI_DRs['totalTMB'])))]))
# print('AR 95% CI: ' + str([i for i in stats.norm.interval(0.95, loc=np.mean(MSI_ARs['totalTMB']), scale=np.std(MSI_ARs['totalTMB'])/np.sqrt(len(MSI_ARs['totalTMB'])))]))
# print('NR 95% CI: ' + str([i for i in stats.norm.interval(0.95, loc=np.mean(MSI_NRs['totalTMB']), scale=np.std(MSI_NRs['totalTMB'])/np.sqrt(len(MSI_NRs['totalTMB'])))]))

print('DR mean num unique neoantigenic mutations: ' + str(np.mean(MSI_DRs['unique_TMB'])) + ' +/- ' + str(np.std(MSI_DRs['unique_TMB'])))
print('AR mean num unique neoantigenic mutations: ' + str(np.mean(MSI_ARs['unique_TMB'])) + ' +/- ' + str(np.std(MSI_ARs['unique_TMB'])))
print('NR mean num unique neoantigenic mutations: ' + str(np.mean(MSI_NRs['unique_TMB'])) + ' +/- ' + str(np.std(MSI_NRs['unique_TMB'])))

print('DR mean num unique neoantigenic mutations in over 1% of tumor: ' + str(np.mean(MSI_DRs['unique_TMB_1perc'])) + ' +/- ' + str(np.std(MSI_DRs['unique_TMB_1perc'])))
print('AR mean num unique neoantigenic mutations in over 1% of tumor: ' + str(np.mean(MSI_ARs['unique_TMB_1perc'])) + ' +/- ' + str(np.std(MSI_ARs['unique_TMB_1perc'])))
print('NR mean num unique neoantigenic mutations in over 1% of tumor: ' + str(np.mean(MSI_NRs['unique_TMB_1perc'])) + ' +/- ' + str(np.std(MSI_NRs['unique_TMB_1perc'])))

print('DR mean num unique neoantigenic mutations in over 10% of tumor: ' + str(np.mean(MSI_DRs['unique_TMB_10perc'])) + ' +/- ' + str(np.std(MSI_DRs['unique_TMB_10perc'])))
print('AR mean num unique neoantigenic mutations in over 10% of tumor: ' + str(np.mean(MSI_ARs['unique_TMB_10perc'])) + ' +/- ' + str(np.std(MSI_ARs['unique_TMB_10perc'])))
print('NR mean num unique neoantigenic mutations in over 10% of tumor: ' + str(np.mean(MSI_NRs['unique_TMB_10perc'])) + ' +/- ' + str(np.std(MSI_NRs['unique_TMB_10perc'])))

slow_responders = MSIdata[(MSIdata['early_response']=='PR') * (MSIdata['response_tmax']=='CR')]
acquire_resisters = MSIdata[(MSIdata['early_response']=='PR') * (MSIdata['response_tmax']=='PD')]
stably_PR = MSIdata[(MSIdata['early_response']=='PR') * (MSIdata['response_tmax']=='PR')]

print('Slow CR median number of subclones: ' + str(np.median(slow_responders['num_subclones'])))
print('Acquired resistance median number of subclones: ' + str(np.median(acquire_resisters['num_subclones'])))
print('Stably PR median number of subclones: ' + str(np.median(stably_PR['num_subclones'])))

print('Count of Slow CR tumors: ' + str(len(slow_responders)))
print('Count of acquired resistance tumors: ' + str(len(acquire_resisters['num_subclones'])))
print('Count of stably PR tumors: ' + str(len(stably_PR['num_subclones'])))

MSIdata['is_clonal_neoant'] = MSIdata['is_clonal_neoant'].replace([0, 1], ['Absent', 'Present'])
MSSdata['is_clonal_neoant'] = MSSdata['is_clonal_neoant'].replace([0, 1], ['Absent', 'Present'])
absentdata = MSIdata[MSIdata['is_clonal_neoant']=='Absent']
presentdata = MSIdata[MSIdata['is_clonal_neoant']=='Present']
absentDRs = absentdata[absentdata['LTR']=='Durable Response']
presentDRs = presentdata[presentdata['LTR']=='Durable Response']

print('MMRD tumors with clonal neoantigen, fraction DR: ' + str(sum(presentdata['LTR']=='Durable Response')/len(presentdata['LTR'])))
print('MMRD tumors without clonal neoantigen, fraction DR: ' + str(sum(absentdata['LTR']=='Durable Response')/len(absentdata['LTR'])))
print('MMRD tumors without clonal neoantigen, fraction AR: ' + str(sum(absentdata['LTR']=='Acquired Resistance')/len(absentdata['LTR'])))
print('MMRD tumors without clonal neoantigen, fraction NR: ' + str(sum(absentdata['LTR']=='No Response')/len(absentdata['LTR'])))
print('MMRD tumors without clonal neoantigen, number DR: ' + str(sum(absentdata['LTR']=='Durable Response')))
print('MMRD tumors without clonal neoantigen, number AR: ' + str(sum(absentdata['LTR']=='Acquired Resistance')))
print('MMRD tumors without clonal neoantigen, number NR: ' + str(sum(absentdata['LTR']=='No Response')))
print('Fraction of MMRD DRs arising from tumor w/ clonal neoantigen: ' + str(sum((MSIdata['is_clonal_neoant']=='Present')*(MSIdata['LTR']=='Durable Response'))/sum(MSIdata['LTR']=='Durable Response')))
print('Fraction of MMRD DRs arising from tumor w/o clonal neoantigen: ' + str(sum((MSIdata['is_clonal_neoant']=='Absent')*(MSIdata['LTR']=='Durable Response'))/sum(MSIdata['LTR']=='Durable Response')))

print('Average num subclones of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(presentDRs['num_subclones']), 2)) + ' +/- ' + str(round(np.std(presentDRs['num_subclones']), 2)))
print('Average num subclones of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(absentDRs['num_subclones']), 2)) + ' +/- ' + str(round(np.std(absentDRs['num_subclones']), 2)))
print('Average tree index of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(presentDRs['tree_index']), 2)) + ' +/- ' + str(round(np.std(presentDRs['tree_index']), 2)))
print('Average tree index of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(absentDRs['tree_index']), 2)) + ' +/- ' + str(round(np.std(absentDRs['tree_index']), 2)))

print('Average weighted immunogenicity of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(presentDRs['axr_weighted_anteginicity']), 2)) + ' +/- ' + str(round(np.std(presentDRs['axr_weighted_anteginicity']), 2)))
print('Average weighted immunogenicity of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(absentDRs['axr_weighted_anteginicity']), 2)) + ' +/- ' + str(round(np.std(absentDRs['axr_weighted_anteginicity']), 2)))
print('Average max immunogenicity of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(presentDRs['maxNAquality']), 2)) + ' +/- ' + str(round(np.std(presentDRs['maxNAquality']), 2)))
print('Average max immunogenicity of MMRD DRs with clonal neoantigen: ' + str(round(np.mean(absentDRs['maxNAquality']), 2)) + ' +/- ' + str(round(np.std(absentDRs['maxNAquality']), 2)))


mmrp_absentdata = MSSdata[MSSdata['is_clonal_neoant']=='Absent']
mmrp_presentdata = MSSdata[MSSdata['is_clonal_neoant']=='Present']
print('Fraction MMRP tumors achieving durable response: ' + str(sum(MSSdata['LTR']=='Durable Response')/len(MSSdata['LTR'])))
print('MMRP tumors with clonal neoantigen, fraction DR: ' + str(sum(mmrp_presentdata['LTR']=='Durable Response')/len(mmrp_presentdata['LTR'])))
print('MMRP tumors without clonal neoantigen, fraction DR: ' + str(sum(mmrp_absentdata['LTR']=='Durable Response')/len(mmrp_absentdata['LTR'])))
print('Fraction of MMRP DRs arising from tumor w/ clonal neoantigen: ' + str(sum((MSSdata['is_clonal_neoant']=='Present')*(MSSdata['LTR']=='Durable Response'))/sum(MSSdata['LTR']=='Durable Response')))
print('Fraction of MMRP DRs arising from tumor w/o clonal neoantigen: ' + str(sum((MSSdata['is_clonal_neoant']=='Absent')*(MSSdata['LTR']=='Durable Response'))/sum(MSSdata['LTR']=='Durable Response')))

print('Fraction of MMRD tumors w/pseudoprogression: ' + str(sum(MSIdata['num_pseudoprogression_events']>0)/len(MSIdata['num_pseudoprogression_events'])))
print('Fraction of MMRD tumors, sampled every 8 weeks, w/pseudoprogression: ' + str(sum((MSIdata['w8_subsampled_num_pseudoprogression_events']>0)/len(MSIdata['w8_subsampled_num_pseudoprogression_events']))))
print('Fraction of MMRD tumors, sampled every 12 weeks, w/pseudoprogression: ' + str(sum((MSIdata['w12_subsampled_num_pseudoprogression_events']>0)/len(MSIdata['w12_subsampled_num_pseudoprogression_events']))))

MSIdata['minNAquality'] = MSIdata['minNAquality'].replace([0], [0.1])
MSIdata_minNA_gr0 = MSIdata.loc[MSIdata['minNAquality']>0.1]
MSIdata_minNA_eq0 = MSIdata.loc[MSIdata['minNAquality']<=0.1]

# Overall fraction in MMRP tumors
print('Fraction of MMRP tumors w/pseudoprogression: ' + str(sum((MSSdata['num_pseudoprogression_events']>0)/len(MSSdata['num_pseudoprogression_events']))))

# Early pseudoprogression
print('Fraction of MMRD tumors w/early pseudoprogression: ' + str(sum((MSIdata['early_events']>0)/len(MSIdata['early_events']))))
print('Fraction of MMRD tumors, sampled every 8 weeks, w/early pseudoprogression: ' + str(sum((MSIdata['w8_subsampled_early_events']>0)/len(MSIdata['w8_subsampled_early_events']))))
print('Fraction of MMRD tumors, sampled every 12 weeks, w/early pseudoprogression: ' + str(sum((MSIdata['w12_subsampled_early_events']>0)/len(MSIdata['w12_subsampled_early_events']))))

# Late pseudoprogression
print('Fraction of MMRD tumors w/late pseudoprogression: ' + str(sum((MSIdata['late_events']>0)/len(MSIdata['late_events']))))
print('Fraction of MMRD tumors, sampled every 8 weeks, w/late pseudoprogression: ' + str(sum((MSIdata['w8_subsampled_late_events']>0)/len(MSIdata['w8_subsampled_late_events']))))
print('Fraction of MMRD tumors, sampled every 12 weeks, w/late pseudoprogression: ' + str(sum((MSIdata['w12_subsampled_late_events']>0)/len(MSIdata['w12_subsampled_late_events']))))

# Median time to first pseudoprogression event
print('Median time of PsP (in weeks), when response evaluated daily: ' + str(np.median(MSIdata['time_to_first_psp'].loc[MSIdata['time_to_first_psp']>0])/7))
print('Median time of PsP (in weeks), when response evaluated every 8 weeks: ' + str(np.median(MSIdata['w8_subsampled_time_to_first_psp'].loc[MSIdata['time_to_first_psp']>0])/7))
print('Median time of PsP (in weeks), when response evaluated every 12 weeks: ' + str(np.median(MSIdata['w12_subsampled_time_to_first_psp'].loc[MSIdata['time_to_first_psp']>0])/7))

print('MMRD tumors with remaining founder, total number: ' + str(len(MSIdata_minNA_eq0)))
print('MMRD tumors with remaining founder, number NR: ' + str(len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['LTR']=='No Response'])))
print('MMRD tumors with remaining founder, number AR: ' + str(len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['LTR']=='Acquired Resistance'])))
print('MMRD tumors with remaining founder, number DR: ' + str(len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['LTR']=='Durable Response'])))

MSI_nofounder_DR = MSIdata_minNA_gr0.loc[MSIdata_minNA_gr0['LTR']=='Durable Response']
MSI_nofounder_AR = MSIdata_minNA_gr0.loc[MSIdata_minNA_gr0['LTR']=='Acquired Resistance']
MSI_nofounder_NR = MSIdata_minNA_gr0.loc[MSIdata_minNA_gr0['LTR']=='No Response']

print('MMRD NR tumors without remaining founder, mean weakest neoantigen: ' + str(np.mean(MSI_nofounder_NR['minNAquality'])) + ' +/- ' + str(np.std(MSI_nofounder_NR['minNAquality'])))
print('MMRD AR tumors without remaining founder, mean weakest neoantigen: ' + str(np.mean(MSI_nofounder_AR['minNAquality'])) + ' +/- ' + str(np.std(MSI_nofounder_AR['minNAquality'])))
print('MMRD DR tumors without remaining founder, mean weakest neoantigen: ' + str(np.mean(MSI_nofounder_DR['minNAquality'])) + ' +/- ' + str(np.std(MSI_nofounder_DR['minNAquality'])))

psp_types = ['NR', 'PsP', 'No PsP']
for i in range(len(MSIdata['response_pseud'])):
    if MSIdata['LTR'].loc[i + 1] == 'No Response':
        MSIdata['response_pseud'].loc[i + 1] = psp_types[0]
    elif MSIdata['num_pseudoprogression_events'].loc[i+1]==0:
        MSIdata['response_pseud'].loc[i+1] = psp_types[2]
    elif MSIdata['num_pseudoprogression_events'].loc[i+1]>0:
        MSIdata['response_pseud'].loc[i+1] = psp_types[1]
    else:
        print('what is happening at i=' + str(i) + '?')
MSIdata_minNA_gr0 = MSIdata.loc[MSIdata['minNAquality']>0.1]
MSIdata_minNA_eq0 = MSIdata.loc[MSIdata['minNAquality']<=0.1]

print('MMRD tumors with remaining founder, total number: ' + str(len(MSIdata_minNA_eq0)))
print('MMRD tumors with remaining founder, number NR: ' + str(len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['response_pseud']=='NR'])))
print('MMRD tumors with remaining founder, number w/ PsP: ' + str(len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['response_pseud']=='PsP'])))
print('MMRD tumors with remaining founder, number w/ no PsP: ' + str(len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['response_pseud']=='No PsP'])))

nonclonal_DRs = MSIdata.loc[(MSIdata['LTR']=='Durable Response') & (MSIdata['is_clonal_neoant']=='Absent')]
nonclonal_ARs = MSIdata.loc[(MSIdata['LTR']=='Acquired Resistance') & (MSIdata['is_clonal_neoant']=='Absent')]
nonclonal_NRs = MSIdata.loc[(MSIdata['LTR']=='No Response') & (MSIdata['is_clonal_neoant']=='Absent')]
clonal_DRs = MSIdata.loc[(MSIdata['LTR']=='Durable Response') & (MSIdata['is_clonal_neoant']=='Present')]
clonal_ARs = MSIdata.loc[(MSIdata['LTR']=='Acquired Resistance') & (MSIdata['is_clonal_neoant']=='Present')]
clonal_NRs = MSIdata.loc[(MSIdata['LTR']=='No Response') & (MSIdata['is_clonal_neoant']=='Present')]
MSInoclonal = MSIdata.loc[MSIdata['is_clonal_neoant']=='Absent']
MSIclonal = MSIdata.loc[MSIdata['is_clonal_neoant']=='Present']
MSI_DRs = MSIdata.loc[MSIdata['LTR']=='Durable Response']
print('Max neoant quality in DR tumors w/ clonal neoantigen: ' + str(np.mean(clonal_DRs['maxNAquality'])) + ' +/- ' + str(np.std(clonal_DRs['maxNAquality'])))
print('Max neoant quality in DR tumors w/o clonal neoantigen: ' + str(np.mean(nonclonal_DRs['maxNAquality'])) + ' +/- ' + str(np.std(nonclonal_DRs['maxNAquality'])))

#
#
#
#
#
#
#
# COHORT PLOTS
#
#
#
#
#
#
#
cohort_plot_path = path_base + 'cohort_plots/'
os.makedirs(cohort_plot_path, exist_ok=True) # create directory for plots of the entire simulated cohort

# Preliminary stuff for Fig 4
MSI_LTRvsBest = np.zeros((3, 4))
MSI_LTRvsInitial = np.zeros((3, 4))
recist_response_types = ['PD', 'SD', 'PR', 'CR']
LTR_response_types = ['No Response', 'Acquired Resistance', 'Durable Response']
for i in range(len(LTR_response_types)):
    for j in range(len(recist_response_types)):
        MSI_LTRvsBest[i, j] = sum((MSIdata['best_response']==recist_response_types[j])*(MSIdata['LTR']==LTR_response_types[i]))
        MSI_LTRvsInitial[i, j] = sum((MSIdata['response_12w']==recist_response_types[j])*(MSIdata['LTR']==LTR_response_types[i]))
raw_table_MSIvsInitial = np.copy(MSI_LTRvsInitial)
for j in range(len(recist_response_types)):
    MSI_LTRvsBest[:, j] = MSI_LTRvsBest[:, j]/sum(MSI_LTRvsBest[:, j]) * 100
    MSI_LTRvsInitial[:, j] = MSI_LTRvsInitial[:, j] / sum(MSI_LTRvsInitial[:, j]) * 100
orig_PR_labels = ['SlowCR']*len(slow_responders) + ['AcquiredResistance']*len(acquire_resisters) + ['StablyPR']*len(stably_PR)
orig_PR_df = pd.concat([slow_responders, acquire_resisters, stably_PR], ignore_index=True)
orig_PR_df['response_type_category'] = orig_PR_labels
# Set aside outliers for swarm/strip plot:
ar_toosmall = orig_PR_df[(orig_PR_df['response_type_category']=='AcquiredResistance') * (orig_PR_df['num_subclones']<2)]
ar_toobig = orig_PR_df[(orig_PR_df['response_type_category']=='AcquiredResistance') * (orig_PR_df['num_subclones']>6)]
pr_toobig = orig_PR_df[(orig_PR_df['response_type_category']=='StablyPR') * (orig_PR_df['num_subclones']>5)]
cr_toobig = orig_PR_df[(orig_PR_df['response_type_category']=='SlowCR') * (orig_PR_df['num_subclones']>5)]
outlier_df = pd.concat([ar_toosmall, ar_toobig, pr_toobig, cr_toobig], ignore_index=True)

# Fig 4(d) Heatmap
plt.rcParams['font.family'] = ['Arial']
plt.rcParams.update({'font.size': 14})
color_palette = ['thistle', 'orchid', 'purple']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
plt.title('MMR-D LTR vs. early response')
sns.heatmap(data=MSI_LTRvsInitial, xticklabels=['PD', 'SD', 'PR', 'CR'], yticklabels=['NR', 'AR', 'DR'], linewidth=0.5, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('12-week response, % originally labeled as')
plt.ylabel('Long Term Response')
plt.savefig(cohort_plot_path + "MSI_durability_response_12w_heatmap.png")
plt.savefig(cohort_plot_path + "MSI_durability_response_12w_heatmap.svg")

# Fig 4(d) Heterogeneity boxplot + outliers (beeswarm)
sns.set_palette(sns.color_palette(color_palette))
plt.figure()
plt.rcParams.update({'font.size': 16})
plt.title('Heterogeneity in MMR-D original PR responders')
ax = sns.boxplot(data=orig_PR_df, x='response_type_category', y='num_subclones', order=['AcquiredResistance', 'StablyPR', 'SlowCR'], showfliers=False)
add_stat_annotation(
    ax, data=orig_PR_df, x='response_type_category', y='num_subclones', order=['AcquiredResistance', 'StablyPR', 'SlowCR'],
    box_pairs=[("AcquiredResistance", "StablyPR"), ("AcquiredResistance", "SlowCR"), ("StablyPR", "SlowCR")],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
sns.swarmplot(data=outlier_df, x='response_type_category', y='num_subclones', order=['AcquiredResistance', 'StablyPR', 'SlowCR'], size=5, color='dimgray')
plt.xlabel('')
plt.ylabel('Number of subclones')
plt.ylim([0.5, 11])
plt.yticks(np.arange(1,9,step=1))
plt.savefig(cohort_plot_path + "origPRresp_heterogeneity.png")
plt.savefig(cohort_plot_path + "origPRresp_heterogeneity.svg")
plt.rcParams.update({'font.size': 14})

# Fig 4(c) right: countplot showing outcomes
NRfrac_MSI = sum(MSIdata['LTR']=='No Response')/len(MSIdata['LTR']) * 100
ARfrac_MSI = sum(MSIdata['LTR']=='Acquired Resistance')/len(MSIdata['LTR']) * 100
DRfrac_MSI = sum(MSIdata['LTR']=='Durable Response')/len(MSIdata['LTR']) * 100
NRfrac_MSS = sum(MSSdata['LTR']=='No Response')/len(MSSdata['LTR']) * 100
ARfrac_MSS = sum(MSSdata['LTR']=='Acquired Resistance')/len(MSSdata['LTR']) * 100
DRfrac_MSS = sum(MSSdata['LTR']=='Durable Response')/len(MSSdata['LTR']) * 100
LTR_MSI = {'LTR':['No Response', 'Acquired Resistance', 'Durable Response'],
           'frac': [NRfrac_MSI, ARfrac_MSI, DRfrac_MSI]}
MSI_LTR_df = pd.DataFrame(data=LTR_MSI)
LTR_MSS = {'LTR':['No Response', 'Acquired Resistance', 'Durable Response'],
           'frac': [NRfrac_MSS, ARfrac_MSS, DRfrac_MSS]}
MSS_LTR_df = pd.DataFrame(data=LTR_MSS)
LTRdf = pd.DataFrame(data={'LTR':['No Response', 'Acquired Resistance', 'Durable Response',
              'No Response', 'Acquired Resistance', 'Durable Response'],
           'frac': [NRfrac_MSI, ARfrac_MSI, DRfrac_MSI,
                    NRfrac_MSS, ARfrac_MSS, DRfrac_MSS],
       'type': ['MMR-D','MMR-D','MMR-D',
                'MMR-P','MMR-P','MMR-P']})
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure()
ax = sns.catplot(
    data=LTRdf, kind="bar",
    x="LTR", y="frac", col="type",
    height=4, aspect=0.6, edgecolor="k"
)
sns.despine(top=False, right=False)
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.90, wspace=0.05)
ax.set_axis_labels('', '')
ax.set_ylabels('Percent')
ax.set_titles('{col_name}')
plt.ylim([0, 69])
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(cohort_plot_path + "LTR_countplot_therapy.png")
plt.savefig(cohort_plot_path + "LTR_countplot_therapy.svg", format='svg')


# Figure 5(e) stacked barplot showing LTR by presence/absence of clonal neoantigen
NRfrac_absent = sum(absentdata['LTR']=='No Response')/len(absentdata['LTR']) * 100
ARfrac_absent = sum(absentdata['LTR']=='Acquired Resistance')/len(absentdata['LTR']) * 100
DRfrac_absent = sum(absentdata['LTR']=='Durable Response')/len(absentdata['LTR']) * 100
NRfrac_present = sum(presentdata['LTR']=='No Response')/len(presentdata['LTR']) * 100
ARfrac_present = sum(presentdata['LTR']=='Acquired Resistance')/len(presentdata['LTR']) * 100
DRfrac_present = sum(presentdata['LTR']=='Durable Response')/len(presentdata['LTR']) * 100
LTR_clonaldf = {'LTR':['No Response', 'Acquired Resistance', 'Durable Response',
                                         'No Response', 'Acquired Resistance', 'Durable Response'],
                                  'frac': [NRfrac_absent, ARfrac_absent, DRfrac_absent,
                                           NRfrac_present, ARfrac_present, DRfrac_present],
                                  'clonal_type': ['Absent','Absent','Absent',
                                           'Present','Present','Present']}
tumor_type = ('Absent', 'Present')
clonal_dict = {
    "NR": np.array([NRfrac_absent, NRfrac_present]),
    "AR": np.array([ARfrac_absent, ARfrac_present]),
    "DR": np.array([DRfrac_absent, DRfrac_present])
}
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = ['Arial']
plt.rc('legend', fontsize=11)
fig, ax = plt.subplots(figsize=(4, 4))
plt.subplots_adjust(left=0.2, right=0.95, bottom = 0.2, top = 0.90, wspace=0.05)
bottom = np.zeros(2)
iter = 0
for boolean, cur_ltr in clonal_dict.items():
    p = ax.bar(tumor_type, cur_ltr, width=0.75, label=boolean, bottom=bottom, edgecolor='k')
    bottom += cur_ltr
#ax.legend()
ax.set_xticklabels(['Absent', 'Present'])
plt.ylim([0, 100])
plt.ylabel('Percent')
plt.xlabel('Clonal neoantigen')
plt.savefig(cohort_plot_path + "LTR_countplot_clonality_improved_MSI.png")
plt.savefig(cohort_plot_path + "LTR_countplot_clonality_improved_MSI.svg", format='svg')
#
#
#
#
# Figure 3: Clinical cohort vs. simulated cohort comparisons
#
#
#
#
MMRD_sim_n_vec = np.array([sum(MSIdata['best_response']==i) for i in ['PD','SD','PR','CR']])
MMRD_sim_perc_vec = MMRD_sim_n_vec / sum(MMRD_sim_n_vec) * 100
MMRD_sim_yerrormin = []
MMRD_sim_yerrmax = []
MMRD_sim_errbar_magnitude = []
MMRD_clinical_n_vec = np.array([45, 30, 49, 20])  # keynote-177 values
MMRD_clinical_frac_vec = MMRD_clinical_n_vec / sum(MMRD_clinical_n_vec) * 100
MMRD_clinical_yerrmin = []
MMRD_clinical_yerrmax = []
MMRD_clinical_errbar_magnitude = []
for ci_i in range(len(MMRD_clinical_n_vec)):
    # Clinical data
    ci = binomtest(MMRD_clinical_n_vec[ci_i], sum(MMRD_clinical_n_vec), MMRD_clinical_frac_vec[ci_i] / 100).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
    MMRD_clinical_yerrmin.append(ci.low * 100)
    MMRD_clinical_yerrmax.append(ci.high * 100)
    MMRD_clinical_errbar_magnitude.append(ci.high - ci.low)  # scale by error bar size in clinical data
    # Simulated data
    ci_sim = binomtest(MMRD_sim_n_vec[ci_i], sum(MMRD_sim_n_vec), MMRD_sim_perc_vec[ci_i] / 100).proportion_ci()
    MMRD_sim_yerrormin.append(ci_sim.low * 100)
    MMRD_sim_yerrmax.append(ci_sim.high * 100)
    MMRD_sim_errbar_magnitude.append(ci_sim.high - ci_sim.low)  # scale by error bar size in simulated data
MMRD_clinical_yerr = [abs(MMRD_clinical_frac_vec - MMRD_clinical_yerrmin), abs(MMRD_clinical_frac_vec - MMRD_clinical_yerrmax)]
MMRD_sim_yerr = [abs(MMRD_sim_perc_vec - MMRD_sim_yerrormin), abs(MMRD_sim_perc_vec - MMRD_sim_yerrmax)]
clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MMRD_clinical_frac_vec[0], MMRD_clinical_frac_vec[1], MMRD_clinical_frac_vec[2], MMRD_clinical_frac_vec[3],
                   MMRD_sim_perc_vec[0], MMRD_sim_perc_vec[1], MMRD_sim_perc_vec[2], MMRD_sim_perc_vec[3]]}
clincomp_df = pd.DataFrame(data = clincomp)

# Figure 3(a) MMRD/MSI-H results, best response during therapy
MSIMSS_color_palette = ['seagreen', 'mediumaquamarine']
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='best_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_loc = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_loc[0:4], y=y_loc[0:4], yerr=MMRD_clinical_yerr, fmt='.', c='k')
ax.errorbar(x=x_loc[4:], y=y_loc[4:], yerr=MMRD_sim_yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None, loc='upper right')
plt.xlabel('')
plt.ylim([0, 50])
plt.ylabel('Percent')
plt.title('MMR-D best response')
plt.savefig(cohort_plot_path + "clincomp_barplot_best_MMRD.png")
plt.savefig(cohort_plot_path + "clincomp_barplot_best_MMRD.svg", format='svg')

# Fig 3(c) right: MMR-P clinical vs sim data at 12 weeks
MMRP_sim_n_vec = np.array([sum(MSSdata['response_12w']==i) for i in ['PD','SD','PR','CR']])
MMRP_sim_perc_vec = MMRP_sim_n_vec / sum(MMRP_sim_n_vec) * 100
MMRP_sim_yerrmin = []
MMRP_sim_yerrmax = []
MMRP_sim_errbar_magnitude = []
MMRP_clinical_n_vec = np.array([11, 2, 0, 0])
MMRP_clinical_frac_vec = MMRP_clinical_n_vec / sum(MMRP_clinical_n_vec) * 100
nMSS = sum(MMRP_clinical_n_vec)
MMRP_clinical_yerrmin = []
MMRP_clinical_yerrmax = []
MMRP_clinical_errbar_magnitude = []
for ci_i in range(len(MMRP_clinical_n_vec)):
    # clinical data
    ci = binomtest(MMRP_clinical_n_vec[ci_i], sum(MMRP_clinical_n_vec), MMRP_clinical_frac_vec[ci_i] / 100).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
    MMRP_clinical_yerrmin.append(ci.low * 100)
    MMRP_clinical_yerrmax.append(ci.high * 100)
    MMRP_clinical_errbar_magnitude.append(ci.high - ci.low)  # scale by error bar size in clinical data
    # simulation data
    ci_sim = binomtest(MMRP_sim_n_vec[ci_i], sum(MMRP_sim_n_vec), MMRP_sim_perc_vec[ci_i] / 100).proportion_ci()
    MMRP_sim_yerrmin.append(ci_sim.low * 100)
    MMRP_sim_yerrmax.append(ci_sim.high * 100)
    MMRP_sim_errbar_magnitude.append(ci_sim.high - ci_sim.low)  # scale by error bar size in clinical data
MMRP_clinical_yerr = [abs(MMRP_clinical_frac_vec - MMRP_clinical_yerrmin), abs(MMRP_clinical_frac_vec - MMRP_clinical_yerrmax)]
MMRP_sim_yerr = [abs(MMRP_sim_perc_vec - MMRP_sim_yerrmin), abs(MMRP_sim_perc_vec - MMRP_sim_yerrmax)]
## order {"PD", "SD", "PR", "CR"}
clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'early_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MMRP_clinical_frac_vec[0], MMRP_clinical_frac_vec[1], MMRP_clinical_frac_vec[2], MMRP_clinical_frac_vec[3],
                   MMRP_sim_perc_vec[0], MMRP_sim_perc_vec[1], MMRP_sim_perc_vec[2], MMRP_sim_perc_vec[3]]}
clincomp_df = pd.DataFrame(data = clincomp)
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='early_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_loc = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_loc[0:4], y=y_loc[0:4], yerr=MMRP_clinical_yerr, fmt='.', c='k')
ax.errorbar(x=x_loc[4:], y=y_loc[4:], yerr=MMRP_sim_yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.title('12-week MMR-P response')
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSS.png")
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSS.svg", format='svg')

# Fig 3(c) left: MMR-D clinical vs sim data at 12 weeks
MMRD_12w_sim_n_vec = np.array([sum(MSIdata['response_12w']==i) for i in ['PD','SD','PR','CR']])
MMRD_12w_sim_perc_vec = MMRD_12w_sim_n_vec / sum(MMRD_12w_sim_n_vec) * 100
MMRD_12w_sim_yerrmin = []
MMRD_12w_sim_yerrmax = []
MMRD_12w_sim_errbar_magnitude = []
MMRD_12w_clinical_n_vec = np.array([1, 5, 4, 0])
MMRD_12w_clinical_perc_vec = MMRD_12w_clinical_n_vec / sum(MMRD_12w_clinical_n_vec) * 100
MMRD_12w_clinical_yerrmin = []
MMRD_12w_clinical_yerrmax = []
MMRD_12w_clinical_errbar_magnitude = []
for ci_i in range(len(MMRD_12w_clinical_n_vec)):
    # clinical data
    ci = binomtest(MMRD_12w_clinical_n_vec[ci_i], sum(MMRD_12w_clinical_n_vec), MMRD_12w_clinical_perc_vec[ci_i] / 100).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
    MMRD_12w_clinical_yerrmin.append(ci.low * 100)
    MMRD_12w_clinical_yerrmax.append(ci.high * 100)
    MMRD_12w_clinical_errbar_magnitude.append(ci.high - ci.low)  # scale by error bar size in clinical data
    # simulation data
    ci_sim = binomtest(MMRD_12w_sim_n_vec[ci_i], sum(MMRD_12w_sim_n_vec), MMRD_12w_sim_perc_vec[ci_i] / 100).proportion_ci()
    MMRD_12w_sim_yerrmin.append(ci_sim.low * 100)
    MMRD_12w_sim_yerrmax.append(ci_sim.high * 100)
    MMRD_12w_sim_errbar_magnitude.append(ci_sim.high - ci_sim.low)  # scale by error bar size in clinical data
MMRD_12w_clinical_yerr = [abs(MMRD_12w_clinical_perc_vec - MMRD_12w_clinical_yerrmin), abs(MMRD_12w_clinical_perc_vec - MMRD_12w_clinical_yerrmax)]
MMRD_12w_sim_yerr = [abs(MMRD_12w_sim_perc_vec - MMRD_12w_sim_yerrmin), abs(MMRD_12w_sim_perc_vec - MMRD_12w_sim_yerrmax)]
## order {"PD", "SD", "PR", "CR"}
clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'early_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MMRD_12w_clinical_perc_vec[0], MMRD_12w_clinical_perc_vec[1], MMRD_12w_clinical_perc_vec[2], MMRD_12w_clinical_perc_vec[3],
                   MMRD_12w_sim_perc_vec[0], MMRD_12w_sim_perc_vec[1], MMRD_12w_sim_perc_vec[2], MMRD_12w_sim_perc_vec[3]]}
clincomp_df = pd.DataFrame(data = clincomp)
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='early_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_loc = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_loc[0:4], y=y_loc[0:4], yerr=MMRD_12w_clinical_yerr, fmt='.', c='k')
ax.errorbar(x=x_loc[4:], y=y_loc[4:], yerr=MMRD_12w_sim_yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
plt.title('12-week MMR-D response')
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSI.png")
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSI.svg", format='svg')

# Fig 3(b) PFS (time-to-progression data)
ttpData = [i/30.4368 for i in MSIdata['time_to_progression']]
sim_med_pfs = np.median(ttpData) # divide by 30.4 to get days -> months
sim_mean_pfs = np.average(ttpData)
sim_36mo_pfs = sum([i > 36 for i in ttpData])/len(ttpData) * 100
clinical_median_pfs = 16.5 # hardcoded for keynote-177
clinical_month36_pfs = 42.3 # hardcoded for keynote-177
clin_med_yerrmin = abs(clinical_median_pfs - 4) # 95% CI was 4 - 38.1 in final analysis of KEYNOTE-177
clin_med_yerrmax = abs(clinical_median_pfs - 38.1)
clin_med_yerr = [[clin_med_yerrmin], [clin_med_yerrmax]]
clin_36mo_yerrmin = abs(clinical_month36_pfs - 34)
clin_36mo_yerrmax = abs(clinical_month36_pfs - 50.4)
clin_36mo_yerr = [[clin_36mo_yerrmin], [clin_36mo_yerrmax]]
# how to get 95% CI of a median? Bootstrapping! Please see CIbootstrap.py
# hard-coding that value here:
sim_ci_min = 6.24
sim_ci_max = 30.39
sim_med_yerrmin = abs(sim_med_pfs - sim_ci_min)
sim_med_yerrmax = abs(sim_med_pfs - sim_ci_max)
sim_med_yerr = [[sim_med_yerrmin], [sim_med_yerrmax]]
# formula from: Practical Nonparametric Statistics, 3rd Edition by W.J. Conover
# month 36 PFS is binomial (independent Bernoulli trials) -- use CLopper-Pearson for CI
sim_ci = binomtest(sum([i > 36 for i in ttpData]), len(ttpData), sim_36mo_pfs/100).proportion_ci()
sim_36mo_yerrmin = abs(sim_36mo_pfs - sim_ci[0]*100)
sim_36mo_yerrmax = abs(sim_36mo_pfs - sim_ci[1]*100)
sim_36mo_yerr = [[sim_36mo_yerrmin], [sim_36mo_yerrmax]]
clincomp = {'Source': ['Clinical', 'Simulation', 'Clinical', 'Simulation'], #, 'Clinical', 'Simulation'],
            'PFS_type': ['Median', 'Median', '36-month', '36-month'], # '24-month', '24-month', '36-month', '36-month'],
           'PFS': [clinical_median_pfs, sim_med_pfs, clinical_month36_pfs, sim_36mo_pfs]} # clinical_month24_pfs, month24_pfs, clinical_month36_pfs, month36_pfs]}
clincomp_df = pd.DataFrame(data = clincomp)
fig, ax = plt.subplots(1, 2, figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax_0 = sns.barplot(data=clincomp_df[2:], x='PFS_type', y='PFS', hue='Source', ax=ax[0])
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax_0.patches]
y_loc = [p.get_height() for p in ax_0.patches]
ax[0].errorbar(x=x_loc[0], y=y_loc[0], yerr=clin_36mo_yerr, fmt='.', c='k')
ax[0].errorbar(x=x_loc[1], y=y_loc[1], yerr=sim_36mo_yerr, fmt='.', c='k')
ax[0].set_xticklabels('')
ax[0].set_xlabel('36-month')
ax[0].set_ylabel('Percent')
ax[0].legend(loc = 'upper right', fontsize=12)
ax[0].set_ylim([0, 70])
ax[0].set_title('MMR-D PFS')

ax_1 = sns.barplot(data=clincomp_df[:2], x='PFS_type', y='PFS', hue='Source', ax=ax[1])
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax_1.patches]
y_loc = [p.get_height() for p in ax_1.patches]
jitter = np.random.normal(0,0.02, len(ttpData))
ax[1].scatter([x_loc[1]] * len(ttpData) + jitter, ttpData, s=2, alpha = 0.2, color='dimgray')
ax[1].errorbar(x=x_loc[0], y=y_loc[0], yerr=clin_med_yerr, fmt='.', c='k')
ax[1].errorbar(x=x_loc[1], y=y_loc[1], yerr=sim_med_yerr, fmt='.', c='k')
ax[1].set_xlabel('Duration')
ax[1].set_xticklabels('')
ax[1].set_ylabel('Months')
ax[1].set_ylim([0, 70])
ax[1].set_title('MMR-D PFS')
ax[1].get_legend().remove()
plt.savefig(cohort_plot_path + "clincomp_PFS.png")
plt.savefig(cohort_plot_path + "clincomp_PFS.svg", format='svg')
#
#
#
#
#
# Supplementary Figure 4
#
#
#
#
#
# Looking only at MMR-D tumors with a durable response to therapy (Supplementary Figure 4)
clonal_types = ['Absent', 'Present']

# Supplementary Figure 4(b) - left
color_palette = ['aliceblue','dodgerblue'] #['lightcyan', 'darkturquoise']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='maxNAquality', order=['Absent', 'Present'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSI_DRs, x='is_clonal_neoant', y='maxNAquality', order=['Absent', 'Present'],
#     box_pairs=[('Absent', 'Present')],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSI_DRs.columns)
for i in range(len(clonal_types)):
    minv, maxv = np.percentile(MSI_DRs.loc[MSI_DRs['is_clonal_neoant']==clonal_types[i]]['maxNAquality'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSI_DRs.loc[(MSI_DRs['is_clonal_neoant']==clonal_types[i]) & ((MSI_DRs['maxNAquality'] < minv) | (MSI_DRs['maxNAquality'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='is_clonal_neoant', y='maxNAquality', order=['Absent', 'Present'], size=2, jitter=0.04, color='dimgray')
#plt.yscale('log')
#plt.ylim([0, 2900])
plt.ylabel('Maximal neoantigen quality')
ax.tick_params(axis='y', labelsize=12)
plt.xlabel('')
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_maxNAquality.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_maxNAquality.svg", format='svg')

# Supplementary Figure 4(b) - right
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='axr_weighted_anteginicity', order=['Absent', 'Present'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSI_DRs, x='is_clonal_neoant', y='axr_weighted_anteginicity', order=['Absent', 'Present'],
#     box_pairs=[('Absent', 'Present')],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSI_DRs.columns)
for i in range(len(clonal_types)):
    minv, maxv = np.percentile(MSI_DRs.loc[MSI_DRs['is_clonal_neoant']==clonal_types[i]]['axr_weighted_anteginicity'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSI_DRs.loc[(MSI_DRs['is_clonal_neoant']==clonal_types[i]) & ((MSI_DRs['axr_weighted_anteginicity'] < minv) | (MSI_DRs['axr_weighted_anteginicity'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='is_clonal_neoant', y='axr_weighted_anteginicity', order=['Absent', 'Present'], size=2, jitter=0.04, color='dimgray')
plt.xlabel('')
plt.ylabel('Weighted mean antigenicity')
# plt.ylim([0, 1680])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_axrmean.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_axrmean.svg", format='svg')

MSI_DRs_minNA_gr0 = MSI_DRs.loc[MSI_DRs['minNAquality']>0.1]
MSI_DRs_minNA_eq0 = MSI_DRs.loc[MSI_DRs['minNAquality']<=0.1]
print('MMRD DR tumors with remaining founder, total number: ' + str(len(MSI_DRs_minNA_eq0)))
print('MMRD DR tumors with remaining founder, number NR: ' + str(len(MSI_DRs_minNA_eq0.loc[MSI_DRs_minNA_eq0['response_pseud']=='NR'])))
print('MMRD DR tumors with remaining founder, number w/ PsP: ' + str(len(MSI_DRs_minNA_eq0.loc[MSI_DRs_minNA_eq0['response_pseud']=='PsP'])))
print('MMRD DR tumors with remaining founder, number w/ no PsP: ' + str(len(MSI_DRs_minNA_eq0.loc[MSI_DRs_minNA_eq0['response_pseud']=='No PsP'])))

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs_minNA_gr0, x='is_clonal_neoant', y='minNAquality', order=['Absent', 'Present'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSI_DRs_minNA_gr0, x='is_clonal_neoant', y='minNAquality', order=['Absent', 'Present'],
#     box_pairs=[('Absent', 'Present')],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSI_DRs_minNA_gr0.columns)
for i in range(len(clonal_types)):
    minv, maxv = np.percentile(MSI_DRs_minNA_gr0.loc[MSI_DRs_minNA_gr0['is_clonal_neoant']==clonal_types[i]]['minNAquality'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSI_DRs_minNA_gr0.loc[(MSI_DRs_minNA_gr0['is_clonal_neoant']==clonal_types[i]) & ((MSI_DRs_minNA_gr0['minNAquality'] < minv) | (MSI_DRs_minNA_gr0['minNAquality'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='is_clonal_neoant', y='minNAquality', order=['Absent', 'Present'], size=2, jitter=0.04, color='dimgray')
ax.set_yscale('log')
plt.xlabel('')
plt.ylabel('Weakest neoantigen')
# plt.ylim([0, 1680])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_weakestneoant.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_weakestneoant.svg", format='svg')

# Supplementary Figure 4(a) - right
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='tree_index', order=['Absent', 'Present'], showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSI_DRs, x='is_clonal_neoant', y='tree_index', order=['Absent', 'Present'],
#     box_pairs=[('Absent', 'Present')],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSI_DRs.columns)
for i in range(len(clonal_types)):
    minv, maxv = np.percentile(MSI_DRs.loc[MSI_DRs['is_clonal_neoant']==clonal_types[i]]['tree_index'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSI_DRs.loc[(MSI_DRs['is_clonal_neoant']==clonal_types[i]) & ((MSI_DRs['tree_index'] < minv) | (MSI_DRs['tree_index'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='is_clonal_neoant', y='tree_index', order=['Absent', 'Present'], size=2, jitter=0.04, color='dimgray')
plt.ylim([0.1, 0.7])
plt.xlabel('')
plt.ylabel('Tree index')
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_treeindex.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_treeindex.svg", format='svg')

# Supplementary Figure 4(a) - left
absent_toosmall = MSI_DRs[(MSI_DRs['is_clonal_neoant']=='Absent') * (MSI_DRs['num_subclones']<2)]
absent_toobig = MSI_DRs[(MSI_DRs['is_clonal_neoant']=='Absent') * (MSI_DRs['num_subclones']>7)]
pres_toobig = MSI_DRs[(MSI_DRs['is_clonal_neoant']=='Present') * (MSI_DRs['num_subclones']>6)]
outlier_df = pd.concat([absent_toosmall, absent_toobig, pres_toobig], ignore_index=True)
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxplot(data=MSI_DRs, x='is_clonal_neoant', y='num_subclones', order=['Absent', 'Present'], showfliers=False)
# add_stat_annotation(
#     ax, data=MSI_DRs, x='is_clonal_neoant', y='num_subclones', order=['Absent', 'Present'],
#     box_pairs=[('Absent', 'Present')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
sns.swarmplot(data=outlier_df, x='is_clonal_neoant', y='num_subclones', order=['Absent', 'Present'], size=5, color='dimgray')
plt.xlabel('')
plt.ylabel('Number of subclones')
plt.ylim([1, 9.25])
plt.savefig(cohort_plot_path + "DR_MSI_heterogeneity.png")
plt.savefig(cohort_plot_path + "DR_MSI_heterogeneity.svg")
#
#
#
# FIGURE 5 WITH OUTLIERS
#
#
#
plt.close('all')
# Figure 5a
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(LTR_response_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['LTR']==LTR_response_types[i]]['maxNAquality'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['LTR']==LTR_response_types[i]) & ((MSIdata['maxNAquality'] < minv) | (MSIdata['maxNAquality'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'], size=2, jitter=0.04, color='dimgray')
plt.ylabel('Maximal neoantigen quality')
ax.set_yscale('log')
ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "Fig5a.png", format='png')
plt.savefig(cohort_plot_path + "Fig5a.svg", format='svg')

# Figure 5b
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(LTR_response_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['LTR']==LTR_response_types[i]]['axr_weighted_anteginicity'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['LTR']==LTR_response_types[i]) & ((MSIdata['axr_weighted_anteginicity'] < minv) | (MSIdata['axr_weighted_anteginicity'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'], size=2, jitter=0.04, color='dimgray')
plt.xlabel('')
plt.ylabel('Weighted mean antigenicity')
# plt.ylim([0, 1950])
ax.set_yscale('log')
ax.set_xticklabels(['NR', 'AR', 'DR'])
# ax.set_yticks([0, 500, 1000, 1500])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "Fig5b.png", format='png')
plt.savefig(cohort_plot_path + "Fig5b.svg", format='svg')

# Figure 5c
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='unique_TMB_10perc', order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='unique_TMB_10perc', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(LTR_response_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['LTR']==LTR_response_types[i]]['unique_TMB_10perc'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['LTR']==LTR_response_types[i]) & ((MSIdata['unique_TMB_10perc'] < minv) | (MSIdata['unique_TMB_10perc'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='LTR', y='unique_TMB_10perc', order=['No Response', 'Acquired Resistance', 'Durable Response'], size=2, jitter=0.04, color='dimgray')
plt.xlabel('')
#plt.yticks([0.5e6, 1e6, 1.5e6, 2e6], ['0.5e6', '1e6', '1.5e6', '2e6'])
plt.ylabel('Number of neoantigenic mutations')
ax.set_xticklabels(['NR', 'AR', 'DR'])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "Fig5c.png")
plt.savefig(cohort_plot_path + "Fig5c.svg")

# Figure 5d
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(LTR_response_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['LTR']==LTR_response_types[i]]['tree_index'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['LTR']==LTR_response_types[i]) & ((MSIdata['tree_index'] < minv) | (MSIdata['tree_index'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'], size=2, jitter=0.04, color='dimgray')
# plt.ylim([0.2, 0.8])
plt.xlabel('')
plt.ylabel('Tree index')
ax.set_xticklabels(['NR', 'AR', 'DR'])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "Fig5d.png")
plt.savefig(cohort_plot_path + "Fig5d.svg")

# Fig 5f
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata_minNA_gr0, x='LTR', y='minNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata_minNA_gr0, x='LTR', y='minNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
outlier_df = pd.DataFrame(columns=MSIdata_minNA_gr0.columns)
for i in range(len(LTR_response_types)):
    minv, maxv = np.percentile(MSIdata_minNA_gr0.loc[MSIdata['LTR']==LTR_response_types[i]]['minNAquality'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata_minNA_gr0.loc[(MSIdata_minNA_gr0['LTR']==LTR_response_types[i]) & ((MSIdata_minNA_gr0['minNAquality'] < minv) | (MSIdata_minNA_gr0['minNAquality'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='LTR', y='minNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'], size=2, jitter=0.04, color='dimgray')
ax.set_yscale('log')
plt.xlabel('')
plt.ylabel('Weakest neoantigen')
ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(cohort_plot_path + "Fig5f.png")
plt.savefig(cohort_plot_path + "Fig5f.svg")

tot = len(MSIdata_minNA_eq0)
ct_NR = len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['LTR']=='No Response'])/tot * 100
ct_AR = len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['LTR']=='Acquired Resistance'])/tot * 100
ct_DR = len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['LTR']=='Durable Response'])/tot * 100
#
#
#
#
#
#
# Supplementary Figure 3
#
#
#
#
#
#
#
# Supplementary Figure 5(a) MMRD PsP Frequency, clinical + sim + sim (8w)
clinical_total_pts = 61
clinical_num_psp = 9
clinical_frac = clinical_num_psp/clinical_total_pts
ci = binomtest(clinical_num_psp, clinical_total_pts, clinical_frac).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
clinical_yerrmin = abs(clinical_frac - ci.low) * 100
clinical_yerrmax = abs(clinical_frac - ci.high) * 100
clinical_errbar_magnitude = ci.high - ci.low  # scale by error bar size in clinical data
clin_yerr = [[clinical_yerrmin], [clinical_yerrmax]]
our_total_pts = 5000
our_num_psp = sum(MSIdata['num_pseudoprogression_events']>0)
our_frac = our_num_psp/our_total_pts
ci = binomtest(our_num_psp, our_total_pts, our_frac).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
our_yerrmin = abs(our_frac - ci.low) * 100
our_yerrmax = abs(our_frac - ci.high) * 100
our_errbar_magnitude = ci.high - ci.low  # scale by error bar size in clinical data
our_yerr = [[our_yerrmin], [our_yerrmax]]
our_8w_total_pts = 5000
our_8w_num_psp = sum(MSIdata['w8_subsampled_num_pseudoprogression_events']>0)
our_8w_frac = our_8w_num_psp/our_8w_total_pts
ci = binomtest(our_8w_num_psp, our_8w_total_pts, our_8w_frac).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
our_8w_yerrmin = abs(our_8w_frac - ci.low) * 100
our_8w_yerrmax = abs(our_8w_frac - ci.high) * 100
our_8w_errbar_magnitude = ci.high - ci.low  # scale by error bar size in clinical data
our_8w_yerr = [[our_8w_yerrmin], [our_8w_yerrmax]]
clincomp = {'Type': ['Clinical', 'Sim', 'Sim (8w)'],
           'PsP': [clinical_frac*100, our_frac*100, our_8w_frac*100]} # clinical_month24_pfs, month24_pfs, clinical_month36_pfs, month36_pfs]}
clincomp_df = pd.DataFrame(data = clincomp)
fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(palette='Greys'))
ax = sns.barplot(data=clincomp_df, x='Type', y='PsP', edgecolor='k')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_loc = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_loc[0], y=y_loc[0], yerr=clin_yerr, fmt='.', c='k')
ax.errorbar(x=x_loc[1], y=y_loc[1], yerr=our_yerr, fmt='.', c='k')
ax.errorbar(x=x_loc[2], y=y_loc[2], yerr=our_8w_yerr, fmt='.', c='k')
ax.set_xlabel('')
ax.set_ylabel('Percent')
ax.set_ylim([0, 30])
ax.set_title('MMR-D PsP Frequency')
plt.savefig(cohort_plot_path + "clincomp_PSP.png")
plt.savefig(cohort_plot_path + "clincomp_PSP.svg", format='svg')

# Sort by NR, No PsP, and PsP
psp_types = ['NR', 'PsP', 'No PsP']
for i in range(len(MSIdata['response_pseud'])):
    if MSIdata['LTR'].loc[i + 1] == 'No Response':
        MSIdata['response_pseud'].loc[i + 1] = psp_types[0]
    elif MSIdata['num_pseudoprogression_events'].loc[i+1]==0:
        MSIdata['response_pseud'].loc[i+1] = psp_types[2]
    elif MSIdata['num_pseudoprogression_events'].loc[i+1]>0:
        MSIdata['response_pseud'].loc[i+1] = psp_types[1]
    else:
        print('what is happening at i=' + str(i) + '?')

print('Number of MMRD tumors with NR: ' + str(len(MSIdata[MSIdata['response_pseud']=='NR'])))
print('Number of MMRD tumors with PsP: ' + str(len(MSIdata[MSIdata['response_pseud']=='PsP'])))
print('Number of MMRD tumors with No PsP: ' + str(len(MSIdata[MSIdata['response_pseud']=='No PsP'])))

# Supplementary Figure 5(b) - max NA quality
color_palette = ['palegreen','chartreuse', 'green']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='response_pseud', y='maxNAquality', order=['NR', 'PsP', 'No PsP'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='response_pseud', y='maxNAquality', order=['NR', 'PsP', 'No PsP'],
#     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# Result:
# NR v.s. PsP: Welch's t-test independent samples with Bonferroni correction, P_val=3.792e-01 stat=1.529e+00
# PsP v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=1.226e-08 stat=-5.913e+00
# NR v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=4.883e-06 stat=-4.805e+00
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(psp_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['response_pseud']==psp_types[i]]['maxNAquality'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['response_pseud']==psp_types[i]) & ((MSIdata['maxNAquality'] < minv) | (MSIdata['maxNAquality'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='response_pseud', y='maxNAquality', order=['NR', 'PsP', 'No PsP'], size=2, jitter=0.04, color='dimgray')
plt.ylabel('Maximal neoantigen quality')
plt.ylim([1e0, 5e3])
ax.set_yscale('log')
ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "psp_brokendown_max_qual.png", format='png')
plt.savefig(cohort_plot_path + "psp_brokendown_max_qual.svg", format='svg')

# Supplementary Figure 5(c) - weighted NA quality
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='response_pseud', y='axr_weighted_anteginicity', order=['NR', 'PsP', 'No PsP'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='response_pseud', y='axr_weighted_anteginicity', order=['NR', 'PsP', 'No PsP'],
#     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# Result:
# NR v.s. PsP: Welch's t-test independent samples with Bonferroni correction, P_val=1.000e+00 stat=6.654e-01
# PsP v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=7.471e-26 stat=-1.074e+01
# NR v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=1.721e-29 stat=-1.146e+01
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(psp_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['response_pseud']==psp_types[i]]['axr_weighted_anteginicity'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['response_pseud']==psp_types[i]) & ((MSIdata['axr_weighted_anteginicity'] < minv) | (MSIdata['axr_weighted_anteginicity'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='response_pseud', y='axr_weighted_anteginicity', order=['NR', 'PsP', 'No PsP'], size=2, jitter=0.04, color='dimgray')
plt.ylabel('Weighted mean antigenicity')
plt.ylim([1e-1, 5e3])
ax.set_yscale('log')
ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "psp_brokendown_mean_qual.png", format='png')
plt.savefig(cohort_plot_path + "psp_brokendown_mean_qual.svg", format='svg')

# Supp Fig 5(e) - Tree index
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='response_pseud', y='tree_index', order=['NR', 'PsP', 'No PsP'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='response_pseud', y='tree_index', order=['NR', 'PsP', 'No PsP'],
#     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# Result:
# NR v.s. PsP: Welch's t-test independent samples with Bonferroni correction, P_val=8.202e-30 stat=1.171e+01
# PsP v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=4.880e-06 stat=4.815e+00
# NR v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=3.273e-112 stat=2.336e+01
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(psp_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['response_pseud']==psp_types[i]]['tree_index'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['response_pseud']==psp_types[i]) & ((MSIdata['tree_index'] < minv) | (MSIdata['tree_index'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='response_pseud', y='tree_index', order=['NR', 'PsP', 'No PsP'], size=2, jitter=0.04, color='dimgray')
plt.ylabel('Tree index')
plt.ylim([0.1, 0.75])
ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "psp_brokendown_tree_index.png", format='png')
plt.savefig(cohort_plot_path + "psp_brokendown_tree_index.svg", format='svg')

# Supplementary Figure 5(d) - number of clonal neoantigens
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata, x='response_pseud', y='num_clonal_neoant', order=['NR', 'PsP', 'No PsP'],
                 showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='response_pseud', y='num_clonal_neoant', order=['NR', 'PsP', 'No PsP'],
#     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# Result:
# NR v.s. PsP: Welch's t-test independent samples with Bonferroni correction, P_val=1.272e-53 stat=-1.634e+01
# PsP v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=5.875e-10 stat=-6.398e+00
# NR v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=3.149e-170 stat=-2.929e+01
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(psp_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['response_pseud']==psp_types[i]]['num_clonal_neoant'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['response_pseud']==psp_types[i]) & ((MSIdata['num_clonal_neoant'] < minv) | (MSIdata['num_clonal_neoant'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='response_pseud', y='num_clonal_neoant', order=['NR', 'PsP', 'No PsP'], size=2, jitter=0.04, color='dimgray')
plt.ylabel('Number of clonal neoantigens')
#plt.ylim([0.1, 0.75])
ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "psp_brokendown_num_clonal_neoant.png", format='png')
plt.savefig(cohort_plot_path + "psp_brokendown_num_clonal_neoant.svg", format='svg')

# Supplementary Figure 5(f) - weakest neoantigen quality
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxenplot(data=MSIdata_minNA_gr0, x='response_pseud', y='minNAquality', order=['NR', 'PsP', 'No PsP'], showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='response_pseud', y='minNAquality', order=['NR', 'PsP', 'No PsP'],
#     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# Result:
# NR v.s. PsP: Welch's t-test independent samples with Bonferroni correction, P_val=2.074e-76 stat=-2.036e+01
# PsP v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=2.333e-53 stat=-1.579e+01
# NR v.s. No PsP: Welch's t-test independent samples with Bonferroni correction, P_val=5.615e-56 stat=-1.620e+01
outlier_df = pd.DataFrame(columns=MSIdata_minNA_gr0.columns)
for i in range(len(psp_types)):
    minv, maxv = np.percentile(MSIdata_minNA_gr0.loc[MSIdata_minNA_gr0['response_pseud']==psp_types[i]]['minNAquality'], [3.125, 96.875])
    outlier_df = pd.concat([outlier_df, MSIdata_minNA_gr0.loc[(MSIdata_minNA_gr0['response_pseud']==psp_types[i]) & ((MSIdata_minNA_gr0['minNAquality'] < minv) | (MSIdata_minNA_gr0['minNAquality'] > maxv))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='response_pseud', y='minNAquality', order=['NR', 'PsP', 'No PsP'], size=2, jitter=0.04, color='dimgray')
ax.set_yscale('log')
plt.xlabel('')
plt.ylabel('Weakest neoantigen quality')
ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
plt.savefig(cohort_plot_path + "psp_brokendown_minNAquality.png")
plt.savefig(cohort_plot_path + "psp_brokendown_minNAquality.svg")
#
#
#
#
#
#
#
#
#
# Objective Response binary classification
#
#
#
# Load in clinical data from R script:
# Objective response group:
filename = 'clin_responsegroup_data.txt'
response_group = []
with open(path_to_clin_data + filename, 'r') as f:
    content = f.readlines()
    for i in content:
        str = i.strip()
        str = str.replace('"', '')
        response_group.append(str)
# Max clonal AxR score:
filename = 'clin_maxclonalAxR_data.txt'
clin_maxclonal = []
with open(path_to_clin_data + filename, 'r') as f:
    content = f.readlines()
    for i in content:
        num = float(''.join(list(i)[:-1]))
        clin_maxclonal.append(num)
# Max clonal AxR score:
filename = 'clin_maxAxR_data.txt'
clin_max = []
with open(path_to_clin_data + filename, 'r') as f:
    content = f.readlines()
    for i in content:
        num = float(''.join(list(i)[:-1]))
        clin_max.append(num)
# Preliminaries:
plt.close('all')
objective_response_types = ['NOR', 'OR']
color_palette = ['lightgray', 'dimgray']
d = {'Response Group': response_group,
     'Max Clonal AxR': clin_maxclonal,
     'Max AxR': clin_max}
clin_df = pd.DataFrame(d)
# IN SILICO Max neoantigen quality:
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxplot(data=MSIdata, x='objective_response', y='maxNAquality', order=['NOR','OR'],
                 showfliers=False)
# add_stat_annotation(
#     ax, data=MSIdata, x='objective_response', y='maxNAquality', order=['NOR','OR'],
#     box_pairs=[("NOR", "OR")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
# Result: NOR v.s. OR: Welch's t-test independent samples with Bonferroni correction, P_val=1.879e-16 stat=-8.258e+00
outlier_df = pd.DataFrame(columns=MSIdata.columns)
for i in range(len(objective_response_types)):
    minv, maxv = np.percentile(MSIdata.loc[MSIdata['objective_response']==objective_response_types[i]]['maxNAquality'], [25, 75])
    IQR = maxv - minv
    outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['objective_response']==objective_response_types[i]) & ((MSIdata['maxNAquality'] < (minv - 1.5*IQR)) | (MSIdata['maxNAquality'] > (maxv + 1.5*IQR)))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='objective_response', y='maxNAquality', order=['NOR','OR'], size=2, jitter=0.09, color='dimgray')
plt.ylabel('Maximal neoantigen quality')
# ax.set_yscale('log')
ax.tick_params(axis='y', labelsize=12)
plt.ylim([0, 1700])
ax.set_xticklabels(['NOR','OR'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "objresp_maxNAquality.png", format='png')
plt.savefig(cohort_plot_path + "objresp_maxNAquality.svg", format='svg')
#
# IN SILICO Max clonal quality:
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxplot(data=MSIclonal, x='objective_response', y='clonal_neoant_quality', order=['NOR', 'OR'],
                 showfliers=False)
# Subset to tumors with at least one clonal neoantigen. This excludes 2659 tumors (n = 2341).
# add_stat_annotation(
#     ax, data=MSIclonal, x='objective_response', y='clonal_neoant_quality', order=['NOR','OR'],
#     box_pairs=[("NOR", "OR")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
# Result: NOR v.s. OR: Welch's t-test independent samples with Bonferroni correction, P_val=1.223e-06 stat=-4.883e+00
outlier_df = pd.DataFrame(columns=MSIclonal.columns)
for i in range(len(objective_response_types)):
    minv, maxv = np.percentile(MSIclonal.loc[MSIclonal['objective_response']==objective_response_types[i]]['clonal_neoant_quality'], [25, 75])
    IQR = maxv - minv
    outlier_df = pd.concat([outlier_df, MSIclonal.loc[(MSIclonal['objective_response']==objective_response_types[i]) & ((MSIclonal['clonal_neoant_quality'] < (minv - 1.5*IQR)) | (MSIdata['clonal_neoant_quality'] > (maxv + 1.5*IQR)))]], ignore_index=True)
sns.stripplot(data=outlier_df, x='objective_response', y='clonal_neoant_quality', order=['NOR','OR'], size=2, jitter=0.09, color='dimgray')
plt.xlabel('')
plt.ylabel('Max clonal AxR score')
plt.ylim([1, 200])
#ax.set_yscale('log')
ax.set_xticklabels(['NOR', 'OR'])
#ax.set_yticks([0, 100, 200, 300])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "objresp_maxClonalQuality.png", format='png')
plt.savefig(cohort_plot_path + "objresp_maxClonalQuality.svg", format='svg')
# IN CLINICO Max neoantigen quality:
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxplot(data=clin_df, x='Response Group', y='Max AxR', order=['NOR','OR'],
                 showfliers=True)
# Hypothesis testing done in R
plt.ylabel('Maximal neoantigen quality')
#ax.set_yscale('log')
ax.tick_params(axis='y', labelsize=12)
plt.ylim([0, 375])
ax.set_xticklabels(['NOR','OR'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "clin_objresp_maxNAquality.png", format='png')
plt.savefig(cohort_plot_path + "clin_objresp_maxNAquality.svg", format='svg')
#
# IN CLINICO Max clonal quality:
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
ax = sns.boxplot(data=clin_df, x='Response Group', y='Max Clonal AxR', order=['NOR', 'OR'],
                 showfliers=True)
# Hypothesis testing done in R
plt.xlabel('')
plt.ylabel('Max clonal AxR score')
plt.ylim([0, 375])
# ax.set_yscale('log')
ax.set_xticklabels(['NOR', 'OR'])
#ax.set_yticks([0, 100, 200, 300])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "clin_objresp_maxClonalQuality.png", format='png')
plt.savefig(cohort_plot_path + "clin_objresp_maxClonalQuality.svg", format='svg')
#
#
# Final export of source data:
# Source data export:
if export_source_data:
    # Source data for SF 4:
    df = MSIdata[['objective_response',
                  'clonal_neoant_quality']]
    df.to_csv(path_to_source_data + 'SF4_insilicoMMRD.csv', index=False)
    clin_df.to_csv(path_to_source_data + 'SF4_inclinicoMMRD.csv', index=False)

    # Source data for SF 5:
    df = MSIdata[['w8_subsampled_num_pseudoprogression_events',
                  'num_pseudoprogression_events',
                  'response_pseud',
                  'maxNAquality',
                  'axr_weighted_anteginicity',
                  'num_clonal_neoant',
                  'tree_index',
                  'minNAquality']]
    df.to_csv(path_to_source_data + 'SF5_MMRD.csv', index=False)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# TESTING ZONE!
# including misc.
# plots that didn't
# make it into
# the paper.
#
#
#
#
#
#
#
#
#
# # Looking only at MMRD tumors with at least one clonal neoantigen
# # color_palette = ['lemonchiffon','khaki','gold']
# color_palette = ['lightcyan', 'cyan', 'darkcyan']
#
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxplot(data=MSIclonal, x='LTR', y='num_clonal_neoant', order=['No Response', 'Acquired Resistance', 'Durable Response'])
# add_stat_annotation(
#     ax, data=MSIclonal, x='LTR', y='num_clonal_neoant', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# # plt.ylim([0, 1])
# plt.xlabel('')
# plt.ylabel('Number of clonal neoantigens')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.savefig(cohort_plot_path + "MSIclonal_boxplot_numclonalneoant.png")
# plt.savefig(cohort_plot_path + "MSIclonal_boxplot_numclonalneoant.svg", format='svg')
#
# plt.figure()
# ax = sns.boxplot(data=MSIclonal, x='LTR', y='num_subclones', order=['No Response', 'Acquired Resistance', 'Durable Response'])
# add_stat_annotation(
#     ax, data=MSIclonal, x='LTR', y='num_subclones', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# plt.xlabel('')
# plt.ylabel('Total number of subclones')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# #plt.ylim([1, 9.25])
# plt.savefig(cohort_plot_path + "MSIclonal_heterogeneity.png")
# plt.savefig(cohort_plot_path + "MSIclonal_heterogeneity.svg")
#
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxplot(data=MSIclonal, x='LTR', y='clonal_neoant_quality',
#                    order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False)
# #add_stat_annotation(
# #    ax, data=MSIclonal, x='LTR', y='clonal_neoant_quality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
# #    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
# #    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=7)
# plt.xlabel('')
# plt.ylabel('Clonal neoantigen AxR')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.yscale('log')
# plt.ylim([1e0, 1e3])
# plt.savefig(cohort_plot_path + "MSIclonal_topclonal.png")
# plt.savefig(cohort_plot_path + "MSIclonal_topclonal.svg")
#
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxplot(data=MSIclonal, x='LTR', y='clonal_neoant_quality',
#                    order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False)
# add_stat_annotation(
#     ax, data=MSIclonal, x='LTR', y='clonal_neoant_quality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# plt.xlabel('')
# plt.ylabel('Clonal neoantigen AxR')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.ylim([3.1e3, 3.3e3])
# plt.savefig(cohort_plot_path + "MSIclonal_topclonal_labels.png")
# plt.savefig(cohort_plot_path + "MSIclonal_topclonal_labels.svg")
#
# # # # LTR and OR / NOR binary categorization:
# # Figure: tumor burden and response, LTR
# color_palette = ['lightgray', 'dimgray']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxplot(data=MSIclonal, x='objective_response', y='clonal_neoant_quality',
#                    order=['NOR','OR'], showfliers=False)
# add_stat_annotation(
#     ax, data=MSIclonal, x='objective_response', y='clonal_neoant_quality', order=['NOR','OR'],
#     box_pairs=[('NOR','OR')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# plt.xlabel('')
# plt.ylabel('Clonal neoantigen AxR')
# ax.set_xticklabels(['NOR', 'OR'])
# plt.yscale('log')
# plt.ylim([1e0, 6e3])
# plt.savefig(cohort_plot_path + "objresp_topclonal.png")
# plt.savefig(cohort_plot_path + "objresp_topclonal.svg")
#
#
# # Figure: tumor burden and response, LTR
# tumor_types = ['persistent_founder', 'subclonal_dominant', 'clonal']
# responses_types = ['No Response', 'Acquired Resistance', 'Durable Response']
# MSI_heatmap_by_tumortype = np.zeros((len(responses_types), len(tumor_types)))
# for i in range(len(tumor_types)):
#     for j in range(len(responses_types)):
#         MSI_heatmap_by_tumortype[i, j] = sum((MSIdata['tumor_type']==tumor_types[i])*(MSIdata['LTR']==responses_types[j]))
# for j in range(len(responses_types)):
#     MSI_heatmap_by_tumortype[:, j] = MSI_heatmap_by_tumortype[:, j]/sum(MSI_heatmap_by_tumortype[:, j]) * 100
#
# # Figure: tumor burden and response, LTR
# color_palette = ['lightcyan', 'cyan', 'darkcyan']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxenplot(data=MSIdata, x='LTR', y='unique_TMB_10perc', order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='unique_TMB_10perc', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# plt.xlabel('')
# #plt.yticks([0.5e6, 1e6, 1.5e6, 2e6], ['0.5e6', '1e6', '1.5e6', '2e6'])
# plt.ylabel('Neoantigens in $\geq$ 10\% of tumor')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_uniqueTMB10perc.png")
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_uniqueTMB10perc.svg", format='svg')
#
# # Figure: treeindex and response, LTR
# color_palette = ['lightcyan', 'cyan', 'darkcyan']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxenplot(data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# plt.ylim([0.2, 0.8])
# plt.xlabel('')
# plt.ylabel('Tree index')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_treeindex.png")
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_treeindex.svg", format='svg')
#
# # Figure: LTR and heterogeneity
# color_palette = ['lightcyan', 'cyan', 'darkcyan']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxplot(data=MSIdata, x='LTR', y='num_subclones', order=['No Response', 'Acquired Resistance', 'Durable Response'])
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='num_subclones', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# # plt.ylim([0, 1])
# plt.xlabel('')
# plt.ylabel('Number of Subclones')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.savefig(cohort_plot_path + "LTR_MSI_boxplot_numsubclones.png")
# plt.savefig(cohort_plot_path + "LTR_MSI_boxplot_numsubclones.svg", format='svg')
#
# # Figure: LTR and clonal heterogeneity
# color_palette = ['lightcyan', 'cyan', 'darkcyan']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxplot(data=MSIdata, x='LTR', y='num_clonal_neoant', order=['No Response', 'Acquired Resistance', 'Durable Response'])
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='num_clonal_neoant', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# # plt.ylim([0, 1])
# plt.xlabel('')
# plt.ylabel('Number of clonal neoantigens')
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.savefig(cohort_plot_path + "LTR_MSI_boxplot_numclonalneoant.png")
# plt.savefig(cohort_plot_path + "LTR_MSI_boxplot_numclonalneoant.svg", format='svg')
#
# # Figure: tumor weighted antigenicity and response, LTR
# color_palette = ['lightcyan', 'cyan', 'darkcyan']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxenplot(data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#                  showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
# plt.xlabel('')
# plt.ylabel('Weighted mean antigenicity')
# plt.ylim([0, 1950])
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# ax.set_yticks([0, 500, 1000, 1500])
# ax.tick_params(axis='y', labelsize=12)
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_axrmean.png")
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_axrmean.svg", format='svg')
#
# # Figure: tumor maximal antigenicity and response, LTR
# color_palette = ['lightcyan', 'cyan', 'darkcyan']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxenplot(data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#                  showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
#     box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
#     test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
# #plt.yscale('log')
# plt.ylim([0, 3100])
# plt.ylabel('Maximal neoantigen quality')
# ax.tick_params(axis='y', labelsize=12)
# ax.set_xticklabels(['NR', 'AR', 'DR'])
# plt.xlabel('')
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_maxNAquality.png")
# plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_maxNAquality.svg",format='svg')
#
# psp_types = ['No PsP', 'PsP']
# for i in range(len(MSIdata['response_pseud'])):
#     if MSIdata['num_pseudoprogression_events'].loc[i+1]==0:
#         MSIdata['response_pseud'].loc[i+1] = psp_types[0]
#     elif MSIdata['num_pseudoprogression_events'].loc[i+1]>0:
#         MSIdata['response_pseud'].loc[i+1] = psp_types[1]
#     else:
#         pass
# color_palette = ['chartreuse', 'green']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
# ax = sns.boxenplot(data=MSIdata, x='response_pseud', y='maxNAquality', order=['No PsP', 'PsP'],
#                  showfliers=False, k_depth=4)
# outlier_df = pd.DataFrame(columns=MSIdata.columns)
# for i in range(len(psp_types)):
#     minv, maxv = np.percentile(MSIdata.loc[MSIdata['response_pseud']==psp_types[i]]['maxNAquality'], [3.125, 96.875])
#     outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['response_pseud']==psp_types[i]) & ((MSIdata['maxNAquality'] < minv) | (MSIdata['maxNAquality'] > maxv))]], ignore_index=True)
# sns.stripplot(data=outlier_df, x='response_pseud', y='maxNAquality', order=['No PsP', 'PsP'], size=2, jitter=0.04, color='dimgray')
# plt.ylabel('Maximal neoantigen quality')
# ax.set_yscale('log')
# ax.tick_params(axis='y', labelsize=12)
# ax.set_xticklabels(['No PsP', 'PsP'])
# plt.xlabel('')
# plt.savefig(cohort_plot_path + "psp_max_qual.png", format='png')
# plt.savefig(cohort_plot_path + "psp_max_qual.svg", format='svg')
#
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
# ax = sns.boxenplot(data=MSIdata, x='response_pseud', y='axr_weighted_anteginicity', order=['No PsP', 'PsP'],
#                  showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='response_pseud', y='axr_weighted_anteginicity', order=['No PsP', 'PsP'],
#     box_pairs=[('No PsP', 'PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# outlier_df = pd.DataFrame(columns=MSIdata.columns)
# for i in range(len(psp_types)):
#     minv, maxv = np.percentile(MSIdata.loc[MSIdata['response_pseud']==psp_types[i]]['axr_weighted_anteginicity'], [3.125, 96.875])
#     outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['response_pseud']==psp_types[i]) & ((MSIdata['axr_weighted_anteginicity'] < minv) | (MSIdata['axr_weighted_anteginicity'] > maxv))]], ignore_index=True)
# sns.stripplot(data=outlier_df, x='response_pseud', y='axr_weighted_anteginicity', order=['No PsP', 'PsP'], size=2, jitter=0.04, color='dimgray')
# plt.ylabel('Mean neoantigen quality')
# plt.ylim([1e-1, 2e3])
# ax.set_yscale('log')
# ax.tick_params(axis='y', labelsize=12)
# ax.set_xticklabels(['No PsP', 'PsP'])
# plt.xlabel('')
# plt.savefig(cohort_plot_path + "psp_mean_qual.png", format='png')
# plt.savefig(cohort_plot_path + "psp_mean_qual.svg", format='svg')
#
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
# ax = sns.boxplot(data=MSIdata, x='response_pseud', y='num_subclones', order=['NR', 'PsP', 'No PsP'],
#                  showfliers=True)
# # add_stat_annotation(
# #     ax, data=MSIdata, x='response_pseud', y='num_subclones', order=['NR', 'PsP', 'No PsP'],
# #     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
# #     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# plt.ylabel('Number of subclones')
# ax.tick_params(axis='y', labelsize=12)
# ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
# plt.xlabel('')
# plt.savefig(cohort_plot_path + "psp_brokendown_numclones.png", format='png')
# plt.savefig(cohort_plot_path + "psp_brokendown_numclones.svg", format='svg')
#
# sns.set_palette(sns.color_palette(color_palette))
# MSI_clonals = MSIdata.loc[MSIdata['is_clonal_neoant']=='Present']
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
# ax = sns.boxenplot(data=MSI_clonals, x='response_pseud', y='clonal_neoant_quality', order=['NR', 'PsP', 'No PsP'],
#                  showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSI_clonals, x='response_pseud', y='clonal_neoant_quality', order=['NR', 'PsP', 'No PsP'],
#     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# outlier_df = pd.DataFrame(columns=MSI_clonals.columns)
# for i in range(len(psp_types)):
#     minv, maxv = np.percentile(MSI_clonals.loc[MSI_clonals['response_pseud']==psp_types[i]]['clonal_neoant_quality'], [3.125, 96.875])
#     outlier_df = pd.concat([outlier_df, MSI_clonals.loc[(MSI_clonals['response_pseud']==psp_types[i]) & ((MSI_clonals['clonal_neoant_quality'] < minv) | (MSI_clonals['clonal_neoant_quality'] > maxv))]], ignore_index=True)
# sns.stripplot(data=outlier_df, x='response_pseud', y='clonal_neoant_quality', order=['NR', 'PsP', 'No PsP'], size=2, jitter=0.04, color='dimgray')
# plt.ylabel('Clonal neoantigen quality')
# plt.ylim([1e-1, 5e3])
# ax.set_yscale('log')
# ax.tick_params(axis='y', labelsize=12)
# ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
# plt.xlabel('')
# plt.savefig(cohort_plot_path + "psp_brokendown_clonal_quality.png", format='png')
# plt.savefig(cohort_plot_path + "psp_brokendown_clonal_quality.svg", format='svg')
#
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
# ax = sns.boxenplot(data=MSIdata, x='response_pseud', y='tumor_growth_rate', order=['NR', 'PsP', 'No PsP'],
#                  showfliers=False, k_depth=4)
# add_stat_annotation(
#     ax, data=MSIdata, x='response_pseud', y='tumor_growth_rate', order=['NR', 'PsP', 'No PsP'],
#     box_pairs=[('NR', 'PsP'), ('NR', 'No PsP'), ('PsP', 'No PsP')],
#     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# outlier_df = pd.DataFrame(columns=MSIdata.columns)
# for i in range(len(psp_types)):
#     minv, maxv = np.percentile(MSIdata.loc[MSIdata['response_pseud']==psp_types[i]]['tumor_growth_rate'], [3.125, 96.875])
#     outlier_df = pd.concat([outlier_df, MSIdata.loc[(MSIdata['response_pseud']==psp_types[i]) & ((MSIdata['tumor_growth_rate'] < minv) | (MSIdata['tumor_growth_rate'] > maxv))]], ignore_index=True)
# sns.stripplot(data=outlier_df, x='response_pseud', y='tumor_growth_rate', order=['NR', 'PsP', 'No PsP'], size=2, jitter=0.04, color='dimgray')
# plt.ylabel('Tumor growth rate')
# #plt.ylim([0.1, 0.75])
# ax.tick_params(axis='y', labelsize=12)
# ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
# plt.xlabel('')
# plt.savefig(cohort_plot_path + "psp_brokendown_tumor_growth_rate.png", format='png')
# plt.savefig(cohort_plot_path + "psp_brokendown_tumor_growth_rate.svg", format='svg')
#
# # tot = len(MSIdata_minNA_eq0)
# # ct_NR = len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['response_pseud']=='NR'])/tot * 100
# # ct_PsP = len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['response_pseud']=='PsP'])/tot * 100
# # ct_noPsP = len(MSIdata_minNA_eq0.loc[MSIdata_minNA_eq0['response_pseud']=='No PsP'])/tot * 100
# #
# # plt.figure(figsize=(5, 4))
# # plt.pie([ct_NR, ct_PsP, ct_noPsP], labels=['NR', 'PsP', 'No PsP'], autopct='%1.1f%%')
# # plt.title('MMRD Tumors w/ surviving founder')
# # plt.savefig(cohort_plot_path + "psp_brokendown_minNAquality_zeros.png")
# # plt.savefig(cohort_plot_path + "psp_brokendown_minNAquality_zeros.svg")
#
# # vertical_jitter = 0.1*np.ones(len(MSIdata.loc[MSIdata['minNAquality']<1])) + 0.3*np.random.rand(len(MSIdata.loc[MSIdata['minNAquality']<1]))
# # MSIdata['minNAquality'].loc[MSIdata['minNAquality']<1] = vertical_jitter
# # Get weakest subclone s.t. Ivana is happy!
# color_palette = ['palegreen','chartreuse', 'green']
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
# ax = sns.stripplot(data=MSIdata, x='response_pseud', y='minNAquality', order=['NR', 'PsP', 'No PsP'],
#                    size=3, alpha=0.6, jitter=0.4)
# ax.set_yscale('log')
# plt.xlabel('')
# plt.ylabel('Weakest neoantigen quality')
# ax.set_xticklabels(['NR', 'PsP', 'No PsP'])
# plt.savefig(cohort_plot_path + "psp_brokendown_minNAquality.png")
# plt.savefig(cohort_plot_path + "psp_brokendown_minNAquality.svg")
#
# # # LTR
# # color_palette = ['cadetblue', 'cyan', 'darkcyan']
# # sns.set_palette(sns.color_palette(color_palette))
# # plt.figure(figsize=(5, 4))
# # plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.1, top = 0.85)
# # ax = sns.stripplot(data=MSIdata, x='LTR', y='minNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
# #                    size=3, alpha=0.6, jitter=0.4)
# # ax.set_yscale('log')
# # # ax.axhline(y=1, color='lightgray', linestyle='--', linewidth=1.2)
# # plt.xlabel('')
# # plt.ylabel('Weakest subclone')
# # ax.set_xticklabels(['NR', 'AR', 'DR'])
# # plt.savefig(cohort_plot_path + "LTR_MSI_minNAquality.png")
# # plt.savefig(cohort_plot_path + "LTR_MSI_minNAquality.svg")
# Supplementary Figure 4(a) - right
# sns.set_palette(sns.color_palette(color_palette))
# plt.figure(figsize=(5, 4))
# plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
# ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='unique_TMB_10perc', order=['Absent', 'Present'], showfliers=False, k_depth=4)
# # add_stat_annotation(
# #     ax, data=MSI_DRs, x='is_clonal_neoant', y='unique_TMB_10perc', order=['Absent', 'Present'],
# #     box_pairs=[('Absent', 'Present')],
# #     test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# outlier_df = pd.DataFrame(columns=MSI_DRs.columns)
# for i in range(len(clonal_types)):
#     minv, maxv = np.percentile(MSI_DRs.loc[MSI_DRs['is_clonal_neoant']==clonal_types[i]]['unique_TMB_10perc'], [3.125, 96.875])
#     outlier_df = pd.concat([outlier_df, MSI_DRs.loc[(MSI_DRs['is_clonal_neoant']==clonal_types[i]) & ((MSI_DRs['unique_TMB_10perc'] < minv) | (MSI_DRs['unique_TMB_10perc'] > maxv))]], ignore_index=True)
# sns.stripplot(data=outlier_df, x='is_clonal_neoant', y='unique_TMB_10perc', order=['Absent', 'Present'], size=2, jitter=0.04, color='dimgray')
# plt.xlabel('')
# # plt.yticks([1e6, 2e6], ['1e6', '2e6'])
# plt.ylabel('Neoantigens in $\geq$10% of tumor')
# plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_unique_TMB_10perc.png")
# plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_unique_TMB_10perc.svg", format='svg')