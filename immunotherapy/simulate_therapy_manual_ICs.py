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
path_base = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_oct_24/' # Path for cohort plots and therapydata
model_type = 'monoclonal' # model type of T cell responses. Acceptable to run 'polyclonal' as well, but it may be slow.
debug = False # Do you want to manually re-simulate immunotherapy for all 10,000 tumors? This is slow if set to True!


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

# Simulate immunotherapy and collect response statistics
try:
    if debug==True:
        raise Exception('Manual re-simulation for debugging purposes.')
    therapydata = pd.read_pickle(open(path_base + model_type + "therapydata_" + str(maxruns) + "_pandas_df.dump", 'rb'))
    print('Immunotherapy dataset with these parameters already created. Loading...')
except:
    print('Immunotherapy dataset with these parameters not already created. Creating...')
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
    all_TMB_timeseries = []
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
        TMB_timeseries = []
        num_solved = 0
        # Load dilled trees from data_gen script
        for dilled_tree_dict in tree_dicts:
            kk += 1
            if kk > maxruns:
                break
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
            orig_num_subclones = np.copy(num_subclones)
            if orig_num_subclones > 1:
                max_NA_quality = max(AxR[AxR_ind:(AxR_ind + orig_num_subclones)])
            else:
                max_NA_quality = 0

            # Assign m,k based on AxR data
            m, k, AxR_vals, new_AxR_ind = assign_k_and_m(AxR, AxR_ind, newNewMat, num_subclones, b_tumor, mu, a, sigma, m_proportionality_constant, k_proportionality_constant)

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

            # Detect oscillations (beta):
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

            # Check immune-specific response criteria. Relate to PD.
            min_increment = 4*7
            time_to_RECIST_PD = Tmax
            RECIST_pdflag = False
            CR_flag = False
            time_to_CR = -1
            needs_confirmation = False
            time_to_imRECIST_iCPD = Tmax
            imRECIST_response_tmax = 'na'
            TMB_list = [TMB_over_time[0]]
            t_list = [t[0]]

            t_i = min_increment
            while t_i < len(t):
                TMB_list.append(TMB_over_time[t_i])
                t_list.append(t[t_i])
                response_to_therapy = getResponse(TMB_list, t_list)
                if response_to_therapy == 'CR' and CR_flag == False:
                    time_to_CR = t[t_i]
                    CR_flag = True
                if response_to_therapy == 'PD':
                    if needs_confirmation == False:
                        if RECIST_pdflag == False:
                            time_to_RECIST_PD = t[t_i]
                            RECIST_pdflag = True
                        needs_confirmation = True
                        t_i = int(t_i + 4 * 7)  # bump up 4 weeks to allow for pseudoprogression, etc.
                        if t_i > len(t):
                            imRECIST_response_tmax = 'iUPD'
                    else:
                        time_to_imRECIST_iCPD = t[t_i]
                        imRECIST_response_tmax = 'iCPD'
                        break
                else:
                    t_i += min_increment
            if imRECIST_response_tmax == 'na':
                imRECIST_response_tmax = response_to_therapy

            # Check immune-specific response criteria. Relate to PD.
            time_to_RECIST_PD = Tmax
            RECIST_pdflag = False
            time_to_iRECIST_iCPD = Tmax
            iRECIST_response_tmax = 'na'
            TMB_list = [TMB_over_time[0]]
            t_list = [t[0]]

            t_i = min_increment
            while t_i < len(t):
                TMB_list.append(TMB_over_time[t_i])
                t_list.append(t[t_i])
                response_to_therapy = getResponse(TMB_list, t_list)
                if response_to_therapy == 'PD':
                    if RECIST_pdflag == False:
                        time_to_RECIST_PD = t[t_i]
                        RECIST_pdflag = True
                    if t[t_i] + 4 * 7 < Tmax:
                        start_t_i = int(t_i + 4 * 7)  # bump up 4 weeks to allow for pseudoprogression, etc.
                        stop_t_i = min(Tmax, t_i + 12 * 7)  # between 4-8 weeks
                        for t_ii in np.arange(start_t_i, stop_t_i + 1, min_increment):
                            TMB_list.append(TMB_over_time[t_ii])
                            t_list.append(t[t_ii])
                            if getResponse(TMB_list, t_list) == 'PD':  # check each time point to confirm iCPD
                                time_to_iRECIST_iCPD = t[t_ii]
                                break
                        if t[t_ii] < stop_t_i:
                            iRECIST_response_tmax = 'iCPD'
                            break
                        else:
                            t_i = int(stop_t_i + min_increment)
                    else:
                        iRECIST_response_tmax = 'iUPD'
                else:
                    t_i += min_increment
            if iRECIST_response_tmax == 'na':
                iRECIST_response_tmax = response_to_therapy

            # Check immune-specific response criteria. Relate to PD.
            time_to_RECIST_PD = Tmax
            RECIST_pdflag = False
            time_to_irRECIST_iCPD = Tmax
            irRECIST_response_tmax = 'na'
            TMB_list = [TMB_over_time[0]]
            t_list = [t[0]]

            t_i = min_increment
            while t_i < len(t):
                TMB_list.append(TMB_over_time[t_i])
                t_list.append(t[t_i])
                response_to_therapy = getResponse(TMB_list, t_list)
                if response_to_therapy == 'PD':
                    if RECIST_pdflag == False:
                        time_to_RECIST_PD = t[t_i]
                        RECIST_pdflag = True
                    if t[t_i] + 4 * 7 < Tmax:
                        start_t_i = int(t_i + 4 * 7)  # bump up 4 weeks to allow for pseudoprogression, etc.
                        stop_t_i = min(Tmax, t_i + 16 * 7)  # between 4-8 weeks
                        for t_ii in np.arange(start_t_i, stop_t_i + 1, min_increment):
                            TMB_list.append(TMB_over_time[t_ii])
                            t_list.append(t[t_ii])
                            if getResponse(TMB_list, t_list) == 'PD':  # check each time point to confirm iCPD
                                time_to_irRECIST_iCPD = t[t_ii]
                                break
                        if t[t_ii] < stop_t_i:
                            irRECIST_response_tmax = 'iCPD'
                            break
                        else:
                            t_i = int(stop_t_i + min_increment)
                    else:
                        irRECIST_response_tmax = 'iUPD'
                else:
                    t_i += min_increment
            if irRECIST_response_tmax == 'na':
                irRECIST_response_tmax = response_to_therapy

            # Get long-term response
            LTR = getLongTermResponse(best_response, response_tmax)
            response_label_dict = {'CR': 'Complete Response', 'PR': 'Partial Response', 'SD': 'Stable Disease',
                                   'PD': 'Progressive Disease'}

            # Check for pseudoprogression (beta):
            pseud, minSLD_percent = getPseudoprogression(TMB_over_time, t)

            ### Plot timeseries of tumor and effector cell responses over time:
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
                        cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
                    plt.plot(t/30.4368, subclone_sol.T / TMB_over_time[0])
                    if len(subclone_sol) > 1:
                        plt.plot(t/30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)
                    plt.legend(
                        ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol) - 1)] + [
                                'Total TB'])
                else:
                    plt.rc('axes', prop_cycle=(cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
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
            ax2.set_ylim([0, 100])
            ax2.tick_params(axis='y', labelcolor=color)
            plt.title(LTR)
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
            plt.title(LTR)
            fig.tight_layout()
            plt.savefig(path + '/tumor_plots/' + model_type + '_largeTmax_effector_fig_' + str(kk) + '.png', format='png')
            plt.savefig(path + '/tumor_plots/' + model_type + '_largeTmax_effector_fig_' + str(kk) + '.svg', format='svg')

            plt.figure(figsize=(5, 4))
            plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
            if num_subclones>0:
                if AxR_vals[0]==0:
                    plt.rc('axes', prop_cycle=(
                        cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
                    plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0])
                    if len(subclone_sol) > 1:
                        plt.plot(t[:t_pseud_index], TMB_over_time[:t_pseud_index] / TMB_over_time[0], 'k--', linewidth=2)
                        plt.legend(
                            ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol) - 1)] + [
                                'Total TB'])
                else:
                    plt.rc('axes', prop_cycle=(cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
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
                          # 'clonal_neoant_quality': [clonal_neoant_quality],
                          'num_subclones': [num_subclones],
                          'ending_num_subclones': [ending_num_subclones],
                          'change_in_num_subclones': [change_in_num_subclones],
                          'starting_TMB': [starting_TMB],
                          'ending_TMB': [ending_TMB],
                          'dTMB': [dTMB],
                          'pseudoprogression': [pseud],
                          'minSLD_percent': [minSLD_percent],
                          'maxNAquality': [max_NA_quality],
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
                          'diseaseControl': [diseaseControl],
                          'tumor_type': [tumor_type]}
            cur_data = pd.DataFrame(tumor_dict)
            therapydata = pd.concat([therapydata, cur_data])
            TMB_timeseries.append(TMB_over_time)
            num_solved += 1
            print('Simulated therapy on ' + ms_stat + ' tumor # ' + str(kk) + ' out of ' + str(maxruns) + ' tumors; best response: ' + best_response)
            all_TMB_timeseries.append(TMB_timeseries)
            all_num_solved.append(num_solved)
        therapydata.index=np.arange(1, len(therapydata)+1)
        dill.dump(therapydata, open(path_base + model_type + "therapydata_" + str(maxruns) + "_pandas_df.dump", 'wb'))
        print('Immunotherapy dataset with these parameters created. Saving to file...')
        print('Number of runs skipped due to nontrivial IC setup: ' + str(ICskip))

MSIdata = therapydata[therapydata['ms_stat']=='MSI']
MSSdata = therapydata[therapydata['ms_stat']=='MSS']

print('DCR = ' + str(sum(MSIdata['diseaseControl'])/len(MSIdata['diseaseControl'])))
print('ORR = ' + str((sum(MSIdata['best_response']=='PR')+sum(MSIdata['best_response']=='CR'))/len(MSIdata['best_response'])))

ttpData = [i for i in MSIdata['time_to_progression'] if i>0]
print('Median PFS = ' + str(np.median(ttpData)/30.4368))
print('36-month PFS = ' + str(sum([i > 36*30.4368 for i in ttpData])/len(ttpData) * 100))

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

print('DR mean num mutations: ' + str(np.mean(MSI_DRs['totalTMB'])))
print('AR mean num mutations: ' + str(np.mean(MSI_ARs['totalTMB'])))
print('NR mean num mutations: ' + str(np.mean(MSI_NRs['totalTMB'])))

print('DR 95% CI: ' + str([i for i in stats.norm.interval(0.95, loc=np.mean(MSI_DRs['totalTMB']), scale=np.std(MSI_DRs['totalTMB'])/np.sqrt(len(MSI_DRs['totalTMB'])))]))
print('AR 95% CI: ' + str([i for i in stats.norm.interval(0.95, loc=np.mean(MSI_ARs['totalTMB']), scale=np.std(MSI_ARs['totalTMB'])/np.sqrt(len(MSI_ARs['totalTMB'])))]))
print('NR 95% CI: ' + str([i for i in stats.norm.interval(0.95, loc=np.mean(MSI_NRs['totalTMB']), scale=np.std(MSI_NRs['totalTMB'])/np.sqrt(len(MSI_NRs['totalTMB'])))]))

slow_responders = MSIdata[(MSIdata['response_12w']=='PR') * (MSIdata['response_tmax']=='CR')]
acquire_resisters = MSIdata[(MSIdata['response_12w']=='PR') * (MSIdata['response_tmax']=='PD')]
stably_PR = MSIdata[(MSIdata['response_12w']=='PR') * (MSIdata['response_tmax']=='PR')]

print('Slow CR number of subclones: ' + str(np.mean(slow_responders['num_subclones'])) + ' +/- ' + str(np.std(slow_responders['num_subclones'])))
print('Acquired resistance number of subclones: ' + str(np.mean(acquire_resisters['num_subclones'])) + ' +/- ' + str(np.std(acquire_resisters['num_subclones'])))
print('Stably PR number of subclones: ' + str(np.mean(stably_PR['num_subclones'])) + ' +/- ' + str(np.std(stably_PR['num_subclones'])))

cohort_plot_path = path_base + 'cohort_plots/'
os.makedirs(cohort_plot_path, exist_ok=True) # create directory for plots of the entire simulated cohort



MSI_LTRvsBest = np.zeros((3, 4))
MSI_LTRvsInitial = np.zeros((3, 4))
recist_response_types = ['PD', 'SD', 'PR', 'CR']
LTR_response_types = ['No Response', 'Acquired Resistance', 'Durable Response']

for i in range(len(LTR_response_types)):
    for j in range(len(recist_response_types)):
        MSI_LTRvsBest[i, j] = sum((MSIdata['best_response']==recist_response_types[j])*(MSIdata['LTR']==LTR_response_types[i]))
        MSI_LTRvsInitial[i, j] = sum((MSIdata['response_12w']==recist_response_types[j])*(MSIdata['LTR']==LTR_response_types[i]))

for j in range(len(recist_response_types)):
    MSI_LTRvsBest[:, j] = MSI_LTRvsBest[:, j]/sum(MSI_LTRvsBest[:, j]) * 100
    MSI_LTRvsInitial[:, j] = MSI_LTRvsInitial[:, j] / sum(MSI_LTRvsInitial[:, j]) * 100






orig_PR_labels = ['SlowCR']*len(slow_responders) + ['AcquiredResistance']*len(acquire_resisters) + ['StablyPR']*len(stably_PR)
orig_PR_df = pd.concat([slow_responders, acquire_resisters, stably_PR], ignore_index=True)
orig_PR_df['response_type_category'] = orig_PR_labels

plt.rcParams['font.family'] = ['Arial']
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
sns.heatmap(data=MSI_LTRvsInitial, xticklabels=['PD', 'SD', 'PR', 'CR'], yticklabels=['NR', 'AR', 'DR'], linewidth=0.5,
            cmap=sns.cubehelix_palette(as_cmap=True, reverse=False))
plt.xlabel('Classification at 12 weeks')
plt.ylabel('Long term response')
plt.savefig(cohort_plot_path + "MSI_durability_response_heatmap.png")
plt.savefig(cohort_plot_path + "MSI_durability_response_heatmap.svg")

color_palette = ['thistle', 'orchid', 'purple']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
plt.title('MMR-D LTR vs. 12-week response')
sns.heatmap(data=MSI_LTRvsInitial, xticklabels=['PD', 'SD', 'PR', 'CR'], yticklabels=['NR', 'AR', 'DR'], linewidth=0.5, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('12-week Response, % originally labeled as')
plt.ylabel('Long Term Response')
plt.savefig(cohort_plot_path + "MSI_durability_response_12w_heatmap.png")
plt.savefig(cohort_plot_path + "MSI_durability_response_12w_heatmap.svg")

plt.figure()
plt.title('Heterogeneity in MMR-D original PR responders')
ax = sns.boxplot(data=orig_PR_df, x='response_type_category', y='num_subclones', order=['AcquiredResistance', 'StablyPR', 'SlowCR'])
add_stat_annotation(
    ax, data=orig_PR_df, x='response_type_category', y='num_subclones', order=['AcquiredResistance', 'StablyPR', 'SlowCR'],
    box_pairs=[("AcquiredResistance", "StablyPR"), ("AcquiredResistance", "SlowCR"), ("StablyPR", "SlowCR")],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Number of subclones')
plt.ylim([1, 11])
plt.yticks(np.arange(1,9,step=1))
plt.savefig(cohort_plot_path + "origPRresp_heterogeneity.png")
plt.savefig(cohort_plot_path + "origPRresp_heterogeneity.svg")


# Plot figure: countplot showing outcomes
plt.figure()
plt.title('Best immunotherapy response by tumor type')
sns.countplot(data=therapydata, x='best_response', hue='ms_stat', order=['PD','SD','PR','CR'])
plt.savefig(cohort_plot_path + "countplot_therapy.png")
plt.savefig(cohort_plot_path + "countplot_therapy.pdf", format='pdf')


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

# Figure: tumor burden and response, LTR
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.25, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='totalTMB', order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='totalTMB', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.yticks([0.5e6, 1e6, 1.5e6, 2e6], ['0.5e6', '1e6', '1.5e6', '2e6'])
plt.ylabel('Total mutations per tumor')
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_totalTMB.png")
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_totalTMB.svg", format='svg')

# Figure: treeindex and response, LTR
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'], showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.ylim([0.2, 0.8])
plt.xlabel('')
plt.ylabel('Tree index')
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_treeindex.png")
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_treeindex.svg", format='svg')

# Figure: LTR and heterogeneity
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxplot(data=MSIdata, x='LTR', y='num_subclones', order=['No Response', 'Acquired Resistance', 'Durable Response'])
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='num_subclones', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
# plt.ylim([0, 1])
plt.xlabel('')
plt.ylabel('Number of Subclones')
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(cohort_plot_path + "LTR_MSI_boxplot_numsubclones.png")
plt.savefig(cohort_plot_path + "LTR_MSI_boxplot_numsubclones.svg", format='svg')

# Figure: tumor weighted antigenicity and response, LTR
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
                 showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Weighted mean antigenicity')
plt.ylim([0, 1950])
ax.set_xticklabels(['NR', 'AR', 'DR'])
ax.set_yticks([0, 500, 1000, 1500])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_axrmean.png")
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_axrmean.svg", format='svg')

color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
                 showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
#plt.yscale('log')
plt.ylim([0, 3100])
plt.ylabel('Maximal neoantigen quality')
ax.tick_params(axis='y', labelsize=12)
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.xlabel('')
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_maxNAquality.png")
plt.savefig(cohort_plot_path + "LTR_MSI_boxenplot_maxNAquality.svg",format='svg')

# set up clonalityy fractions because countplot sucks
MSIdata['is_clonal_neoant'] = MSIdata['is_clonal_neoant'].replace([0, 1], ['Absent', 'Present'])
absentdata = MSIdata[MSIdata['is_clonal_neoant']=='Absent']
presentdata = MSIdata[MSIdata['is_clonal_neoant']=='Present']

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
fig, ax = plt.subplots(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.90, wspace=0.05)
bottom = np.zeros(2)
for boolean, cur_ltr in clonal_dict.items():
    p = ax.bar(tumor_type, cur_ltr, width=0.5, label=boolean, bottom=bottom, edgecolor='k')
    bottom += cur_ltr
ax.legend()
plt.ylim([0, 100])
plt.ylabel('Percent')
plt.xlabel('Clonal neoantigen')
plt.savefig(cohort_plot_path + "LTR_countplot_clonality_improved_MSI.png")
plt.savefig(cohort_plot_path + "LTR_countplot_clonality_improved_MSI.svg", format='svg')


# Figure 3: Clinical cohort vs. simulated cohort comparisons
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
    MMRD_sim_errbar_magnitude.append(ci_sim.high - ci_sim.low)  # scale by error bar size in clinical data

MMRD_clinical_yerr = [abs(MMRD_clinical_frac_vec - MMRD_clinical_yerrmin), abs(MMRD_clinical_frac_vec - MMRD_clinical_yerrmax)]
MMRD_sim_yerr = [abs(MMRD_sim_perc_vec - MMRD_sim_yerrormin), abs(MMRD_sim_perc_vec - MMRD_sim_yerrmax)]

clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MMRD_clinical_frac_vec[0], MMRD_clinical_frac_vec[1], MMRD_clinical_frac_vec[2], MMRD_clinical_frac_vec[3],
                   MMRD_sim_perc_vec[0], MMRD_sim_perc_vec[1], MMRD_sim_perc_vec[2], MMRD_sim_perc_vec[3]]}
clincomp_df = pd.DataFrame(data = clincomp)

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


# MMR-P: clinical vs sim data at 12 weeks
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
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MMRP_clinical_frac_vec[0], MMRP_clinical_frac_vec[1], MMRP_clinical_frac_vec[2], MMRP_clinical_frac_vec[3],
                   MMRP_sim_perc_vec[0], MMRP_sim_perc_vec[1], MMRP_sim_perc_vec[2], MMRP_sim_perc_vec[3]]}
clincomp_df = pd.DataFrame(data = clincomp)
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='best_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_loc = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_loc[0:4], y=y_loc[0:4], yerr=MMRP_clinical_yerr, fmt='.', c='k')
ax.errorbar(x=x_loc[4:], y=y_loc[4:], yerr=MMRP_sim_yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.title('MMR-P response at 12 weeks')
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSS.png")
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSS.svg", format='svg')


# MMR-D: clinical vs sim data at 12 weeks
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
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MMRD_12w_clinical_perc_vec[0], MMRD_12w_clinical_perc_vec[1], MMRD_12w_clinical_perc_vec[2], MMRD_12w_clinical_perc_vec[3],
                   MMRD_12w_sim_perc_vec[0], MMRD_12w_sim_perc_vec[1], MMRD_12w_sim_perc_vec[2], MMRD_12w_sim_perc_vec[3]]}
clincomp_df = pd.DataFrame(data = clincomp)
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='best_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_loc = [p.get_height() for p in ax.patches]
ax.errorbar(x=x_loc[0:4], y=y_loc[0:4], yerr=MMRD_12w_clinical_yerr, fmt='.', c='k')
ax.errorbar(x=x_loc[4:], y=y_loc[4:], yerr=MMRD_12w_sim_yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
plt.title('MMR-D response at 12 weeks')
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSI.png")
plt.savefig(cohort_plot_path + "clincomp_barplot_12wk_MSI.svg", format='svg')


# Time-to-progression data
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

# how to get 95% CI of a median?
sorted_ttp = np.sort(ttpData)
n = len(ttpData)
q = 0.5 # quantile of interest: median
z = 1.96 # corresponding to 95% confidence interval
j = int(np.ceil(n*q - z*np.sqrt(n*q*(1-q))))
k = int(np.ceil(n*q + z*np.sqrt(n*q*(1-q))))
print('95% CI median PFS: ('+ str(sorted_ttp[j]) + ', ' + str(sorted_ttp[k]) + ')')
sim_med_yerrmin = abs(sim_med_pfs - sorted_ttp[j])
sim_med_yerrmax = abs(sim_med_pfs - sorted_ttp[k])
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
ax[0].set_xlabel('')
ax[0].set_ylabel('Percent')
ax[0].legend(loc = 'upper right', fontsize=12)
ax[0].set_ylim([0, 70])
ax[0].set_title('MMR-D PFS')

ax_1 = sns.barplot(data=clincomp_df[:2], x='PFS_type', y='PFS', hue='Source', ax=ax[1])
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax_1.patches]
y_loc = [p.get_height() for p in ax_1.patches]
ax[1].errorbar(x=x_loc[0], y=y_loc[0], yerr=clin_med_yerr, fmt='.', c='k')
ax[1].errorbar(x=x_loc[1], y=y_loc[1], yerr=sim_med_yerr, fmt='.', c='k')
ax[1].set_xlabel('')
ax[1].set_ylabel('Months')
ax[1].set_ylim([0, 70])
ax[1].set_title('MMR-D PFS')
ax[1].legend(loc = 'upper right', fontsize=12)

plt.savefig(cohort_plot_path + "clincomp_PFS.png")
plt.savefig(cohort_plot_path + "clincomp_PFS.svg", format='svg')


# Looking only at MMR-D tumors with a durable response to therapy (Supplementary Figure 3)
MSIdata['is_clonal_neoant'] = MSIdata['is_clonal_neoant'].replace([0, 1], ['Absent', 'Present'])
nonclonal_DRs = MSIdata.loc[(MSIdata['LTR']=='Durable Response') & (MSIdata['is_clonal_neoant']=='Absent')]
nonclonal_ARs = MSIdata.loc[(MSIdata['LTR']=='Acquired Resistance') & (MSIdata['is_clonal_neoant']=='Absent')]
nonclonal_NRs = MSIdata.loc[(MSIdata['LTR']=='No Response') & (MSIdata['is_clonal_neoant']=='Absent')]
MSInoclonal = MSIdata.loc[MSIdata['is_clonal_neoant']=='Absent']
MSI_DRs = MSIdata.loc[MSIdata['LTR']=='Durable Response']

color_palette = ['aliceblue','dodgerblue'] #['lightcyan', 'darkturquoise']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='maxNAquality', order=['Absent', 'Present'],
                 showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='maxNAquality', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
#plt.yscale('log')
plt.ylim([0, 2900])
plt.ylabel('Maximal neoantigen quality')
ax.tick_params(axis='y', labelsize=12)
plt.xlabel('')
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_maxNAquality.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_maxNAquality.svg", format='svg')

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='axr_weighted_anteginicity', order=['Absent', 'Present'],
                 showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='axr_weighted_anteginicity', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Weighted mean antigenicity')
plt.ylim([0, 1680])
ax.tick_params(axis='y', labelsize=12)
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_axrmean.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_axrmean.svg", format='svg')

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='totalTMB', order=['Absent', 'Present'], showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='totalTMB', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.yticks([1e6, 2e6], ['1e6', '2e6'])
plt.ylabel('Total mutations per MMR-D tumor')
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_totalTMB.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_totalTMB.svg", format='svg')

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxenplot(data=MSI_DRs, x='is_clonal_neoant', y='tree_index', order=['Absent', 'Present'], showfliers=False, k_depth=4)
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='tree_index', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
plt.ylim([0.2, 0.8])
plt.xlabel('')
plt.ylabel('Tree index')
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_treeindex.png")
plt.savefig(cohort_plot_path + "DR_LTR_MSI_boxenplot_treeindex.svg", format='svg')


plt.figure()
ax = sns.boxplot(data=MSI_DRs, x='is_clonal_neoant', y='num_subclones', order=['Absent', 'Present'])
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='num_subclones', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Number of subclones')
plt.ylim([1, 9.25])
plt.savefig(cohort_plot_path + "DR_MSI_heterogeneity.png")
plt.savefig(cohort_plot_path + "DR_MSI_heterogeneity.svg")

