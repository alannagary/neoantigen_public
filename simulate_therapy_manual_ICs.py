import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from analyze_raw_tumors import *
import pandas as pd
import dill
import seaborn as sns
from scipy import stats
from statannot import add_stat_annotation


def get_subclones(path, tree, treenum, min_size=10000, save_df=False):
    try:
        # raise Exception('debugging') # comment out when done debugging
        mylist = dill.load(open(path + '/tumor_mats/' + f"/df_{ms_stat}_{treenum}.dump", 'wb'))
        newNewMat = mylist[0]
        num_subclones = mylist[1]
        is_there_a_clonal_neoant = mylist[2]
    except:
        neoant_info_list = neoant_node_info_search(tree, 0)
        unique_pops, adj_pops, ords = get_abundance_alt(neoant_info_list)
        num_subclones = len([i for i in adj_pops if i>=min_size])
        new_neoant_info_list = [i for i in neoant_info_list if i[2]>=min_size]
        mat = get_abundance_mat(tree, new_neoant_info_list, num_subclones)
        # Sort the populations for input into newMat:
        sort_inds = np.argsort(-adj_pops) # use negative so that we ensure sorted pops are arranged in descending order
        adj_pops_sorted = adj_pops[sort_inds]
        unique_pops_sorted = unique_pops[sort_inds] #use the same sort indices as for adj_pops so that we retain the right information tied to the right subclone
        # Sort the matrix of tree structure based on population size, by reference to new_neoant_info_list
        new_neoant_info_list.sort(
            key=lambda x: x[0])  # sort new_neoant_info_list in the same way that mat will be sorted
        new_neoant_info_list_popsizes = np.array([i[2] for i in new_neoant_info_list])
        mat_sort_inds = np.argsort(-new_neoant_info_list_popsizes)
        new_neoant_info_list_ords = np.array([i[0] for i in np.array(new_neoant_info_list)[mat_sort_inds]]) #prep for sorting by ord after sorting by pop
        ord_mat_sort_inds = np.argsort(new_neoant_info_list_ords)
        mat = mat[mat_sort_inds]
        # Use the sorted values to get our population mat
        newMat = np.zeros((np.shape(mat)[0], np.shape(mat)[0] + 2))
        newMat[:, 0] = adj_pops_sorted[:num_subclones]  # TOTAL populations with a given mutation
        newMat[:, 1] = unique_pops_sorted[:num_subclones]  # UNIQUE populations
        newMat[:, 2:] = mat  # sort rows of sMat by descending order of population
        newMat = newMat[ord_mat_sort_inds]  # sort by DESCENDING ORDER based on TOTAL population

        newNewMat = np.zeros((np.shape(newMat)[0]+1, np.shape(newMat)[1]))
        newNewMat[0, 0] = tree.get_total_size()
        newNewMat[0, 1] = tree.size
        newNewMat[1:, :] = newMat
        if len(newNewMat) == 1:
            is_there_a_clonal_neoant = False
        else:
            is_there_a_clonal_neoant = max([newNewMat[i, 0]==1e5 for i in range(1, len(newNewMat))]) #check to see if any total subclone sizes == 100,000 (i.e. present in every cell)
        if save_df is True:
            dill.dump([newNewMat, num_subclones, is_there_a_clonal_neoant],
                open(path + '/tumor_mats' + f"/df_{ms_stat}_{treenum}.dump", 'wb'))

    return newNewMat, num_subclones, is_there_a_clonal_neoant


def assign_k_and_m(AxR_data, AxR_ind, mat, num_subclones, b_tumor, mu, a, sigma, mprop, kprop):
    AxR = AxR_data[AxR_ind:(AxR_ind + num_subclones)]
    m_0 = (b_tumor / mu) / 100000  # baseline clonal m, no mut
    k_0 = (a / sigma) / 1000  # baseline clonal k, no mut
    m = [m_0]  # else, we need to assign a (m,k) pair to the parental, unmutated population.
    k = [k_0]
    AxR_vals = [0] # no AxR for no-neoantigenic population
    for j in range(1, num_subclones + 1):  # need to add one for the parental population
        potential_AxRs = mat[j][2:] * AxR[:num_subclones]  # find all possible AxR for this particular subclone.
        clone_AxR = max(potential_AxRs)
        m.append(m_0 + mprop * clone_AxR)
        k.append(k_0 + kprop * clone_AxR)
        AxR_vals.append(clone_AxR)
    new_ind = AxR_ind + num_subclones
    if new_ind + 8 >= len(AxR_data): # take into account that there may be >= 8 subclones
        new_ind = new_ind + 8 - len(AxR_data)
    return m, k, AxR_vals, new_ind

# NB: is_there_a_clonal_neoant is 1 if there is no surviving root population, SO we can rename this as such in this function
def assign_ICs(mat):
    ICs = [mat[i,0] for i in range(len(mat))]
    m_trunc = 0 # currently, do not truncate any m values
    if len(mat)==1:
        return ICs, m_trunc
    else:
        for i in range(1, len(mat)):
            child_inds = [j for j in range(i+1, len(mat)) if mat[j, i+1]==1 and sum(mat[j, i+2:])==1] # record indices of children of this clone. START: row after this one. AND: need to make sure that we don't have more than 1 additional mutation.
            if len(child_inds)==0: # no children? Keep ICs as is
                continue
            else:
                child_tot_size = sum([mat[j, 0] for j in child_inds]) # sum together the total sizes of child nodes
                ICs[i] = ICs[i] - child_tot_size
        # Now, consider the clonal population (either parental, or a clonal neoantigen)
        ICs[0] = 1e5 - sum(ICs[1:])  # then the founder cells will all go into this
        # What if there is neither surviving parental population NOR a clonal neoantigen?
        if ICs[0]==0: # if all cells accounted for in the children, then there's nothing left in the original pop
            ICs = ICs[1:] # so cut it out of consideration.
            m_trunc = 1 # we do not need m_0, so remove it from the list of potential m values
        return ICs, m_trunc


def estimate_lesion_diameter(lesion_pop):
    # In units of cells, assuming spherical geometry of a lesion (assumption...)
    # V = (4pi/3)r^3 cells
    # r^3 = 3V/4pi
    # r = cbrt(3V/4pi)
    # d = 2r
    return 2 * np.cbrt((3*lesion_pop)/(4*np.pi))


def estimate_tumor_sum_diameters(subclone_pops):
    subclone_diameter_sum = 0
    for subclone in subclone_pops:
        subclone_diameter_sum += estimate_lesion_diameter(subclone)
    return subclone_diameter_sum


def getTreeIndex(root):
    normalization_sum, balance_sum = getSubtreeIndex(root, normalization_sum=0, balance_sum=0)
    return balance_sum / normalization_sum

def getSubtreeIndex(node, normalization_sum, balance_sum):
    if node.children is None:
        normalization_sum = 0
        balance_sum = 0
    else:
        S_star_node = node.get_total_size() - node.size
        W_i = 0
        d = len(node.children)
        if d >= 2:
            for j in node.children:
                p_ij = j.get_total_size()/S_star_node
                if p_ij > 0:
                    W_i += (-p_ij) * np.emath.logn(d, p_ij)
        normalization_sum += S_star_node
        balance_sum += S_star_node * W_i
        if W_i > 1:
            print("W_i: " + str(W_i))
            print("W_i > 1, which is never supposed to happen. Double-check how you compute p_ij and W_i.")
        # now do the same for the children:
        for j in node.children:  # Recurse over children.
            normalization_sum, balance_sum = getSubtreeIndex(j, normalization_sum, balance_sum)
    return normalization_sum, balance_sum

def getResponse(tumor_size_over_time, t, tmax_index):
    sum_lesion_diams = [estimate_lesion_diameter(tumor_size_over_time[iii]) for iii in range(len(t[:tmax_index]))]
    start_sum_lesion_diameters = sum_lesion_diams[0]
    end_sum_lesion_diameters = sum_lesion_diams[-1]
    baseline_comp_SLD = end_sum_lesion_diameters / start_sum_lesion_diameters * 100  # percent change in lesion diameters compared to baseline (pre-treatment)
    # Classify response to therapy based on SLDs
    if baseline_comp_SLD < 0.005:  # If the SLD at end of treatment is < 0.005% that of beginning, we consider this CR. That's less than 5 cells.
        response = 'CR'  # code as complete response
    elif baseline_comp_SLD <= 70:  # at least a 30% decrease => at most 70% of tumor remains, compared to BASELINE
        response = 'PR'  # code as partial response
    else:
        min_sum_lesion_diameters = min(sum_lesion_diams)
        min_comp_SLD = end_sum_lesion_diameters / min_sum_lesion_diameters * 100  # percent change in lesion diams compared to min in study
        if min_comp_SLD >= 120:  # more than 20% increase, compared to MINIMUM SLD in study.
            response = 'PD'  # code as progressive disease (this is where "survival" ends)
        else:
            response = 'SD'  # code as stable disease
    return response

def detectOscillations(tumor_size_over_time, t):
    approx_deriv = []
    dt = t[1] - t[0]
    oscillation_counter = 0
    for k in range(1, len(tumor_size_over_time)):
        approx_deriv.append((tumor_size_over_time[k]- tumor_size_over_time[k-1])/dt)
        if k>1:
            if np.sign(approx_deriv[k-1]) != np.sign(approx_deriv[k-2]):
                oscillation_counter += 1
    if (oscillation_counter >= 3) and (0.01<tumor_size_over_time[-1]/tumor_size_over_time[0]<2):
        return True # if first deriv changes sign more than 2x, we consider this an oscillating response; also require equilibration between 1% and 200% of original size
    else:
        return False


def getLongTermResponse(best_response, response_tmax):
    if (response_tmax=='SD' or response_tmax=='PR' or response_tmax=='CR'):
        LTR = 'Durable Response'
    elif (best_response=='SD' or best_response=='PR' or best_response=='CR') and (response_tmax=='PD'):
        LTR = 'Acquired Resistance'
    elif (best_response=='PD'):
        LTR = 'No Response'
    return LTR


def getTMB(tree, depth, TMB):  # recursion is the best, dawg
    TMB += depth * tree.size
    if len(tree.children) > 0:
        for child in tree.children:
            TMB = getTMB(child, depth+1, TMB)
    return TMB


def getTMB_threshold(tree, depth, TMB, threshold):  # recursion is the best, dawg
    TMB += depth * tree.size * (tree.get_total_size() >= threshold)
    if len(tree.children) > 0:
        for child in tree.children:
            TMB = getTMB_threshold(child, depth+1, TMB, threshold)
    return TMB


def getPseudoprogression(tumor_size_over_time, t):
    epsilon = 1e-6
    sum_lesion_diams = [estimate_lesion_diameter(tumor_size_over_time[iii]) for iii in range(len(t))]
    deriv_at_tmax = (3*sum_lesion_diams[-1] - 4*sum_lesion_diams[-2] + sum_lesion_diams[-3])/(2*(t[1]-t[0])) # second-order dofference formula
    start_sum_lesion_diameters = sum_lesion_diams[0]
    end_sum_lesion_diameters = sum_lesion_diams[-1]
    baseline_comp_SLD = end_sum_lesion_diameters / start_sum_lesion_diameters * 100
    if min(sum_lesion_diams)==0:
        return 'tumor eliminated', 0
    else:
        min_sum_lesion_diameters = min(sum_lesion_diams)
        min_comp_SLD = end_sum_lesion_diameters / min_sum_lesion_diameters * 100  # percent change in lesion diams compared to min in study
        if min_comp_SLD >= 120:  # more than 20% increase, compared to MINIMUM SLD in study.
            # apparent progression
            if baseline_comp_SLD <= 120:
                response = 'pseudoprogression'
            else:
                response = 'true progression'
        else:
            response = 'no progression'
        return response, min_comp_SLD




# Load in AxR data for later use
path_to_ARdata = 'C:/Users/Alanna/Desktop/Research_Code/Desktop_research/AxR_data_trimmed.txt'
AxR = []
with open(path_to_ARdata, 'r') as f:
    content = f.readlines()
    for i in content:
        num = float(''.join(list(i)[:-1]))
        AxR.append(num)

# Permute AxR data (with seed) for later use
np.random.seed(757)
AxR = np.random.permutation(AxR)
AxR_ind = 0

# Define parameters for trees
path_base = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_march_23/'
path_addenda = ['MSI_fast/11022721',
                'MSS_fast/11022766']
all_deltas = [0.955, 0.955]
u_c_list = [0.002026, 0.0002327]
ms_stat_list = ['MSI', 'MSS']
speed_list = ['fast', 'fast']
b = 0.25
maxruns = 5000

# Plot neoantigen fitness
axr_df = pd.DataFrame({'axr': AxR})
color_palette = ['orchid', 'purple']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.95)
ax = sns.histplot(data=axr_df, x='axr', stat='percent', log_scale=True, bins=20)
plt.xlabel('Neoantigen Fitness (AxR)')
plt.ylabel('Percent')
plt.savefig(path_base + "AxR_hist.png")
plt.savefig(path_base + "AxR_hist.svg", format='svg')


try:
    # raise Exception('skip this time so we can rerun') # comment this out unless debugging
    therapydata = pd.read_pickle(open(path_base + "therapydata_" + str(maxruns) + "_pandas_df.dump", 'rb'))
    print('Immunotherapy dataset with these parameters already created. Loading...')
except:
    print('Immunotherapy dataset with these parameters not already created. Creating...')
    # Define parameters for tumor growth, model
    sigma = 10  # cells produced per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
    mu = 1e-2  # per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
    t0 = 0
    #a = 1.195  # previously was 1. This now represents per day behavior. b-d per day
    b_tumor = 1e-7  # dimesions: 1 / cells
    m_mult = 6.4  # 9.575 for nivo (sigma = 1, mu = 1); 6.747 for pembro (sigma = 1000, mu = 1)
    k_mult = 1.0  # 0.0579 for nivo (sigma = 1, mu = 1); 0.905 for pembro (sigma = 1000, mu = 1)
    ici_type = 'Pembro'
    m_proportionality_constant = 20 * (b_tumor/mu) * (m_mult) / 10000 # Number in parentheses is multiplier optimized in simulate_therapy_optimize_params.py
    #k_proportionality_constant = 0.5 * (a/sigma) * (k_mult)  # Number in parentheses is multiplier optimized in simulate_therapy_optimize_params.py
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
        path = path_base + path_addenda[i]
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
            a = a_rng.lognormal(mean= -4.483, sigma=0.828)  # sample a from distribution of doubling times for CRC. true mean is -6.448, what we were using: -4.483
            k_proportionality_constant = 0.5 * (a / sigma) * (
                k_mult) / 100  # 1.5 to increase by 50% relative to original Kamran value

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

            if len(init_vals) != len(m):
                print('ICs have different length than the number of (m,k) pairs.')
            AxR_ind = np.copy(new_AxR_ind)

            # Set up ODEfun based on number of subclones that are of the right size
            if len(m) == 1:  # only one clonal pop to keep track of
                dE = lambda E, T: sigma - mu * E + m[0]*E*T
                dT = lambda E, T: a*T*(1 - b_tumor*T) - k[0]*E*T
                def odefun(t,y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1]), dT(y[0], y[1])]
                y0 = [sigma/mu, 1e5]
            elif len(m) == 2:
                dE = lambda E, T1, T2: sigma - mu * E + E * (m[0]*T1 + m[1]*T2)
                dT1 = lambda E, T1, T2: a * T1 * (1 - b_tumor * (T1 + T2)) - E * (k[0] * T1)
                dT2 = lambda E, T1, T2: a * T2 * (1 - b_tumor * (T1 + T2)) - E * (k[1] * T2)

                def odefun(t, y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1], y[2]), dT1(y[0], y[1], y[2]), dT2(y[0], y[1], y[2])]
                y0 = [sigma/mu, init_vals[0], init_vals[1]]
            elif len(m) == 3:
                dE = lambda E, T1, T2, T3: sigma - mu * E + E * (m[0]*T1 + m[1]*T2 + m[2]*T3)
                dT1 = lambda E, T1, T2, T3: a * T1 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[0] * T1)
                dT2 = lambda E, T1, T2, T3: a * T2 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[1] * T2)
                dT3 = lambda E, T1, T2, T3: a * T3 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[2] * T3)

                def odefun(t, y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1], y[2], y[3]), dT1(y[0], y[1], y[2], y[3]), dT2(y[0], y[1], y[2], y[3]), dT3(y[0], y[1], y[2], y[3])]
                y0 = [sigma/mu, init_vals[0], init_vals[1], init_vals[2]]
            elif len(m) == 4:
                dE = lambda E, T1, T2, T3, T4: sigma - mu * E + E * (m[0]*T1 + m[1]*T2 + m[2]*T3 + m[3]*T4)
                dT1 = lambda E, T1, T2, T3, T4: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[0] * T1)
                dT2 = lambda E, T1, T2, T3, T4: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[1] * T2)
                dT3 = lambda E, T1, T2, T3, T4: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[2] * T3)
                dT4 = lambda E, T1, T2, T3, T4: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[3] * T4)

                def odefun(t, y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1], y[2], y[3], y[4]), dT1(y[0], y[1], y[2], y[3], y[4]), dT2(y[0], y[1], y[2], y[3], y[4]), dT3(y[0], y[1], y[2], y[3], y[4]), dT4(y[0], y[1], y[2], y[3], y[4])]
                y0 = [sigma/mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3]]
            elif len(m) == 5:
                dE = lambda E, T1, T2, T3, T4, T5: sigma - mu * E + E * (m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5)
                dT1 = lambda E, T1, T2, T3, T4, T5: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[0] * T1)
                dT2 = lambda E, T1, T2, T3, T4, T5: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[1] * T2)
                dT3 = lambda E, T1, T2, T3, T4, T5: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[2] * T3)
                dT4 = lambda E, T1, T2, T3, T4, T5: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[3] * T4)
                dT5 = lambda E, T1, T2, T3, T4, T5: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[4] * T5)

                def odefun(t, y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1], y[2], y[3], y[4], y[5]), dT1(y[0], y[1], y[2], y[3], y[4], y[5]),
                                       dT2(y[0], y[1], y[2], y[3], y[4], y[5]), dT3(y[0], y[1], y[2], y[3], y[4], y[5]),
                                       dT4(y[0], y[1], y[2], y[3], y[4], y[5]), dT5(y[0], y[1], y[2], y[3], y[4], y[5])]
                y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4]]
            elif len(m) == 6:
                dE = lambda E, T1, T2, T3, T4, T5, T6: sigma - mu * E + E * (
                            m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6)
                dT1 = lambda E, T1, T2, T3, T4, T5, T6: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E * (k[0] * T1)
                dT2 = lambda E, T1, T2, T3, T4, T5, T6: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E * (k[1] * T2)
                dT3 = lambda E, T1, T2, T3, T4, T5, T6: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E * (k[2] * T3)
                dT4 = lambda E, T1, T2, T3, T4, T5, T6: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E * (k[3] * T4)
                dT5 = lambda E, T1, T2, T3, T4, T5, T6: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E * (k[4] * T5)
                dT6 = lambda E, T1, T2, T3, T4, T5, T6: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E * (
                            k[5] * T6)

                def odefun(t, y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                                       dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                                       dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                                       dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6])]
                y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5]]
            elif len(m) == 7:
                dE = lambda E, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E + E * (
                            m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7)
                dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[0] * T1)
                dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[1] * T2)
                dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[2] * T3)
                dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[3] * T4)
                dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[4] * T5)
                dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                            k[5] * T6)
                dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T7 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[6] * T7)

                def odefun(t, y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]), dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                                       dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]), dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                                       dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]), dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                                       dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]), dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])]
                y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5], init_vals[6]]
            elif len(m) == 8:
                dE = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E + E * (
                        m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7 + m[7] * T8)
                dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T1 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[0] * T1)
                dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T2 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[1] * T2)
                dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T3 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[2] * T3)
                dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T4 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[3] * T4)
                dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T5 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[4] * T5)
                dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T6 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[5] * T6)
                dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T7 * (
                            1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[6] * T7)
                dT8 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8: a * T7 * (
                        1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E * (k[7] * T8)

                def odefun(t, y):
                    for mm in range(len(y[1:])):
                        if y[mm+1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                            y[mm+1] = 0
                    return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]),
                                       dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8])]
                y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
                      init_vals[6], init_vals[7]]
            else:
                print('Too many subclones for current iteration of code. Moving to next tree.')
                print('Number of subclones: ' + str(num_subclones))
                kk -= 1
                continue

            # Solve ODE using solve_IVP
            # scale_odefun = lambda t, y: [0.01*x for x in odefun(t, y)]
            Tmax = round(55 * 30.437)  # x 100 to get to days. 84 days total = 12 weeks. Large Tmax = 5, small = 0.84 (12 weeks)
            dt = 1  # "sample" every 1 day.
            t_span = [t0, Tmax]
            t_eval = np.arange(t0, Tmax+dt, dt)
            sol = solve_ivp(odefun, t_span, y0, t_eval=t_eval)
            t = sol.t
            t_pseud_index = round((12*7) / dt + 1) # time to pseudoprogression is 5.7 weeks -- not 12 weeks. Add one for python weirdness
            subclone_sol = sol.y[1:, ]
            effector_sol = sol.y[0, ] # how nondimensionalized?
            TMB_over_time = [sum(subclone_sol[:, iii]) for iii in range(len(t))]

            # Response classifier
            # Detect oscillations:
            does_it_oscillate = detectOscillations(TMB_over_time, t)

            # Response at 84 days, if longer
            response_pseud = getResponse(TMB_over_time, t, t_pseud_index)
            response_tmax = getResponse(TMB_over_time, t, len(t))

            # Check response every n weeks
            response_score_dict = {'CR': 0, 'PR': 1, 'SD': 2, 'PD': 3}
            response_dict = {}
            iw = 1
            best_response = 'PD'
            best_resp_score = -1
            time_to_PD = -1
            pdflag = False
            while round(12*iw*7 + 1) <= Tmax:
                if iw == 1:
                    t_index = round(12 * 7 + 1)  # Check at 12 weeks for the very first time point
                elif (iw*9*7+1) > 365: # past first year, check every 12 weeks.
                    t_index = round(12 * iw * 7 + 1)
                else:
                    t_index = round(9 * iw * 7 + 1)
                cur_resp = getResponse(TMB_over_time, t, t_index)
                if cur_resp == 'PD' and pdflag == False:
                    time_to_PD = t[t_index]
                    pdflag = True
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
                response_dict['wk_' + str(iw*6) + '_response'] = [cur_resp]
                iw += 1
            # Finally, check exactly at Tmax
            cur_resp = response_tmax
            best_resp_score = min(response_score_dict[best_response], response_score_dict[cur_resp])
            if best_resp_score == 0:
                best_response = 'CR'
            elif best_resp_score == 1:
                best_response = 'PR'
            elif best_resp_score == 2:
                best_response = 'SD'
            elif best_resp_score == 3:
                best_response = 'PD'
            if pdflag==False:
                time_to_PD = t[-1] # if not progressed, set PFS to tmax

            if time_to_PD > (24*7 + 1): # compute DCR as in Le et al. 2023
                diseaseControl = 1
            else:
                diseaseControl = 0

            #response_df = pd.DataFrame(response_dict)
            # if getResponse(TMB_over_time, t, t[-1]) == 'PD':  # if PD is achieved by end of trial,
            #     for it in range(1, len(t)):
            #         if getResponse(TMB_over_time, t, t[it]) == 'PD':
            #             time_to_PD = it
            #             break

            LTR = getLongTermResponse(best_response, response_tmax)

            response_label_dict = {'CR': 'Complete Response', 'PR': 'Partial Response', 'SD': 'Stable Disease', 'PD': 'Progressive Disease'}
            pseud, minSLD_percent = getPseudoprogression(TMB_over_time, t)


            ### Uncomment the following if you want to plot timeseries of tumor and effector cell responses over time:
            # plt.rcParams.update({'font.size': 14})
            # plt.rcParams['font.family'] = ['Arial']
            # plt.rc('legend', fontsize=11)
            #
            # plt.figure(figsize=(5, 4))
            # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
            # if num_subclones>0:
            #     if AxR_vals[0]==0:
            #         plt.rc('axes', prop_cycle=(
            #             cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
            #         plt.plot(t/30.4368, subclone_sol.T / TMB_over_time[0])
            #         if len(subclone_sol) > 1:
            #             plt.plot(t/30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)
            #         plt.legend(
            #             ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol) - 1)] + [
            #                     'Total TB'])
            #     else:
            #         plt.rc('axes', prop_cycle=(cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
            #         plt.plot(t/30.4368, subclone_sol.T / TMB_over_time[0])
            #         if len(subclone_sol) > 1:
            #             plt.plot(t/30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)
            #         plt.legend(['Neoantigen ' + str(iii+1) for iii in range(len(subclone_sol))] + ['Total TB'])
            # else:
            #     plt.plot(t/30.4368, subclone_sol.T / TMB_over_time[0], 'fuchsia') # Fuchsia: non-neoantigen containing subclone (Founder)
            #     plt.legend(['Total TB (no neoantigenic clones)'])
            # plt.xlabel('Months since start of therapy')
            # plt.ylabel('Relative tumor volume')
            # plt.xticks([0, 10, 20, 30, 40, 50])
            # plt.ylim([0, 2]) #max(2, max(TMB_over_time)/TMB_over_time[0])])
            # plt.title(LTR)
            # plt.savefig(path + '/tumor_mats/largeTmax_subclone_fig_' + str(kk) + '.png', format='png')
            # plt.savefig(path + '/tumor_mats/largeTmax_subclone_fig_' + str(kk) + '.svg', format='svg')
            #
            # plt.rcParams.update({'font.size': 14})
            # plt.figure(figsize=(5, 4))
            # plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
            # if num_subclones>0:
            #     if AxR_vals[0]==0:
            #         plt.rc('axes', prop_cycle=(
            #             cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
            #         plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0])
            #         if len(subclone_sol) > 1:
            #             plt.plot(t[:t_pseud_index], TMB_over_time[:t_pseud_index] / TMB_over_time[0], 'k--', linewidth=2)
            #         if kk==227:
            #             plt.legend(
            #                 ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol) - 1)] + [
            #                     'Total TB'], loc=(0.60, 0.28))
            #         else:
            #             plt.legend(
            #                 ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol) - 1)] + [
            #                     'Total TB'])
            #     else:
            #         plt.rc('axes', prop_cycle=(cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
            #         plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0])
            #         if len(subclone_sol) > 1:
            #             plt.plot(t[:t_pseud_index], TMB_over_time[:t_pseud_index] / TMB_over_time[0], 'k--', linewidth=2)
            #         plt.legend(['Neoantigen ' + str(iii+1) for iii in range(len(subclone_sol))] + ['Total TB'])
            # else:
            #     plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0], 'fuchsia') # Fuchsia: non-neoantigen containing subclone (Founder)
            #     plt.legend(['Total TB (no neoantigenic clones)'])
            # plt.xlabel('Days since start of therapy')
            # plt.ylabel('Relative tumor volume')
            # plt.ylim([0, 2])
            # plt.title(response_label_dict[best_response])
            # plt.savefig(path + '/tumor_mats/smallTmax_subclone_fig_' + str(kk) + '.png', format='png')
            # plt.savefig(path + '/tumor_mats/smallTmax_subclone_fig_' + str(kk) + '.svg', format='svg')
            ### NB: if kk=227, set legend(loc = (0.62, 0.28))

            #plt.rcdefaults()
            #plt.close('all')
            # Compute and append response variables
            ending_TMB = TMB_over_time[-1]
            # Unmutated fraction at start of sim:
            unmut_frac_IC = tree.size / tree.get_total_size()
            # Combined m-k-init val checker
            # k_weighted_mean = sum([k[i]*init_vals[i] for i in range(len(init_vals))])
            # m_weighted_mean = sum([m[i]*init_vals[i] for i in range(len(init_vals))])
            # combined_weighted_mean = sum([k[i]*m[i]*init_vals[i] for i in range(len(init_vals))])
            axr_weighted_anteginicity = sum([AxR_vals[i]*init_vals[i] for i in range(len(init_vals))])/1e5
            # Peak effector cell response
            peak_E_response = max(effector_sol)
            peak_E_foldchange = peak_E_response/effector_sol[0]
            # AUC effector cell response
            AUC_E_response = np.trapz(effector_sol, x=t)
            dTMB = ending_TMB/starting_TMB
            tree_index = getTreeIndex(tree)
            tumor_dict = {'ms_stat': [ms_stat],
                 'speed': [speed],
                 'is_clonal_neoant': [is_there_a_clonal_neoant],
                 'num_subclones': [num_subclones],
                 'starting_TMB': [starting_TMB],
                 'ending_TMB': [ending_TMB],
                 'dTMB': [dTMB],
                 'pseudoprogression': [pseud],
                 'minSLD_percent': [minSLD_percent],
                 'maxNAquality': [max_NA_quality],
                 'unmut_frac_IC': [unmut_frac_IC],
                 'time_to_progression': [time_to_PD],
                 'peak_effector_response': [peak_E_response],
                 'peak_E_foldchange': [peak_E_foldchange],
                 'AUC_effector_response': [AUC_E_response],
                 'tree_index': [tree_index],
                 'response_pseud': [response_pseud],
                 'response_tmax': [response_tmax],
                 'best_response': [best_response],
                 'tumor_growth_rate': [a],
                 'axr_weighted_anteginicity': [axr_weighted_anteginicity],
                 'oscillation': [does_it_oscillate],
                 'LTR': [LTR],
                 'totalTMB': [totalTMB],
                 'TMB_1perc': [TMB_1perc],
                 'TMB_10perc': [TMB_10perc],
                 'diseaseControl': [diseaseControl]}
            #cur_data = pd.DataFrame({**tumor_dict, **response_dict})
            #therapydata = pd.concat([therapydata, cur_data]) # therapydata.append(cur_data)
            cur_data = pd.DataFrame(tumor_dict)
            therapydata = pd.concat([therapydata, cur_data])
            TMB_timeseries.append(TMB_over_time)
            num_solved += 1
            print('Simulated therapy on ' + ms_stat + ' tumor # ' + str(kk) + ' out of ' + str(maxruns) + ' tumors; best response: ' + best_response)
        all_TMB_timeseries.append(TMB_timeseries)
        all_num_solved.append(num_solved)

    therapydata.index=np.arange(1, len(therapydata)+1)
    dill.dump(therapydata, open(path_base + "therapydata_" + str(maxruns) + "_pandas_df.dump", 'wb'))
    # dill.dump(therapydata, open(path_base + "therapydata_smallTmax" + str(maxruns) + "_pandas_df.dump", 'wb'))
    # dill.dump(therapydata, open(path_base + "therapydata_largeTmax" + str(maxruns) + "_pandas_df.dump", 'wb'))
    print('Immunotherapy dataset with these parameters created. Saving to file...')
    print('Number of runs skipped due to nontrivial IC setup: ' + str(ICskip))

MSIdata = therapydata[therapydata['ms_stat']=='MSI']
MSSdata = therapydata[therapydata['ms_stat']=='MSS']

print('Pseudoprogression rate (MSI): ' + str(sum(MSIdata['pseudoprogression']=='pseudoprogression')))

print('DCR = ' + str(sum(MSIdata['diseaseControl'])/len(MSIdata['diseaseControl'])))
print('ORR = ' + str((sum(MSIdata['best_response']=='PR')+sum(MSIdata['best_response']=='CR'))/len(MSIdata['best_response'])))

ttpData = [i for i in MSIdata['time_to_progression'] if i>0]
print('Median PFS = ' + str(np.median(ttpData)/30.4368))
print('36-month PFS = ' + str(sum([i > 36*30.4368 for i in ttpData])/len(ttpData) * 100))

print('MSI PD best responses: ' + str(sum(MSIdata['best_response']=='PD')/len(MSIdata['best_response']) * 100) + '%')
print('MSI SD best responses: ' + str(sum(MSIdata['best_response']=='SD')/len(MSIdata['best_response']) * 100) + '%')
print('MSI PR best responses: ' + str(sum(MSIdata['best_response']=='PR')/len(MSIdata['best_response']) * 100) + '%')
print('MSI CR best responses: ' + str(sum(MSIdata['best_response']=='CR')/len(MSIdata['best_response']) * 100) + '%')

print('MSI PD 12w responses: ' + str(sum(MSIdata['response_pseud']=='PD')/len(MSIdata['response_pseud']) * 100) + '%')
print('MSI SD 12w responses: ' + str(sum(MSIdata['response_pseud']=='SD')/len(MSIdata['response_pseud']) * 100) + '%')
print('MSI PR 12w responses: ' + str(sum(MSIdata['response_pseud']=='PR')/len(MSIdata['response_pseud']) * 100) + '%')
print('MSI CR 12w responses: ' + str(sum(MSIdata['response_pseud']=='CR')/len(MSIdata['response_pseud']) * 100) + '%')

print('MSS PD 12w responses: ' + str(sum(MSSdata['response_pseud']=='PD')/len(MSSdata['response_pseud']) * 100) + '%')
print('MSS SD 12w responses: ' + str(sum(MSSdata['response_pseud']=='SD')/len(MSSdata['response_pseud']) * 100) + '%')
print('MSS PR 12w responses: ' + str(sum(MSSdata['response_pseud']=='PR')/len(MSSdata['response_pseud']) * 100) + '%')
print('MSS CR 12w responses: ' + str(sum(MSSdata['response_pseud']=='CR')/len(MSSdata['response_pseud']) * 100) + '%')

MSI_DRs = MSIdata[MSIdata['LTR']=='Durable Response']
MSI_ARs = MSIdata[MSIdata['LTR']=='Acquired Resistance']
MSI_NRs = MSIdata[MSIdata['LTR']=='No Response']

print('DR tree index: ' + str(np.mean(MSI_DRs['tree_index'])) + ' +/- ' + str(np.std(MSI_DRs['tree_index'])))
print('AR tree index: ' + str(np.mean(MSI_ARs['tree_index'])) + ' +/- ' + str(np.std(MSI_ARs['tree_index'])))
print('NR tree index: ' + str(np.mean(MSI_NRs['tree_index'])) + ' +/- ' + str(np.std(MSI_NRs['tree_index'])))

print('DR mean: ' + str(np.mean(MSI_DRs['totalTMB'])/1e5))
print('AR mean: ' + str(np.mean(MSI_ARs['totalTMB'])/1e5))
print('NR mean: ' + str(np.mean(MSI_NRs['totalTMB'])/1e5))

print('DR 95% CI: ' + str([i/1e5 for i in stats.norm.interval(0.95, loc=np.mean(MSI_DRs['totalTMB']), scale=np.std(MSI_DRs['totalTMB'])/np.sqrt(len(MSI_DRs['totalTMB'])))]))
print('AR 95% CI: ' + str([i/1e5 for i in stats.norm.interval(0.95, loc=np.mean(MSI_ARs['totalTMB']), scale=np.std(MSI_ARs['totalTMB'])/np.sqrt(len(MSI_ARs['totalTMB'])))]))
print('NR 95% CI: ' + str([i/1e5 for i in stats.norm.interval(0.95, loc=np.mean(MSI_NRs['totalTMB']), scale=np.std(MSI_NRs['totalTMB'])/np.sqrt(len(MSI_NRs['totalTMB'])))]))

m_mult = 6.4
k_mult = 1.0
print('\gamma_m = ' + str((20 / 10000) * m_mult))
print('\gamma_k = ' + str(0.5 * k_mult / 100))



MSI_LTRvsBest = np.zeros((3, 4))
MSI_LTRvsInitial = np.zeros((3, 4))
recist_response_types = ['PD', 'SD', 'PR', 'CR']
LTR_response_types = ['No Response', 'Acquired Resistance', 'Durable Response']

for i in range(len(LTR_response_types)):
    for j in range(len(recist_response_types)):
        MSI_LTRvsBest[i, j] = sum((MSIdata['best_response']==recist_response_types[j])*(MSIdata['LTR']==LTR_response_types[i]))
        MSI_LTRvsInitial[i, j] = sum((MSIdata['response_pseud']==recist_response_types[j])*(MSIdata['LTR']==LTR_response_types[i]))

for j in range(len(recist_response_types)):
    MSI_LTRvsBest[:, j] = MSI_LTRvsBest[:, j]/sum(MSI_LTRvsBest[:, j]) * 100
    MSI_LTRvsInitial[:, j] = MSI_LTRvsInitial[:, j] / sum(MSI_LTRvsInitial[:, j]) * 100






slow_responders = MSIdata[(MSIdata['response_pseud']=='PR') * (MSIdata['response_tmax']=='CR')]
acquire_resisters = MSIdata[(MSIdata['response_pseud']=='PR') * (MSIdata['response_tmax']=='PD')]
stably_PR = MSIdata[(MSIdata['response_pseud']=='PR') * (MSIdata['response_tmax']=='PR')]
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
plt.savefig(path_base + "MSI_durability_response_heatmap.png")
plt.savefig(path_base + "MSI_durability_response_heatmap.svg")

color_palette = ['thistle', 'orchid', 'purple']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
plt.title('MSI LTR vs. 12-week response')
sns.heatmap(data=MSI_LTRvsInitial, xticklabels=['PD', 'SD', 'PR', 'CR'], yticklabels=['NR', 'AR', 'DR'], linewidth=0.5, cmap=sns.cubehelix_palette(as_cmap=True))
plt.xlabel('12-week Response, % originally labeled as')
plt.ylabel('Long Term Response')
plt.savefig(path_base + "MSI_durability_response_12w_heatmap.png")
plt.savefig(path_base + "MSI_durability_response_12w_heatmap.svg")

plt.figure()
plt.title('Heterogeneity in MMR-D original PR responders')
ax = sns.boxplot(data=orig_PR_df, x='response_type_category', y='num_subclones', order=['AcquiredResistance', 'StablyPR', 'SlowCR'])
add_stat_annotation(
    ax, data=orig_PR_df, x='response_type_category', y='num_subclones', order=['AcquiredResistance', 'StablyPR', 'SlowCR'],
    box_pairs=[("AcquiredResistance", "StablyPR"), ("AcquiredResistance", "SlowCR"), ("StablyPR", "SlowCR")],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Number of subclones')
plt.ylim([1, 8.25])
plt.savefig(path_base + "origPRresp_heterogeneity.png")
plt.savefig(path_base + "origPRresp_heterogeneity.svg")


# Plot figure: countplot showing outcomes
plt.figure()
plt.title('Best immunotherapy response by tumor type')
sns.countplot(data=therapydata, x='best_response', hue='ms_stat', order=['PD','SD','PR','CR'])
plt.savefig(path_base + "countplot_therapy.png")
plt.savefig(path_base + "countplot_therapy.pdf", format='pdf')


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
plt.savefig(path_base + "LTR_countplot_therapy.png")
plt.savefig(path_base + "LTR_countplot_therapy.svg", format='svg')

# Figure: tumor burden and response, LTR
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.violinplot(data=MSIdata, x='LTR', y='totalTMB', order=['No Response', 'Acquired Resistance', 'Durable Response'])
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='totalTMB', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.yticks([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5], ['1e5', '2e5', '3e5', '4e5', '5e5', '6e5', '7e5', '8e5'])
plt.ylabel('Total mutations per tumor')
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(path_base + "LTR_MSI_boxplot_totalTMB.png")
plt.savefig(path_base + "LTR_MSI_boxplot_totalTMB.svg", format='svg')

# Figure: treeindex and response, LTR
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.violinplot(data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'])
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='tree_index', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.ylim([0, 1])
plt.xlabel('')
plt.ylabel('Tree index')
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(path_base + "LTR_MSI_boxplot_treeindex.png")
plt.savefig(path_base + "LTR_MSI_boxplot_treeindex.svg", format='svg')

# Figure: tumor weighted antigenicity and response, LTR
color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxplot(data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
                 showfliers=False)
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='axr_weighted_anteginicity', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Weighted mean antigenicity')
plt.ylim([0, 20])
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(path_base + "LTR_MSI_boxplot_axrmean.png")
plt.savefig(path_base + "LTR_MSI_boxplot_axrmean.svg", format='svg')


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
plt.ylim([0, 38])
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(path_base + "LTR_MSI_boxenplot_axrmean.png")
plt.savefig(path_base + "LTR_MSI_boxenplot_axrmean.svg", format='svg')



color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxplot(data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
                 showfliers=False)
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='maxNAquality', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
#plt.yscale('log')
plt.ylim([0, 39])
plt.ylabel('Maximal neoantigen quality')
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.xlabel('')
plt.savefig(path_base + "LTR_MSI_boxplot_maxNAquality.png")
plt.savefig(path_base + "LTR_MSI_boxplot_maxNAquality.svg",format='svg')


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
plt.ylim([0, 61])
plt.ylabel('Maximal neoantigen quality')
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.xlabel('')
plt.savefig(path_base + "LTR_MSI_boxenplot_maxNAquality.png")
plt.savefig(path_base + "LTR_MSI_boxenplot_maxNAquality.svg",format='svg')



frac_with_geq_1_neoant = [1 - i for i in MSIdata['unmut_frac_IC']]
MSIdata.insert(len(MSIdata.columns), 'frac_with_geq_1_neoantigen', frac_with_geq_1_neoant)

color_palette = ['lightcyan', 'cyan', 'darkcyan']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.boxplot(data=MSIdata, x='LTR', y='frac_with_geq_1_neoantigen', order=['No Response', 'Acquired Resistance', 'Durable Response'],
                 showfliers=True)
add_stat_annotation(
    ax, data=MSIdata, x='LTR', y='frac_with_geq_1_neoantigen', order=['No Response', 'Acquired Resistance', 'Durable Response'],
    box_pairs=[("No Response", "Acquired Resistance"), ("No Response", "Durable Response"), ("Acquired Resistance", "Durable Response")],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
#plt.yscale('log')
plt.ylim([0.3, 1.25])
plt.ylabel('Fraction with neoantigen')
ax.set_xticklabels(['NR', 'AR', 'DR'])
ax.set_yticks([0.4, 0.6, 0.8, 1.0])
plt.xlabel('')
plt.savefig(path_base + "LTR_MSI_boxplot_fracgeq1neoant.png")
plt.savefig(path_base + "LTR_MSI_boxplot_fracgeq1neoant.svg", format='svg')

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

LTR_clonaldf = pd.DataFrame(data={'LTR':['No Response', 'Acquired Resistance', 'Durable Response',
                                         'No Response', 'Acquired Resistance', 'Durable Response'],
                                  'frac': [NRfrac_absent, ARfrac_absent, DRfrac_absent,
                                           NRfrac_present, ARfrac_present, DRfrac_present],
                                  'clonal_type': ['Absent','Absent','Absent',
                                           'Present','Present','Present']})

plt.figure()
ax = sns.catplot(
    data=LTR_clonaldf, kind="bar",
    x="LTR", y="frac", col="clonal_type",
    height=4, aspect=.6, edgecolor="k"
)
sns.despine(top=False, right=False)
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.90, wspace=0.05)
ax.set_axis_labels('', '')
ax.set_ylabels('Percent')
ax.set_titles('Clonal Mut. {col_name}')
plt.ylim([0, 100])
ax.set_xticklabels(['NR', 'AR', 'DR'])
plt.savefig(path_base + "LTR_countplot_clonality_improved_MSI.png")
plt.savefig(path_base + "LTR_countplot_clonality_improved_MSI.svg", format='svg')

# Plot figure: comparing clinical data to our data (MSI)
PDfrac = sum(MSIdata['best_response']=='PD')/len(MSIdata['best_response']) * 100
SDfrac = sum(MSIdata['best_response']=='SD')/len(MSIdata['best_response']) * 100
PRfrac = sum(MSIdata['best_response']=='PR')/len(MSIdata['best_response']) * 100
CRfrac = sum(MSIdata['best_response']=='CR')/len(MSIdata['best_response']) * 100
MSI_n_vec = np.array([25, 13, 13, 9])  # PEMBRO: Le et al 2023, cohort B;  NIVO: Overman et al. 2017;
MSI_frac_vec = MSI_n_vec / sum(MSI_n_vec) * 100
MSI_yerrormin = abs(MSI_frac_vec - [27.6, 11.5, 11.5, 6.7])
MSI_yerrmax = abs(MSI_frac_vec - [52.8, 32.7, 32.7, 25.4])
MSI_yerr = [MSI_yerrormin, MSI_yerrmax]



PDfrac2 = sum(MSSdata['response_pseud']=='PD')/len(MSSdata['response_pseud']) * 100
SDfrac2 = sum(MSSdata['response_pseud']=='SD')/len(MSSdata['response_pseud']) * 100
PRfrac2 = sum(MSSdata['response_pseud']=='PR')/len(MSSdata['response_pseud']) * 100
CRfrac2 = sum(MSSdata['response_pseud']=='CR')/len(MSSdata['response_pseud']) * 100
MSS_n_vec = np.array([11, 2, 0, 0])
MSS_frac_vec = MSS_n_vec / sum(MSS_n_vec) * 100
nMSS = sum(MSS_n_vec)
# binomtest(MSS_n_vec[0], nMSS, MSS_frac_vec[0]/100).proportion_ci() # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
MSS_yerrmin = abs(MSS_frac_vec - [54.6, 1.9, 0, 0])
MSS_yerrmax = abs(MSS_frac_vec - [98.1, 45.4, 24.7, 24.7])
MSS_yerr = [MSS_yerrmin, MSS_yerrmax]
## central BICR cohort, excluding not identified pts;
## order {"PD", "SD", "PR", "CR"}

clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MSI_frac_vec[0], MSI_frac_vec[1], MSI_frac_vec[2], MSI_frac_vec[3],
                   PDfrac, SDfrac, PRfrac, CRfrac]}
clincomp_df = pd.DataFrame(data = clincomp)

MSIMSS_color_palette = ['seagreen', 'mediumaquamarine']
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='best_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
x_loc = x_loc[0:4]
y_loc = [p.get_height() for p in ax.patches]
y_loc = y_loc[0:4]
ax.errorbar(x=x_loc, y=y_loc, yerr=MSI_yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
plt.title('MMR-D best response')
plt.savefig(path_base + "clincomp_barplot_best_MSI.png")
plt.savefig(path_base + "clincomp_barplot_best_MSI.svg", format='svg')

clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[MSS_frac_vec[0], MSS_frac_vec[1], MSS_frac_vec[2], MSS_frac_vec[3],
                   PDfrac2, SDfrac2, PRfrac2, CRfrac2]}
clincomp_df = pd.DataFrame(data = clincomp)
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='best_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
x_loc = x_loc[0:4]
y_loc = [p.get_height() for p in ax.patches]
y_loc = y_loc[0:4]
ax.errorbar(x=x_loc, y=y_loc, yerr=MSS_yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.title('MMR-P response at 12 weeks')
plt.savefig(path_base + "clincomp_barplot_12wk_MSS.png")
plt.savefig(path_base + "clincomp_barplot_12wk_MSS.svg", format='svg')

PDfrac3 = sum(MSIdata['response_pseud']=='PD')/len(MSIdata['response_pseud']) * 100
SDfrac3 = sum(MSIdata['response_pseud']=='SD')/len(MSIdata['response_pseud']) * 100
PRfrac3 = sum(MSIdata['response_pseud']=='PR')/len(MSIdata['response_pseud']) * 100
CRfrac3 = sum(MSIdata['response_pseud']=='CR')/len(MSIdata['response_pseud']) * 100
n_vec = np.array([1, 5, 4, 0])
frac_vec = n_vec / sum(n_vec) * 100
numsamp = sum(n_vec)
# binomtest(n_vec[2], numsamp, frac_vec[2]/100).proportion_ci() # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.

yerrmin = abs(frac_vec - [0.25, 18.7, 12.2, 0])
yerrmax = abs(frac_vec - [44.5, 81.2, 73.8, 30.8])
yerr = [yerrmin, yerrmax]
clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[frac_vec[0], frac_vec[1], frac_vec[2], frac_vec[3],
                   PDfrac3, SDfrac3, PRfrac3, CRfrac3]}
clincomp_df = pd.DataFrame(data = clincomp)
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='best_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
x_loc = x_loc[0:4]
y_loc = [p.get_height() for p in ax.patches]
y_loc = y_loc[0:4]
ax.errorbar(x=x_loc, y=y_loc, yerr=yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
plt.title('MMR-D response at 12 weeks')
plt.savefig(path_base + "clincomp_barplot_12wk_MSI.png")
plt.savefig(path_base + "clincomp_barplot_12wk_MSI.svg", format='svg')


ttpData = [i for i in MSIdata['time_to_progression'] if i>0]
median_pfs = np.median(ttpData)/30.4368 # divide by 30.4 to get days -> months
mean_pfs = np.average(ttpData)/30.4368
month36_pfs = sum([i > 36*30.4368 for i in ttpData])/len(ttpData) * 100
month24_pfs = sum([i > 24*30.4368 for i in ttpData])/len(ttpData) * 100
clinical_median_pfs = 4.1
clinical_month36_pfs = 34.1
clinical_month24_pfs = 36.7

yerrmin = abs(clinical_median_pfs - 2.1)
yerrmax = abs(clinical_median_pfs - 18.9)
yerr = [[yerrmin], [yerrmax]]
clincomp = {'Source': ['Clinical', 'Simulation', 'Clinical', 'Simulation'], #, 'Clinical', 'Simulation'],
            'PFS_type': ['Median', 'Median', '36-month', '36-month'], # '24-month', '24-month', '36-month', '36-month'],
           'PFS': [clinical_median_pfs, median_pfs, clinical_month36_pfs, month36_pfs]} # clinical_month24_pfs, month24_pfs, clinical_month36_pfs, month36_pfs]}
clincomp_df = pd.DataFrame(data = clincomp)


fig, ax = plt.subplots(1, 2, figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax_0 = sns.barplot(data=clincomp_df[2:], x='PFS_type', y='PFS', hue='Source', ax=ax[0])
ax[0].set_xlabel('')
ax[0].set_ylabel('Percent')
ax[0].legend(loc = 'upper right')
ax[0].set_ylim([0, 49])
ax[0].set_title('MMR-D PFS')

ax_1 = sns.barplot(data=clincomp_df[:2], x='PFS_type', y='PFS', hue='Source', ax=ax[1])
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax_1.patches]
x_loc = x_loc[0]
y_loc = [p.get_height() for p in ax_1.patches]
y_loc = y_loc[0]
ax[1].errorbar(x=x_loc, y=y_loc, yerr=yerr, fmt='.', c='k')
ax[1].set_xlabel('')
ax[1].set_ylabel('Months')
ax[1].set_ylim([0, 28])
ax[1].set_title('MMR-D PFS')
ax[1].legend(loc = 'upper right')

plt.savefig(path_base + "subplots_clincomp_PFS.png")
plt.savefig(path_base + "subplots_clincomp_PFS.svg", format='svg')

PDfrac3 = sum(MSIdata['best_response']=='PD')/len(MSIdata['best_response']) * 100
SDfrac3 = sum(MSIdata['best_response']=='SD')/len(MSIdata['best_response']) * 100
PRfrac3 = sum(MSIdata['best_response']=='PR')/len(MSIdata['best_response']) * 100
CRfrac3 = sum(MSIdata['best_response']=='CR')/len(MSIdata['best_response']) * 100
n_vec = np.array([12, 19, 18, 1])  # number PD, SD, PR, CR in Nivo paper (BICR central assessment cohort, Overman 2017)
frac_vec = n_vec / sum(n_vec) * 100
numsamp = sum(n_vec)
# binomtest(n_vec[2], numsamp, frac_vec[2]/100).proportion_ci() # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.

yerrmin = abs(frac_vec - [13.1, 24.7, 22.9, 0])
yerrmax = abs(frac_vec - [38.2, 52.8, 50.8, 10.6])
yerr = [yerrmin, yerrmax]
clincomp = {'Source':['Clinical','Clinical','Clinical','Clinical', 'Simulation','Simulation','Simulation','Simulation'],
           'best_response':['PD','SD','PR','CR','PD','SD','PR','CR'],
           'frac':[frac_vec[0], frac_vec[1], frac_vec[2], frac_vec[3],
                   PDfrac3, SDfrac3, PRfrac3, CRfrac3]}
clincomp_df = pd.DataFrame(data = clincomp)
plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.1, right=0.9, wspace=0.35, hspace=0.45)
plt.rcParams.update({'font.size': 16})
sns.set_palette(sns.color_palette(MSIMSS_color_palette))
ax = sns.barplot(data=clincomp_df, x='best_response', y='frac', hue='Source')
x_loc = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
x_loc = x_loc[0:4]
y_loc = [p.get_height() for p in ax.patches]
y_loc = y_loc[0:4]
ax.errorbar(x=x_loc, y=y_loc, yerr=yerr, fmt='.', c='k')
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title=None)
plt.xlabel('')
plt.ylabel('Percent')
plt.title('MMR-D best response (Nivolumab)')
plt.savefig(path_base + "clincomp_best_nivo_MSI.png")
plt.savefig(path_base + "clincomp_best_nivo_MSI.svg", format='svg')


nonclonal_DRs = MSIdata.loc[(MSIdata['LTR']=='Durable Response') & (MSIdata['is_clonal_neoant']=='Absent')]
nonclonal_ARs = MSIdata.loc[(MSIdata['LTR']=='Acquired Resistance') & (MSIdata['is_clonal_neoant']=='Absent')]
nonclonal_NRs = MSIdata.loc[(MSIdata['LTR']=='No Response') & (MSIdata['is_clonal_neoant']=='Absent')]
MSInoclonal = MSIdata.loc[MSIdata['is_clonal_neoant']=='Absent']
all(MSIdata['LTR'].loc[MSIdata['is_clonal_neoant']=='Present']=='Durable Response') # clonal mutation is SUFFICIENT for DR
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
plt.ylim([0, 61])
plt.ylabel('Maximal neoantigen quality')
plt.xlabel('')
plt.savefig(path_base + "DR_LTR_MSI_boxenplot_maxNAquality.png")
plt.savefig(path_base + "DR_LTR_MSI_boxenplot_maxNAquality.svg", format='svg')

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
plt.ylim([0, 38])
plt.savefig(path_base + "DR_LTR_MSI_boxenplot_axrmean.png")
plt.savefig(path_base + "DR_LTR_MSI_boxenplot_axrmean.svg", format='svg')

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.violinplot(data=MSI_DRs, x='is_clonal_neoant', y='totalTMB', order=['Absent', 'Present'])
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='totalTMB', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.yticks([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5], ['1e5', '2e5', '3e5', '4e5', '5e5', '6e5', '7e5', '8e5'])
plt.ylabel('Total mutations per MMR-D tumor')
plt.savefig(path_base + "DR_LTR_MSI_boxplot_totalTMB.png")
plt.savefig(path_base + "DR_LTR_MSI_boxplot_totalTMB.svg", format='svg')

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.violinplot(data=MSI_DRs, x='is_clonal_neoant', y='tree_index', order=['Absent', 'Present'])
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='tree_index', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='outside', verbose=2, fontsize=9)
plt.ylim([0, 1.1])
plt.xlabel('')
plt.ylabel('Tree index')
plt.savefig(path_base + "DR_LTR_MSI_boxplot_treeindex.png")
plt.savefig(path_base + "DR_LTR_MSI_boxplot_treeindex.svg", format='svg')

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.violinplot(data=MSI_DRs, x='is_clonal_neoant', y='totalTMB', order=['Absent', 'Present'])
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='totalTMB', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.yticks([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5], ['1e5', '2e5', '3e5', '4e5', '5e5', '6e5', '7e5', '8e5'])
plt.ylabel('Total mutations per MMR-D tumor')
plt.savefig(path_base + "DR_LTR_MSI_boxplot_totalTMB.png")
plt.savefig(path_base + "DR_LTR_MSI_boxplot_totalTMB.svg", format='svg')

sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom = 0.2, top = 0.95)
ax = sns.violinplot(data=MSI_DRs, x='is_clonal_neoant', y='tumor_growth_rate', order=['Absent', 'Present'])
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='tumor_growth_rate', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Tumor growth rate')
plt.savefig(path_base + "DR_LTR_MSI_boxplot_growthrate.png")
plt.savefig(path_base + "DR_LTR_MSI_boxplot_growthrate.svg", format='svg')

plt.figure()
ax = sns.boxplot(data=MSI_DRs, x='is_clonal_neoant', y='num_subclones', order=['Absent', 'Present'])
add_stat_annotation(
    ax, data=MSI_DRs, x='is_clonal_neoant', y='num_subclones', order=['Absent', 'Present'],
    box_pairs=[('Absent', 'Present')],
    test='t-test_welch', text_format='star', loc='inside', verbose=2, fontsize=9)
plt.xlabel('')
plt.ylabel('Number of subclones')
plt.ylim([1, 8.25])
plt.savefig(path_base + "DR_MSI_heterogeneity.png")
plt.savefig(path_base + "DR_MSI_heterogeneity.svg")
