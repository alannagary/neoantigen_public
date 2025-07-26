from scipy.integrate import solve_ivp
import dill
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statannot import add_stat_annotation
from cycler import cycler
import sys

# Class and functions for data_gen_Hyak:
parser = argparse.ArgumentParser('data_gen_Hyak.py')
parser.add_argument('--task_id', type=str, default='0')
parser.add_argument('--db_frac', type=float, default=0.72)
parser.add_argument('--mut_rate', type=float, default=0.015)
parser.add_argument('--job_id', type=str, default='0')
args = parser.parse_args()

class Population:

    def __init__(self, parent=None, size=1, birth_rate=0.25, basal_death_rate=0.25 * args.db_frac, death_rate=0.25 * args.db_frac,
                 birth_prob=0.25 / (0.25 + (0.25 * args.db_frac) + args.mut_rate), neoant_mut_rate=1, mut_rate = args.mut_rate,  # CHANGED FROM 0.1
                 mutation_prob=args.mut_rate / (0.25 + (0.25 * args.db_frac) + args.mut_rate),
                 pop_size_at_appearance=0, order=0, **kwargs): #extra arguments get passed to kwargs, and doesn't crash the script
        self.parent = parent  # Population object
        self.children = []  # Population object
        self.size = size  # Variable, int
        self.neoant_fitness = [0]  # Variable, list, each element needs to be >0
        self.birth_rate = birth_rate  # Fixed parameter
        self.neoant_mut_rate = neoant_mut_rate  # Fixed parameter
        self.mut_rate = mut_rate # Fixed parameter
        self.mutation_prob = mutation_prob  # Fixed parameter
        self.basal_death_rate = basal_death_rate  # Fixed parameter
        self.pop_size_at_appearance = pop_size_at_appearance  # Fixed at appearance, int
        self.order = order  # Fixed at appearance, int
        self.death_rate = death_rate  # (1 - max(self.neoant_fitness)) * (self.basal_death_rate - 1) + 1  # Fixed at appearance, float
        self.birth_prob = birth_prob  # self.birth_rate / (self.birth_rate + self.death_rate)  # Fixed at appearance, float

    def get_total_size(self):
        return self.size + sum([child.get_total_size() for child in self.children])

    def _proceed_search(self, count, number):
        if count + self.size >= number:
            return self, count
        else:
            new_count = count + self.size
            for child in self.children:
                node, new_count = child._proceed_search(new_count, number)
                if node is not None:
                    return node, new_count
            return None, new_count

    def add_one_member(self):
        self.size += 1

    def remove_one_member(self):
        assert self.size > 0, "Death in an empty population"
        self.size -= 1
        my_total_size = self.get_total_size()
        if my_total_size == 0:
            self.trim_back_to_living_branch()

    def trim_back_to_living_branch(self):
        if self.parent is not None:
            parent_total_size = self.parent.get_total_size()
            if parent_total_size == 0:
                self.parent.trim_back_to_living_branch()
            else:
                self.parent.children.remove(self)

    def find_population_of_member(self, number):
        node, _ = self._proceed_search(0, number)
        return node

    def print_population(self, mutations):
        print(mutations, ": ", self.size)
        for i, child in enumerate(self.children):
            child.print_population(mutations + ("%d" % i))


def store_children_as_dict(root, root_dict):
    if len(root.children) == 0:
        return {None}
    else:
        nested_child_dict = {}
        i = 0
        for child in root.children:
            nested_child_dict[i] = {  # 'parent': child.parent,
                'size': child.size,
                'neoant_fitness': child.neoant_fitness,
                'birth_rate': child.birth_rate,
                'neoant_mut_rate': child.neoant_mut_rate,
                'mut_rate': child.mut_rate,
                'basal_death_rate': child.basal_death_rate,
                'death_rate': child.death_rate,
                'birth_prob': child.birth_prob,
                'mutation_prob': child.mutation_prob,
                'pop_size_at_appearance': child.pop_size_at_appearance,
                'order': child.order}
            nested_child_dict[i]['parent'] = root_dict
            nested_child_dict[i]['children'] = store_children_as_dict(child, nested_child_dict[i])
            i += 1
    return nested_child_dict


def store_root_as_dict(root):
    assert root.parent is None
    root_dict = {'parent': root.parent,
                 'size': root.size,
                 'neoant_fitness': root.neoant_fitness,
                 'birth_rate': root.birth_rate,
                 'neoant_mut_rate': root.neoant_mut_rate,
                 'mut_rate': root.mut_rate,
                 'basal_death_rate': root.basal_death_rate,
                 'death_rate': root.death_rate,
                 'birth_prob': root.birth_prob,
                 'mutation_prob': root.mutation_prob,
                 'pop_size_at_appearance': root.pop_size_at_appearance,
                 'order': root.order}
    root_dict['children'] = store_children_as_dict(root, root_dict)
    return root_dict


def unpack_children_to_Population(root, root_dict):
    pass
    if root_dict['children'] == {None}:
        return root
    else:
        for child_dict in root_dict['children']:
            new_child = Population(parent=root,
                                   size=root_dict['children'][child_dict]['size'],
                                   birth_rate=root_dict['children'][child_dict]['birth_rate'],
                                   basal_death_rate=root_dict['children'][child_dict]['basal_death_rate'],
                                   neoant_mut_rate=root_dict['children'][child_dict]['neoant_mut_rate'],
                                   mut_rate=root_dict['children'][child_dict]['mut_rate'],
                                   mutation_prob=root_dict['children'][child_dict]['mutation_prob'],
                                   pop_size_at_appearance=root_dict['children'][child_dict]['pop_size_at_appearance'],
                                   neoant_fitness=root_dict['children'][child_dict]['neoant_fitness'],
                                   order=root_dict['children'][child_dict]['order'])
            new_child.death_rate = (1 - max(new_child.neoant_fitness)) * (new_child.basal_death_rate - 1) + 1
            new_child.birth_prob = new_child.birth_rate / (new_child.birth_rate + new_child.death_rate + new_child.mut_rate)
            new_child.mutation_prob = new_child.mut_rate / (new_child.birth_rate + new_child.death_rate + new_child.mut_rate)
            new_child = unpack_children_to_Population(new_child,
                                                      root_dict['children'][child_dict])  # changed from new_child.child
            root.children.append(new_child)
    return root


def unpack_root_dict_to_Population(root_dict):
    assert root_dict['parent'] is None
    root = Population(parent=None,
                      size=root_dict['size'],
                      birth_rate=root_dict['birth_rate'],
                      basal_death_rate=root_dict['basal_death_rate'],
                      neoant_mut_rate=root_dict['neoant_mut_rate'],
                      mut_rate=root_dict['mut_rate'],
                      mutation_prob=root_dict['mutation_prob'],
                      pop_size_at_appearance=root_dict['pop_size_at_appearance'],
                      neoant_fitness=root_dict['neoant_fitness'],
                      order=root_dict['order'])
    root.death_rate = root.basal_death_rate  # we know it's the basal death rate because we require starting from the root
    root.birth_prob = root.birth_rate / (root.birth_rate + root.death_rate + root.mut_rate)
    root.mutation_prob = root.mut_rate / (root.birth_rate + root.death_rate + root.mut_rate)
    root = unpack_children_to_Population(root, root_dict)
    return root

def generate_simple_label(db_frac, mut_rate, task_id):
    lab = ''
    # Label by MMRD or MMRP status:
    if np.isclose(mut_rate, 9.8e-3):
        lab += 'MMRD_'
    elif np.isclose(mut_rate, 1.1e-3):
        lab += 'MMRP_'
    else:
        lab += 'CustomMutationRate_' + str(mut_rate)

    # Label by dbfrac if not standard:
    if not np.isclose(db_frac, 0.2387 / 0.25):
        lab += 'CustomGrowthRate_' + str(round(db_frac, 4))

    # Label by task ID:
    lab += str(task_id)
    return lab


def generate_tree_simple(db_frac, mut_rate, task_id, path):
    os.makedirs(path + "data/", exist_ok=True) # create directory for tree outputs
    sys.setrecursionlimit(
        10 ** 6)  # This is required for high-mutation-rate trees. If you increase the mutation rate even more, you may need to set the recursion limit even higher!
    fitness_scaler = 0
    np.random.seed(int(int(task_id) + 1e5*db_frac + 1e7*mut_rate))
    while True:
        total_size = 1
        times = []
        birth_rate = 0.25
        basal_death_rate = birth_rate * db_frac
        birth_prob = birth_rate / (birth_rate + basal_death_rate + mut_rate)
        mut_prob = mut_rate / (birth_rate + basal_death_rate + mut_rate)
        maxpop = 1e5
        root = Population(size=total_size,
                          birth_rate=birth_rate,
                          basal_death_rate=basal_death_rate,
                          birth_prob=birth_prob,
                          mut_rate=mut_rate,
                          mutation_prob=mut_prob)
        # maxpop = 1e4
        iter = 0
        current_time = 0
        iter_gen = 10 ** 6
        mut_order = 0
        taus = np.random.exponential(1, (iter_gen,))
        deciders = np.random.uniform(0, 1, (
            4, iter_gen))  # 3 is to decide if we get a neoantigen, 4 is to set its fitness cost

        while maxpop > total_size > 0:
            iter += 1
            if iter >= iter_gen:
                iter = 0
                taus = np.random.exponential(1, (iter_gen,))
                deciders = np.random.uniform(0, 1, (4, iter_gen))
            # if total_size % (maxpop/100) == 0:
            #   print(total_size)
            dt = taus[iter] / total_size  # note that if b + d = 1, then Rtot = (b + d)*total_size = total_size
            current_time += dt
            times.append(current_time)
            member_number = np.random.randint(low=1,
                                              high=(total_size + 1))  # Can also try np.rint(deciders[2,iter]*total_size)
            population_of_event = root.find_population_of_member(member_number)
            birth = population_of_event.birth_prob > deciders[0, iter]
            mut = (population_of_event.birth_prob + population_of_event.mutation_prob) > deciders[0, iter]
            if birth:
                population_of_event.add_one_member()
                total_size += 1
            elif mut:
                mut_order += 1
                neoant_mutation = population_of_event.neoant_mut_rate > deciders[1, iter]
                if neoant_mutation:
                    new_child = Population(parent=population_of_event,
                                           pop_size_at_appearance=total_size,
                                           order=mut_order)
                    new_child.neoant_fitness = population_of_event.neoant_fitness + [
                        fitness_scaler * deciders[2, iter]]
                else:
                    new_child = Population(parent=population_of_event,
                                           pop_size_at_appearance=total_size,
                                           order=mut_order)
                    new_child.neoant_fitness = population_of_event.neoant_fitness
                new_child.death_rate = (1 - max(new_child.neoant_fitness)) * (
                        new_child.basal_death_rate - 1) + 1
                new_child.birth_prob = new_child.birth_rate / (
                        new_child.birth_rate + new_child.death_rate + new_child.mut_rate)
                new_child.mutation_prob = new_child.mut_rate / (new_child.birth_rate + new_child.death_rate + new_child.mut_rate)
                population_of_event.children.append(new_child)  # add new clonal population
                population_of_event.remove_one_member() # and remove one from existing population, because this is NOT a birth event!
                # Total size does not change in mutation case.
            else:
                population_of_event.remove_one_member()
                total_size -= 1

        if total_size > 0:
            break

    # Save the tree of Population objects as a tree of dictionaries
    root_dict = store_root_as_dict(root)
    tree_lab = generate_simple_label(db_frac, mut_rate, task_id)
    dill.dump(root_dict,
              open(path + "data/dilltree_" + tree_lab + ".dump", 'wb'))
    return root, root_dict

# Functions for analyze_raw_tumors:
def unpack_children_to_Population(root, root_dict):
    pass
    if root_dict['children'] == {None}:
        return root
    else:
        for child_dict in root_dict['children']:
            new_child = Population(parent=root,
                                   size=root_dict['children'][child_dict]['size'],
                                   birth_rate=root_dict['children'][child_dict]['birth_rate'],
                                   basal_death_rate=root_dict['children'][child_dict]['basal_death_rate'],
                                   neoant_mut_rate=root_dict['children'][child_dict]['neoant_mut_rate'],
                                   mut_rate=root_dict['children'][child_dict]['mut_rate'],
                                   mutation_prob=root_dict['children'][child_dict]['mutation_prob'],
                                   pop_size_at_appearance=root_dict['children'][child_dict]['pop_size_at_appearance'],
                                   neoant_fitness=root_dict['children'][child_dict]['neoant_fitness'],
                                   order=root_dict['children'][child_dict]['order'])
            new_child.death_rate = (1 - max(new_child.neoant_fitness)) * (new_child.basal_death_rate - 1) + 1
            new_child.birth_prob = new_child.birth_rate / (new_child.birth_rate + new_child.death_rate + new_child.mut_rate)
            new_child.mutation_prob = new_child.mut_rate / (new_child.birth_rate + new_child.death_rate + new_child.mut_rate)
            new_child = unpack_children_to_Population(new_child,
                                                      root_dict['children'][child_dict])  # changed from new_child.child
            root.children.append(new_child)
    return root


def unpack_root_dict_to_Population(root_dict):
    assert root_dict['parent'] is None
    root = Population(parent=None,
                      size=root_dict['size'],
                      birth_rate=root_dict['birth_rate'],
                      basal_death_rate=root_dict['basal_death_rate'],
                      neoant_mut_rate=root_dict['neoant_mut_rate'],
                      mut_rate=root_dict['mut_rate'],
                      mutation_prob=root_dict['mutation_prob'],
                      pop_size_at_appearance=root_dict['pop_size_at_appearance'],
                      neoant_fitness=root_dict['neoant_fitness'],
                      order=root_dict['order'])
    root.death_rate = root.basal_death_rate  # we know it's the basal death rate because we require starting from the root
    root.birth_prob = root.birth_rate / (root.birth_rate + root.death_rate + root.mut_rate)
    root.mutation_prob = root.mut_rate / (root.birth_rate + root.death_rate + root.mut_rate)
    root = unpack_children_to_Population(root, root_dict)
    return root


def get_total_size(self):
    return self.size + sum([child.get_total_size() for child in self.children])


def getNumNeoant(node, curnum):
    if node:
        if node.parent is not None:
            curnum = getNumNeoant(node.parent, curnum+1)
        return curnum

def neoant_node_info_search(node, totmut):
    node_info_list = []
    for child in node.children:
        node_info_list.append((child.order, child.size, get_total_size(child), max(child.neoant_fitness),
                                   getNumNeoant(child, 0), totmut + 1))
        if len(child.children) is not 0:
            node_info_list = node_info_list + neoant_node_info_search(child, totmut + 1)
    return node_info_list


def is_node_present(node, my_ord):
    if node.order == my_ord:
        flag = 1
    else:
        flag = 0
        children_orders = [child.order for child in node.children]
        if children_orders.count(my_ord) > 0:
            flag = 1
        else:
            for child in node.children:
                flag = is_node_present(child, my_ord)
                if flag == 1:
                    break
    return flag


def find_ancestor(node, neoant_ord, parent_list):
    for child in node.children:
        if is_node_present(child, neoant_ord):
            parent_list.append(node.order)
            if child.order == neoant_ord:
                break
            else:
                parent_list = find_ancestor(child, neoant_ord, parent_list)
    return parent_list


def get_abundance_alt(neoant_info_list):
    orders = [clone[0] for clone in neoant_info_list]
    total_na_containing_clones = len(orders)
    unique_pops = np.zeros((total_na_containing_clones,))
    mut_order = np.zeros((total_na_containing_clones,))
    total_pops = np.zeros((total_na_containing_clones,))
    for k in range(total_na_containing_clones):
        clone = neoant_info_list[k]
        unique_pops[k] = clone[1]  # size of self, NOT including descendants w/ additional mutations
        mut_order[k] = clone[0]
        total_pops[k] = clone[2]
    return unique_pops, total_pops, mut_order


def getFirst(item):
    return item[0]


def get_abundance_mat(node,neoant_info_list,trunc):
    neoant_info_list.sort(key=getFirst)
    orders = [clone[0] for clone in neoant_info_list]
    mat = np.zeros((trunc, trunc))
    for k in range(trunc):
        clone = neoant_info_list[k]
        neoant_ord = clone[0]
        mat[k, orders.index(neoant_ord)] = 1  # we will want the matrix value of this index to be 1
        if clone[4] > 1:
            parental_path = find_ancestor(node, neoant_ord, [])
            for n in range(len(parental_path)):
                if orders.count(parental_path[n]) > 0:
                    mat[k, orders.index(parental_path[n])] = 1
    return mat


def listify_data(path, ms_stat, speed, maxruns, excel_trunc=100, excel_true=True):
    try:
        phylos = dill.load(open(path + f"/phylos_{ms_stat}_{speed}_totruns_{maxruns}.dump", 'rb'))
        unmut = dill.load(open(path + f"/unmut_{ms_stat}_{speed}_totruns_{maxruns}.dump", 'rb'))
        return phylos, unmut
    except:
        print('No phylos file found. Analyzing data for the first time with these settings.')
    tree_dicts = [f for f in os.listdir(path) if f.startswith('dilltree_' + speed + '_' + ms_stat)]
    phylos = []
    unmut = []
    i = 0
    EOFflag = 0
    for dilled_tree_dict in tree_dicts:
        i += 1
        if i > maxruns:
            break
        # Unpack and process
        try:
            tree_dict = dill.load(open(path + '/' + dilled_tree_dict, "rb"))
        except EOFError:
            EOFflag += 1
            print('EOF Skip, i = ' + '%0.f' % i)
            continue
        tree = unpack_root_dict_to_Population(tree_dict)
        neoant_info_list = neoant_node_info_search(tree, 0)
        unique_pops, adj_pops, ords = get_abundance_alt(neoant_info_list)
        sort_adjpops = adj_pops[np.argsort(ords)]

        phylos.append(sort_adjpops)  # store matrix sorted by order of appearance
        unmut.append(tree.size) # this is the number of unmutated, parental pop


        if excel_true:
            mat = get_abundance_mat(tree, neoant_info_list, excel_trunc)
            newMat = np.zeros((np.shape(mat)[0], np.shape(mat)[0] + 2))
            newMat[:, 0] = adj_pops[:excel_trunc]  # TOTAL populations with a given mutation
            newMat[:, 1] = unique_pops[:excel_trunc]  # UNIQUE populations
            newMat[:, 2:] = mat  # sort rows of sMat by descending order of population

            newMat = newMat[np.argsort(-newMat[:, 0]), :]  # sort by DESCENDING ORDER based on TOTAL population
            newNewMat = np.zeros((np.shape(newMat)[0]+1, np.shape(newMat)[1]))
            newNewMat[0, 0] = tree.get_total_size()
            newNewMat[0, 1] = tree.size
            newNewMat[1:, :] = newMat

            df = pd.DataFrame(newNewMat)
            filepath = path + '/xls_tumors/' + dilled_tree_dict + '_trunc_' + str(excel_trunc) + '.xlsx'
            df.to_excel(filepath, index=False)

        print(str(round(i / maxruns * 100, 1)) + '% done')
    dill.dump(phylos,
              open(path + f"/phylos_{ms_stat}_{speed}_totruns_{maxruns}.dump", 'wb'))
    dill.dump(unmut,
              open(path + f"/unmut_{ms_stat}_{speed}_totruns_{maxruns}.dump", 'wb'))

    return phylos, unmut


# Functions for simulate_therapy_optimize_params:
def getWeights_overall(expected_vals, rel_err):
    #expected_vals = [24, 38, 36, 2]  # PD, SD, PR, CR, overall - INPUT specific data
    w = [1, 0, 0, 0]
    for i in range(1, len(w)):
        w[i] = (w[i - 1] * expected_vals[i - 1] * rel_err[i - 1]) / (expected_vals[i] * rel_err[i])
    wnorm = np.linalg.norm(w) # norm these
    w = [i/wnorm for i in w] # and divide to normalize weights
    return w

def getWeights_12wk(expected_vals, abs_err_CR, rel_err):
    # expected_vals = [10, 50, 40, 0]  # PD, SD, PR, CR, overall
    w = [0, 0, 0, 10]
    w[2] = (w[3]*abs_err_CR)/(rel_err[2]*expected_vals[2])
    w[1] = (rel_err[2]*expected_vals[2]*w[2]) / (rel_err[1] * expected_vals[1])
    w[0] = (rel_err[1] * expected_vals[1] * w[1]) / (rel_err[0] * expected_vals[0])
    wnorm = np.linalg.norm(w) # norm these
    w = [i/wnorm for i in w] # and be sure to divide by it so that we're not weighting one over the other
    return w

def get_subclones(path, tree, treenum, ms_stat, min_size=10000, save_df=False):
    try:
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
        is_there_a_clonal_neoant = tree.size==0
        if save_df is True:
            dill.dump([newNewMat, num_subclones, is_there_a_clonal_neoant],
                      open(path + '/tumor_mats' + f"/df_{ms_stat}_{treenum}.dump", 'wb'))

    return newNewMat, num_subclones, is_there_a_clonal_neoant

def assign_k_and_m(AxR_data, AxR_ind, mat, num_subclones, b_tumor, mu, a, sigma, m_prop_const, k_prop_const):
    AxR = AxR_data[AxR_ind:(AxR_ind + num_subclones)]
    m_0 = 0 #1e-10 #(b_tumor / mu) / 100000 # baseline clonal m, no mut
    k_0 = 0 #1e-10 #(a / sigma) / 1000 # baseline clonal k, no mut
    m = [m_0]  # else, we need to assign a (m,k) pair to the parental, unmutated population.
    k = [k_0]
    for j in range(1, num_subclones + 1):  # need to add one for the parental population
        potential_AxRs = mat[j][2:] * AxR[:num_subclones]  # find all possible AxR for this particular subclone.
        clone_AxR = max(potential_AxRs)
        m.append(m_0 + m_prop_const * clone_AxR)
        k.append(k_0 + k_prop_const * clone_AxR)
    new_ind = AxR_ind + num_subclones
    if new_ind + 8 >= len(AxR_data): # take into account that there may be >= 8 subclones
        new_ind = new_ind + 8 - len(AxR_data)
    return m, k, new_ind

def define_odefun_polyclonal(m, k, sigma, mu, a, b_tumor, init_vals):
    if len(m) == 1:  # only one clonal pop to keep track of
        dE = lambda E, T: sigma - mu * E + m[0] * E * T
        dT = lambda E, T: a * T * (1 - b_tumor * T) - k[0] * E * T

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1]), dT(y[0], y[1])]

        y0 = [sigma / mu, 1e5]
    elif len(m) == 2:
        dE = lambda E, T1, T2: sigma - mu * E + E * (m[0] * T1 + m[1] * T2)
        dT1 = lambda E, T1, T2: a * T1 * (1 - b_tumor * (T1 + T2)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2: a * T2 * (1 - b_tumor * (T1 + T2)) - E * (k[1] * T2)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2]), dT1(y[0], y[1], y[2]), dT2(y[0], y[1], y[2])]

        y0 = [sigma / mu, init_vals[0], init_vals[1]]
    elif len(m) == 3:
        dE = lambda E, T1, T2, T3: sigma - mu * E + E * (m[0] * T1 + m[1] * T2 + m[2] * T3)
        dT1 = lambda E, T1, T2, T3: a * T1 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3: a * T2 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3: a * T3 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[2] * T3)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3]), dT1(y[0], y[1], y[2], y[3]), dT2(y[0], y[1], y[2], y[3]),
                    dT3(y[0], y[1], y[2], y[3])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2]]
    elif len(m) == 4:
        dE = lambda E, T1, T2, T3, T4: sigma - mu * E + E * (m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4)
        dT1 = lambda E, T1, T2, T3, T4: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[3] * T4)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4]), dT1(y[0], y[1], y[2], y[3], y[4]),
                    dT2(y[0], y[1], y[2], y[3], y[4]), dT3(y[0], y[1], y[2], y[3], y[4]),
                    dT4(y[0], y[1], y[2], y[3], y[4])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3]]
    elif len(m) == 5:
        dE = lambda E, T1, T2, T3, T4, T5: sigma - mu * E + E * (
                    m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5)
        dT1 = lambda E, T1, T2, T3, T4, T5: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[4] * T5)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
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
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5]]
    elif len(m) == 7:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[6] * T7)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6]]
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
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
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
    elif len(m) == 9:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7 + m[7] * T8 + m[
            8] * T9)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[6] * T7)
        dT8 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[7] * T8)
        dT9 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[8] * T9)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6], init_vals[7], init_vals[8]]
    elif len(m) == 10:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7 + m[7] * T8 + m[
            8] * T9 + m[9] * T10)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[6] * T7)
        dT8 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[7] * T8)
        dT9 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[8] * T9)
        dT10 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[9] * T10)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6], init_vals[7], init_vals[8], init_vals[9]]
    elif len(m) == 11:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7 + m[7] * T8 +
                m[8] * T9 + m[9] * T10 + m[10] * T11)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[6] * T7)
        dT8 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[7] * T8)
        dT9 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[8] * T9)
        dT10 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[9] * T10)
        dT11 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T11 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[10] * T11)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT11(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6], init_vals[7], init_vals[8], init_vals[9], init_vals[10]]
    else:
        print('Too many subclones for current iteration of code. Moving to next tree.')
        print('Number of subclones: ' + str(len(m)))
    return odefun, y0

def define_odefun_monoclonal(m, k, sigma, mu, a, b_tumor, init_vals):
    if len(m) == 1:  # only one clonal pop to keep track of
        dE = lambda E, T: sigma - mu * E + m[0] * E * T
        dT = lambda E, T: a * T * (1 - b_tumor * T) - k[0] * E * T

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if any pop less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE(y[0], y[1]), dT(y[0], y[1])]

    elif len(m) == 2:
        dE1 = lambda E1, E2, T1, T2: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, T1, T2: sigma - mu * E2 + E2 * m[1] * T2
        dT1 = lambda E1, E2, T1, T2: a * T1 * (1 - b_tumor * (T1 + T2)) - E1 * k[0] * T1
        dT2 = lambda E1, E2, T1, T2: a * T2 * (1 - b_tumor * (T1 + T2)) - E2 * k[1] * T2

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if any pop less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3]), dE2(y[0], y[1], y[2], y[3]),
                    dT1(y[0], y[1], y[2], y[3]), dT2(y[0], y[1], y[2], y[3])]

    elif len(m) == 3:
        dE1 = lambda E1, E2, E3, T1, T2, T3: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, T1, T2, T3: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, T1, T2, T3: sigma - mu * E3 + E3 * m[2] * T3
        dT1 = lambda E1, E2, E3, T1, T2, T3: a * T1 * (1 - b_tumor * (T1 + T2 + T3)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, T1, T2, T3: a * T2 * (1 - b_tumor * (T1 + T2 + T3)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, T1, T2, T3: a * T3 * (1 - b_tumor * (T1 + T2 + T3)) - E3 * (k[2] * T3)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if any pop less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5])]

    elif len(m) == 4:
        dE1 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E4 + E4 * m[3] * T4
        dT1 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E4 * (k[3] * T4)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])]

    elif len(m) == 5:
        dE1 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E5 + E5 * m[4] * T5
        dT1 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E5 * (k[4] * T5)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9])]

    elif len(m) == 6:
        dE1 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E6 + E6 * m[5] * T6
        dT1 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E6 * (k[5] * T6)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11])]

    elif len(m) == 7:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E6 + E6 * m[5] * T6
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E7 + E7 * m[6] * T7
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T7 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E7 * (k[6] * T7)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13])]

    elif len(m) == 8:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E6 + E6 * m[5] * T6
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E7 + E7 * m[6] * T7
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E8 + E8 * m[7] * T8
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E8 * (k[7] * T8)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15])]

    elif len(m) == 9:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E6 + E6 * m[5] * T6
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E7 + E7 * m[6] * T7
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E8 + E8 * m[7] * T8
        dE9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E9 + E9 * m[8] * T9
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E8 * (k[7] * T8)
        dT9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E9 * (k[8] * T9)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17])]

    elif len(m) == 10:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E1 + E1 * m[0] * T1)
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E2 + E2 * m[1] * T2)
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E3 + E3 * m[2] * T3)
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E4 + E4 * m[3] * T4)
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E5 + E5 * m[4] * T5)
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E6 + E6 * m[5] * T6)
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E7 + E7 * m[6] * T7)
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E8 + E8 * m[7] * T8)
        dE9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E9 + E9 * m[8] * T9)
        dE10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E10 + E10 * m[9] * T10)
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E8 * (k[7] * T8)
        dT9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E9 * (k[8] * T9)
        dT10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E10 * (k[9] * T10)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                       y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                         y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19])]

    elif len(m) == 11:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E1 + E1 * m[0] * T1)
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E2 + E2 * m[1] * T2)
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E3 + E3 * m[2] * T3)
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E4 + E4 * m[3] * T4)
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E5 + E5 * m[4] * T5)
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E6 + E6 * m[5] * T6)
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E7 + E7 * m[6] * T7)
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E8 + E8 * m[7] * T8)
        dE9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E9 + E9 * m[8] * T9)
        dE10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E10 + E10 * m[9] * T10)
        dE11 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E11 + E11 * m[10] * T11)
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E8 * (k[7] * T8)
        dT9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E9 * (k[8] * T9)
        dT10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E10 * (k[9] * T10)
        dT11 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T11 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E11 * (k[10] * T11)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                       y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE11(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                         y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT11(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                         y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21])]

    else:
        print('Too many subclones for current iteration of code. Moving to next tree.')
        print('Number of subclones: ' + str(len(m)))
    y0 = list(np.ones(len(init_vals)) / len(init_vals) * (sigma / mu)) + list(init_vals) # equal proportions at start of therapy
    return odefun, y0


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
        # What if there is neither surviving parental population OR a clonal neoantigen?
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
            print("W_i > 1, which is never supposed to happen. Double-check how you compute p_ij and W_i.")
        # now do the same for the children:
        for j in node.children:  # Recurse over children.
            normalization_sum, balance_sum = getSubtreeIndex(j, normalization_sum, balance_sum)
    return normalization_sum, balance_sum

def getResponse(tumor_size_over_time, t):
    sum_lesion_diams = [estimate_lesion_diameter(tumor_size_over_time[iii]) for iii in range(len(t))]
    try:
        start_sum_lesion_diameters = sum_lesion_diams[0]
    except IndexError:
        pass
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

def simulateTherapy(m_mult, k_mult, AxR, maxruns=500, model_type='polyclonal'):
    # Permute AxR data (with seed) for later use
    np.random.seed(757)
    AxR = np.random.permutation(AxR)
    AxR_ind = 0
    fails = 0
    dist_mean = 0.0113  # that's our b-d per day
    dist_var = (((0.00027 - 0.0113) / (-3) + (0.039 - 0.0113) / 3) / 2) ** 2
    lognorm_sigma = np.sqrt(np.log(dist_var / (dist_mean ** 2) + 1))
    lognorm_mu = np.log(dist_mean) - lognorm_sigma ** 2 / 2

    # Define parameters for trees
    path_base = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_oct_24/'
    path_addenda = ['MMRD',
                    'MMRP']
    all_deltas = [0.955, 0.955]
    ms_stat_list = ['MSI', 'MSS']
    speed_list = ['fast', 'fast']

    # Define parameters for tumor growth, model
    sigma = 10  # cells produced per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
    mu = 1e-2  # per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
    t0 = 0
    # a =   # (0.25 - 0.2387)*100  # previously was 1. b-d per day * 100 days DRAW RANDOMLY; SET A LATER
    a_rng = np.random.default_rng(757)
    b_tumor = 1e-7  # dimesions: 1 / cells
    m_proportionality_constant = m_mult  # removed nondimensionalization factors from m, k prop constants (Gamma)
    k_proportionality_constant = k_mult

    #model_type = 'monoclonal' # model type of T cell responses
    # Set up outcome variables
    MSI_best_resp = {"PD": 0, "SD": 0, "PR": 0, "CR": 0}
    MSI_month36_resp = {"progression": 0, "PFS": 0}
    MSI_ttp = []

    # Load in trees and begin work
    for i in range(len(all_deltas)):
        ms_stat = ms_stat_list[i]
        if ms_stat=='MSS':
            break  # UNCOMMENT IF YOU WANT TO WORK WITH MSS AS WELL
        path = path_base + path_addenda[i]
        speed = speed_list[i]
        tree_dicts = [f for f in os.listdir(path) if f.startswith('dilltree_' + speed + '_' + ms_stat)]
        kk = 0
        # Load dilled trees from data_gen script
        for dilled_tree_dict in tree_dicts:
            kk += 1
            if kk > maxruns:
                break
            # Unpack and process
            # try:
            tree_dict = dill.load(open(path + '/' + dilled_tree_dict, "rb"))
            # except EOFError:
            #     EOFflag += 1
            #     print('EOF Skip, kk = ' + '%0.f' % kk)
            #     continue
            # Convert dilled tree into Population object
            tree = unpack_root_dict_to_Population(tree_dict)

            # Sample tumor growth rate from lognormal distribution
            a = a_rng.lognormal(mean=lognorm_mu, sigma=lognorm_sigma)  # sample a from distribution of doubling times for CRC. true mean is -6.448, what we were using: -4.483
            #k_proportionality_constant = 0.5 * (a / sigma) * (
            #    k_mult) / 100  # 1.5 to increase by 50% relative to original Kamran value

            # Get clonal structure matrix
            newNewMat, num_subclones_defunct, is_there_a_clonal_neoant = get_subclones(path, tree, kk, ms_stat, min_size=10000, save_df=False)

            # Assign m,k based on AxR data
            m, k, AxR_ind = assign_k_and_m(AxR, AxR_ind, newNewMat, num_subclones_defunct, b_tumor, mu, a, sigma, m_proportionality_constant, k_proportionality_constant)

            # Find initial value populations
            init_vals, m_trunc = assign_ICs(newNewMat)

            # If necessary (parental/founder population has 0 pop), truncate m and k to remove that from consideration
            m = m[m_trunc:]
            k = k[m_trunc:]

            # Group together uniquely evolving populations based on (m,k) uniqueness
            unique_m = list(set(m))  # use set only for # of unique values, and not for anything else!!!
            unique_k = list(set(k))
            if len(unique_m) != len(unique_k):
                print('The number of (m,k) pairs is strange -- check this.')
            if len(unique_m) != len(m): # if we have overlapping populations, reduce subclonal structure for final input.
                adj_IC = []
                adj_m = []
                adj_k = []
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
                    ii += 1
                init_vals = adj_IC
                m = adj_m
                k = adj_k
                if m[0] != adj_m[0]:
                    print('Something went wrong with reducing subclonal populations by (m,k) pairs; m got scrambled.')

                # Trim off any size-zero subclones before running
                adj_IC = []
                adj_m = []
                adj_k = []
                for ii in range(len(m)):
                    if init_vals[ii] == 0:
                        continue # skip this one!
                    else:
                        adj_IC.append(init_vals[ii])
                        adj_m.append(m[ii])
                        adj_k.append(k[ii])
                if sum(adj_IC)!= 1e5:
                    print('Error! Sum of initial conditions is not correct')
                init_vals = adj_IC
                m = adj_m
                k = adj_k
                num_subclones_defunct = len(m)  # adjust number of effective subclones

            if len(init_vals) != len(m):
                print('ICs have different length than the number of (m,k) pairs.')
            # if len(m) != num_subclones_defunct:
            #      print('Was there a population with size 0, to which descended pops were added during initial value adjustment?')
            num_subclones = len(m)  # num_subclones needs to be set to len(m) after doing all init value adjustments
            # Set up ODEfun based on number of subclones that are of the right size

            if model_type == 'polyclonal':
                odefun, y0 = define_odefun_polyclonal(m, k, sigma, mu, a, b_tumor, init_vals)
            elif model_type == 'monoclonal':
                odefun, y0 = define_odefun_monoclonal(m, k, sigma, mu, a, b_tumor, init_vals)

            # Solve ODE using solve_IVP
            # scale_odefun = lambda t, y: [0.01*x for x in odefun(t, y)]
            Tmax = round(55 * 30.437) # le et al. 2023, 56 month follow up * 30.436 days/month
            dt = 1  # "sample" every 1 day.
            t_span = [t0, Tmax]
            t_eval = np.arange(t0, Tmax+dt, dt)
            sol = solve_ivp(odefun, t_span, y0, t_eval=t_eval)
            t = sol.t
            if len(t) == 1: # This would assume that solve_IVP was unable to integrate.
                print(sol.message)
                print('m: ' + str(m))
                print('k: ' + str(k))
                fails += 1
                continue
            if model_type == 'polyclonal':
                subclone_sol = sol.y[1:, ]
            elif model_type == 'monoclonal':
                subclone_sol = sol.y[len(m):, ]
            TMB_over_time = [sum(subclone_sol[:, iii]) for iii in range(len(t))]


            if ms_stat=='MSI':
                # Check response every 6 weeks
                #resp_t_inds = [0]  # indices for checking
                #resp_TMB_val_subset = [TMB_over_time[0]]  # values checked
                response_score_dict = {"PD": 0, "SD": 1, "PR": 2, "CR": 3}
                iw = 1
                best_response = 'PD'
                pdflag = 0
                time_to_PD = -1
                while round(12 * iw * 7 + 1) <= Tmax:
                    if iw == 1:
                        t_index = round(12 * 7 + 1)  # Check at 12 weeks for the very first time point
                    elif (iw * 9 * 7 + 1) > 365:  # past first year, check every 12 weeks.
                        t_index = round(12 * iw * 7 + 1)
                    else:
                        t_index = round(9 * iw * 7 + 1)
                    #resp_t_inds.append(t_index)  # append the indices for checking
                    #resp_TMB_val_subset.append(TMB_over_time[t_index])  # values checked
                    cur_resp = getResponse(TMB_over_time[:t_index], t[:t_index])
                    best_resp_score = max(response_score_dict[best_response], response_score_dict[cur_resp])
                    if best_resp_score == 3:
                        best_response = 'CR'
                    elif best_resp_score == 2:
                        best_response = 'PR'
                    elif best_resp_score == 1:
                        best_response = 'SD'
                    elif best_resp_score == 0:
                        best_response = 'PD'
                    else:
                        print('Something went wrong with finding best response over time.')
                    if iw == 1:
                        response_12w = cur_resp
                    if cur_resp == 'PD' and pdflag == False:  # once it's PD, no use in checking further ***
                        time_to_PD = t[t_index]
                        pdflag = True
                        response_tmax = cur_resp  # terminate therapy and use this as response_tmax
                        break
                    iw += 1
                if pdflag == False:
                    # Finally, check exactly at Tmax, if PD not achieved yet
                    t_index = Tmax
                    #resp_t_inds.append(t_index)  # append the indices for checking
                    #resp_TMB_val_subset.append(TMB_over_time[t_index])  # values checked
                    response_tmax = getResponse(TMB_over_time[:t_index], t[:t_index])
                    cur_resp = response_tmax
                    if pdflag == False:  # if not progressed yet, check if this could be our best score yet.
                        best_resp_score = max(response_score_dict[best_response], response_score_dict[cur_resp])
                        if best_resp_score == 3:
                            best_response = 'CR'
                        elif best_resp_score == 2:
                            best_response = 'PR'
                        elif best_resp_score == 1:
                            best_response = 'SD'
                        elif best_resp_score == 0:
                            best_response = 'PD'
                        time_to_PD = t[-1]  # if not progressed, set PFS to tmax

                MSI_best_resp[best_response] += 1
                #resp_to_therapy = True     # recall: MSI_month36_resp = {"progression": 0, "PFS": 0}
                if time_to_PD <= round(36*30.437 + 1): # if the time to progression is <= 24 months,
                    MSI_month36_resp["progression"] += 1 # then code this as progression before 36 months.
                else: # if time to PD > 36 months, then we code this as PFS for 36 months.
                    MSI_month36_resp["PFS"] += 1

                # for the MSI tumors, check at 36 months to use Le et al. 2023 data
                MSI_ttp.append(time_to_PD) # add time to PD
    # Fractional responses overall:
    #PEMBRO keynote-177 final analysis, order {"PD", "SD", "PR", "CR"}
    MSI_n_vec = np.array([45, 30, 49, 20]) # keynote-177 values
    MSI_perc_vec = MSI_n_vec/sum(MSI_n_vec) * 100
    MSI_sim_n_vec = np.array(list(MSI_best_resp.values()))
    MSI_sim_perc_vec = MSI_sim_n_vec / sum(MSI_sim_n_vec) * 100

    # error bars using clopper-pearson
    from scipy.stats import binomtest
    MSI_yerrormin = []
    MSI_yerrmax = []
    MSI_errbar_magnitude = []
    for ci_i in range(len(MSI_n_vec)):
        ci = binomtest(MSI_n_vec[ci_i], sum(MSI_n_vec), MSI_perc_vec[ci_i]/100).proportion_ci()  # Use this line to compute the 95% CI for the proportions. Clopper-Pearson.
        MSI_yerrormin.append(ci.low * 100)
        MSI_yerrmax.append(ci.high * 100)
        MSI_errbar_magnitude.append(ci.high - ci.low) # scale by error bar size in clinical data

    overall_weights = getWeights_overall(MSI_perc_vec, rel_err=MSI_errbar_magnitude)

    # Long-term response metrics:
    #month36resp = np.array(list(MSI_month36_resp.values()))
    #expected_month36_pfs = 34.1  # hard-coded for Le et al 2023 result; 34.1% cohort b pfs at
    #actual_month36_pfs = (month36resp[1])/sum(month36resp) * 100 # these may not be long enough.
    month36resp = np.array(list(MSI_month36_resp.values()))
    expected_month36_pfs = 42.3  # hard-coded keynote-177
    actual_month36_pfs = (month36resp[1])/sum(month36resp) * 100 # these may not be long enough.
    expected_median_pfs = 16.5 * 30.437 # 4.1 * 30.437  # units DAYS, hard-coded for Le et al 2023 result: 14.3 months * 30.4 days/month
    actual_median_pfs = np.median(MSI_ttp)

    loss = 0.1 * abs(expected_month36_pfs - actual_month36_pfs) + 0.1 * abs(expected_median_pfs - actual_median_pfs)
    for wi in range(len(overall_weights)):
        ovrl_diffnce = abs(MSI_perc_vec[wi] - MSI_sim_perc_vec[wi]) # compute |difference between expected and actual results overall|
        loss += overall_weights[wi]*ovrl_diffnce # + wk12_weights[wi]*wk12_diffnce

    print('Simulation complete for m = ' + str(m_mult) + ', k = ' + str(k_mult) + '; loss = ' + str(loss))
    print('Model type: ' + model_type)
    print('Sim percentages: ' + str(MSI_sim_perc_vec))
    print('36-month PFS: ' + str(actual_month36_pfs) + '%')
    print('median PFS: ' + str(actual_median_pfs/30.437) + ' months')
    # print('IVP solution failure rate: ' + str(fails/maxruns * 100) + '%')
    return loss # for this value of (m_mult, k_mult), return the normed difference between the desired response percentages and the actual ones.

# Functions for simulate_therapy_manual_ICs:
def get_subclones(path, tree, treenum, min_size=10000, save_df=False, ms_stat='MSI'):
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
    m_0 = 0 #1e-10 #(b_tumor / mu) / 100000  # baseline clonal m, no mut
    k_0 = 0 #1e-10 #(a / sigma) / 1000  # baseline clonal k, no mut
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

def getResponse(tumor_size_over_time, t):
    sum_lesion_diams = [estimate_lesion_diameter(tumor_size_over_time[iii]) for iii in range(len(t))]
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

def define_odefun_polyclonal(m, k, sigma, mu, a, b_tumor, init_vals):
    if len(m) == 1:  # only one clonal pop to keep track of
        dE = lambda E, T: sigma - mu * E + m[0] * E * T
        dT = lambda E, T: a * T * (1 - b_tumor * T) - k[0] * E * T

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1]), dT(y[0], y[1])]

        y0 = [sigma / mu, 1e5]
    elif len(m) == 2:
        dE = lambda E, T1, T2: sigma - mu * E + E * (m[0] * T1 + m[1] * T2)
        dT1 = lambda E, T1, T2: a * T1 * (1 - b_tumor * (T1 + T2)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2: a * T2 * (1 - b_tumor * (T1 + T2)) - E * (k[1] * T2)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2]), dT1(y[0], y[1], y[2]), dT2(y[0], y[1], y[2])]

        y0 = [sigma / mu, init_vals[0], init_vals[1]]
    elif len(m) == 3:
        dE = lambda E, T1, T2, T3: sigma - mu * E + E * (m[0] * T1 + m[1] * T2 + m[2] * T3)
        dT1 = lambda E, T1, T2, T3: a * T1 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3: a * T2 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3: a * T3 * (1 - b_tumor * (T1 + T2 + T3)) - E * (k[2] * T3)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3]), dT1(y[0], y[1], y[2], y[3]), dT2(y[0], y[1], y[2], y[3]),
                    dT3(y[0], y[1], y[2], y[3])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2]]
    elif len(m) == 4:
        dE = lambda E, T1, T2, T3, T4: sigma - mu * E + E * (m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4)
        dT1 = lambda E, T1, T2, T3, T4: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E * (k[3] * T4)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4]), dT1(y[0], y[1], y[2], y[3], y[4]),
                    dT2(y[0], y[1], y[2], y[3], y[4]), dT3(y[0], y[1], y[2], y[3], y[4]),
                    dT4(y[0], y[1], y[2], y[3], y[4])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3]]
    elif len(m) == 5:
        dE = lambda E, T1, T2, T3, T4, T5: sigma - mu * E + E * (
                    m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5)
        dT1 = lambda E, T1, T2, T3, T4, T5: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E * (k[4] * T5)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
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
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6]), dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5]]
    elif len(m) == 7:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                    k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (
                k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E * (k[6] * T7)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6]]
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
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
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
    elif len(m) == 9:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7 + m[7] * T8 + m[
            8] * T9)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[6] * T7)
        dT8 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[7] * T8)
        dT9 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E * (k[8] * T9)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6], init_vals[7], init_vals[8]]
    elif len(m) == 10:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7 + m[7] * T8 + m[
            8] * T9 + m[9] * T10)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[6] * T7)
        dT8 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[7] * T8)
        dT9 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[8] * T9)
        dT10 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E * (k[9] * T10)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6], init_vals[7], init_vals[8], init_vals[9]]
    elif len(m) == 11:
        dE = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: sigma - mu * E + E * (
                m[0] * T1 + m[1] * T2 + m[2] * T3 + m[3] * T4 + m[4] * T5 + m[5] * T6 + m[6] * T7 + m[7] * T8 +
                m[8] * T9 + m[9] * T10 + m[10] * T11)
        dT1 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[0] * T1)
        dT2 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[1] * T2)
        dT3 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[2] * T3)
        dT4 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[3] * T4)
        dT5 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[4] * T5)
        dT6 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[5] * T6)
        dT7 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[6] * T7)
        dT8 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[7] * T8)
        dT9 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[8] * T9)
        dT10 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[9] * T10)
        dT11 = lambda E, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T11 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E * (k[10] * T11)

        def odefun(t, y):
            for mm in range(len(y[1:])):
                if y[mm + 1] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm + 1] = 0
            return [dE(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT11(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11])]

        y0 = [sigma / mu, init_vals[0], init_vals[1], init_vals[2], init_vals[3], init_vals[4], init_vals[5],
              init_vals[6], init_vals[7], init_vals[8], init_vals[9], init_vals[10]]
    else:
        print('Too many subclones for current iteration of code. Moving to next tree.')
        print('Number of subclones: ' + str(len(m)))
    return odefun, y0

def define_odefun_monoclonal(m, k, sigma, mu, a, b_tumor, init_vals):
    if len(m) == 1:  # only one clonal pop to keep track of
        dE = lambda E, T: sigma - mu * E + m[0] * E * T
        dT = lambda E, T: a * T * (1 - b_tumor * T) - k[0] * E * T

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if any pop less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE(y[0], y[1]), dT(y[0], y[1])]

    elif len(m) == 2:
        dE1 = lambda E1, E2, T1, T2: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, T1, T2: sigma - mu * E2 + E2 * m[1] * T2
        dT1 = lambda E1, E2, T1, T2: a * T1 * (1 - b_tumor * (T1 + T2)) - E1 * k[0] * T1
        dT2 = lambda E1, E2, T1, T2: a * T2 * (1 - b_tumor * (T1 + T2)) - E2 * k[1] * T2

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if any pop less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3]), dE2(y[0], y[1], y[2], y[3]),
                    dT1(y[0], y[1], y[2], y[3]), dT2(y[0], y[1], y[2], y[3])]

    elif len(m) == 3:
        dE1 = lambda E1, E2, E3, T1, T2, T3: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, T1, T2, T3: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, T1, T2, T3: sigma - mu * E3 + E3 * m[2] * T3
        dT1 = lambda E1, E2, E3, T1, T2, T3: a * T1 * (1 - b_tumor * (T1 + T2 + T3)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, T1, T2, T3: a * T2 * (1 - b_tumor * (T1 + T2 + T3)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, T1, T2, T3: a * T3 * (1 - b_tumor * (T1 + T2 + T3)) - E3 * (k[2] * T3)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if any pop less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5])]

    elif len(m) == 4:
        dE1 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, T1, T2, T3, T4: sigma - mu * E4 + E4 * m[3] * T4
        dT1 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, T1, T2, T3, T4: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4)) - E4 * (k[3] * T4)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])]

    elif len(m) == 5:
        dE1 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: sigma - mu * E5 + E5 * m[4] * T5
        dT1 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, T1, T2, T3, T4, T5: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5)) - E5 * (k[4] * T5)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9])]

    elif len(m) == 6:
        dE1 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: sigma - mu * E6 + E6 * m[5] * T6
        dT1 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, T1, T2, T3, T4, T5, T6: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6)) - E6 * (k[5] * T6)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11])]

    elif len(m) == 7:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E6 + E6 * m[5] * T6
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: sigma - mu * E7 + E7 * m[6] * T7
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T1 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T2 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T3 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T4 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T5 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T6 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, T1, T2, T3, T4, T5, T6, T7: a * T7 * (1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)) - E7 * (k[6] * T7)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13])]

    elif len(m) == 8:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E6 + E6 * m[5] * T6
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E7 + E7 * m[6] * T7
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: sigma - mu * E8 + E8 * m[7] * T8
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, T1, T2, T3, T4, T5, T6, T7, T8: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8)) - E8 * (k[7] * T8)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14],
                        y[15])]

    elif len(m) == 9:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E1 + E1 * m[0] * T1
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E2 + E2 * m[1] * T2
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E3 + E3 * m[2] * T3
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E4 + E4 * m[3] * T4
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E5 + E5 * m[4] * T5
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E6 + E6 * m[5] * T6
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E7 + E7 * m[6] * T7
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E8 + E8 * m[7] * T8
        dE9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: sigma - mu * E9 + E9 * m[8] * T9
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E8 * (k[7] * T8)
        dT9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, T1, T2, T3, T4, T5, T6, T7, T8, T9: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9)) - E9 * (k[8] * T9)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dE9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9],
                        y[10], y[11], y[12], y[13], y[14], y[15], y[16], y[17])]

    elif len(m) == 10:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E1 + E1 * m[0] * T1)
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E2 + E2 * m[1] * T2)
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E3 + E3 * m[2] * T3)
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E4 + E4 * m[3] * T4)
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E5 + E5 * m[4] * T5)
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E6 + E6 * m[5] * T6)
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E7 + E7 * m[6] * T7)
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E8 + E8 * m[7] * T8)
        dE9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E9 + E9 * m[8] * T9)
        dE10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: (
                sigma - mu * E10 + E10 * m[9] * T10)
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E8 * (k[7] * T8)
        dT9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E9 * (k[8] * T9)
        dT10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10)) - E10 * (k[9] * T10)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                       y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dE10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                        y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10],
                         y[11], y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19])]

    elif len(m) == 11:
        dE1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E1 + E1 * m[0] * T1)
        dE2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E2 + E2 * m[1] * T2)
        dE3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E3 + E3 * m[2] * T3)
        dE4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E4 + E4 * m[3] * T4)
        dE5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E5 + E5 * m[4] * T5)
        dE6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E6 + E6 * m[5] * T6)
        dE7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E7 + E7 * m[6] * T7)
        dE8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E8 + E8 * m[7] * T8)
        dE9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E9 + E9 * m[8] * T9)
        dE10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E10 + E10 * m[9] * T10)
        dE11 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: (
                sigma - mu * E11 + E11 * m[10] * T11)
        dT1 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T1 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E1 * (k[0] * T1)
        dT2 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T2 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E2 * (k[1] * T2)
        dT3 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T3 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E3 * (k[2] * T3)
        dT4 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T4 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E4 * (k[3] * T4)
        dT5 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T5 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E5 * (k[4] * T5)
        dT6 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T6 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E6 * (k[5] * T6)
        dT7 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T7 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E7 * (k[6] * T7)
        dT8 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T8 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E8 * (k[7] * T8)
        dT9 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T9 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E9 * (k[8] * T9)
        dT10 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T10 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E10 * (k[9] * T10)
        dT11 = lambda E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11: a * T11 * (
                1 - b_tumor * (T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9 + T10 + T11)) - E11 * (k[10] * T11)

        def odefun(t, y):
            for mm in range(int(len(y)/2), len(y)):
                if y[mm] < 1e-2:  # if the tumor is less than 1 cell, then set this pop to 0.
                    y[mm] = 0
            return [dE1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                       y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dE11(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT1(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT2(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT3(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT4(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT5(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT6(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT7(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT8(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT9(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                        y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT10(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                         y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21]),
                    dT11(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11],
                         y[12], y[13], y[14], y[15], y[16], y[17], y[18], y[19], y[20], y[21])]

    else:
        print('Too many subclones for current iteration of code. Moving to next tree.')
        print('Number of subclones: ' + str(len(m)))
    y0 = list(np.ones(len(init_vals)) / len(init_vals) * (sigma / mu)) + list(init_vals) # equal proportions at start of therapy
    return odefun, y0

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


def simulate_immunotherapy(
        make_plots = True,
        all_deltas = [0.955, 0.955],
        ms_stat_list = ['MMRD', 'MMRP'],
        model_type = 'monoclonal',
        path_to_ARdata="./AxR_data.txt",
        path_to_MMRD_data="./MMRD/",
        path_to_MMRP_data="./MMRP/",
        path_base="./",
        debug = False):
    # Load in AxR data for later use
    AxR = []
    with open(path_to_ARdata, 'r') as f:
        content = f.readlines()
        for i in content:
            num = float(''.join(list(i)[:-1]))
            if num >= 1:
                AxR.append(num)

    # Permute AxR data (with seed) for later use
    np.random.seed(757)
    AxR = np.random.permutation(AxR)
    AxR_ind = 0

    # Define parameters for trees
    paths_to_tumordata = [path_to_MMRD_data, path_to_MMRP_data]  # ORDER IS VERY IMPORTANT!
    speed_list = ['fast', 'fast']  # Speed of branching process (hard-coded)
    b = 0.25  # Birth rate for branching process (hard-coded)
    maxruns = 5000

    # Find or make data folder for therapydata:
    path_data = path_base + 'data/'
    os.makedirs(path_data, exist_ok=True)

    # Simulate immunotherapy and collect response statistics
    try:
        if debug == True:
            raise Exception('Manual re-simulation for debugging purposes.')
        therapydata = pd.read_pickle(
            open(path_data + model_type + "_therapydata_pandas_df.dump", 'rb'))
        print('Immunotherapy dataset with these parameters already created. Loading...')
        return therapydata
    except:
        print('Immunotherapy dataset with these parameters not already created. Creating...')
        # Define parameters for tumor growth, model
        sigma = 10  # cells produced per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
        mu = 1e-2  # per day, per Garcia, Bonhoeffer, Fu 2020 (more ref there)
        t0 = 0
        b_tumor = 1e-7  # dimesions: 1 / cells
        m_proportionality_constant = 1.757e-7
        k_proportionality_constant = 3.419e-7
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
            speed = speed_list[i]
            tree_dicts = [f for f in os.listdir(path) if f.startswith('dilltree_' + ms_stat)]
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
                a = a_rng.lognormal(mean=-4.483,
                                    sigma=0.828)  # sample a from distribution of doubling times for CRC. true mean is -6.448, what we were using: -4.483

                # Get clonal structure matrix
                newNewMat, num_subclones, is_there_a_clonal_neoant = get_subclones(path, tree, kk, min_size=10000,
                                                                                   save_df=False)  # save_df=True) #save df if you changed get_sublcones function
                totalTMB = getTMB(tree, depth=0, TMB=0)
                TMB_1perc = getTMB_threshold(tree, depth=0, TMB=0, threshold=0.01 * 1e5)
                TMB_10perc = getTMB_threshold(tree, depth=0, TMB=0, threshold=0.10 * 1e5)
                orig_num_subclones = np.copy(num_subclones)
                if orig_num_subclones > 1:
                    max_NA_quality = max(AxR[AxR_ind:(AxR_ind + orig_num_subclones)])
                else:
                    max_NA_quality = 0

                # Assign m,k based on AxR data
                m, k, AxR_vals, new_AxR_ind = assign_k_and_m(AxR, AxR_ind, newNewMat, num_subclones, b_tumor, mu, a,
                                                             sigma, m_proportionality_constant,
                                                             k_proportionality_constant)

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
                if len(unique_m) != len(
                        m):  # if we have overlapping populations, reduce subclonal structure for final input.
                    adj_IC = []
                    adj_m = []
                    adj_k = []
                    adj_AxR = []
                    ii = 0
                    skip = 0
                    while ii < (
                            len(unique_m) + skip):  # for each m, preserving order, and not exceeding total number of unique m values!!!
                        if m[ii] in adj_m:  # if we already covered this one,
                            skip += 1
                            ii += 1
                            continue
                        inds = np.array(
                            [jj for jj in range(len(m)) if m[jj] == m[ii]])  # find indices of all pops with same m
                        totpop = sum(np.array(init_vals)[inds])  # add together those populations
                        adj_IC.append(totpop)  # store this into the adjusted IC list (effective subclone number)
                        adj_m.append(m[ii])  # store this m into the adjusted m list (no repeats)
                        adj_k.append(k[ii])  # preserves the ordering of original m and k
                        adj_AxR.append(AxR_vals[ii])
                        ii += 1
                    init_vals = adj_IC
                    m = adj_m
                    k = adj_k
                    AxR_vals = adj_AxR
                    if m[0] != adj_m[0]:
                        print(
                            'Something went wrong with reducing subclonal populations by (m,k) pairs; m got scrambled.')

                    # Trim off any size-zero subclones before running
                    adj_IC = []
                    adj_m = []
                    adj_k = []
                    adj_AxR = []
                    for ii in range(len(m)):
                        if init_vals[ii] == 0:
                            continue  # skip this one!
                        else:
                            adj_IC.append(init_vals[ii])
                            adj_m.append(m[ii])
                            adj_k.append(k[ii])
                            adj_AxR.append(AxR_vals[ii])
                    if sum(adj_IC) != 1e5:
                        print('Error! Sum of initial conditions is not correct')
                    init_vals = adj_IC
                    m = adj_m
                    k = adj_k
                    AxR_vals = adj_AxR
                    num_subclones = len(m)  # adjust number of effective subclones
                check_zero_remnants_vec = [iiii > 0 for iiii in init_vals]
                if sum(check_zero_remnants_vec) < len(check_zero_remnants_vec):
                    m = [m[nn] for nn in range(len(init_vals)) if init_vals[nn] > 0]
                    k = [k[nn] for nn in range(len(init_vals)) if init_vals[nn] > 0]
                    AxR_vals = [AxR_vals[nn] for nn in range(len(init_vals)) if init_vals[nn] > 0]
                    init_vals = [init_vals[nn] for nn in range(len(init_vals)) if init_vals[nn] > 0]
                    if AxR_vals[0] == 0:
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
                Tmax = round(
                    55 * 30.437)  # x 100 to get to days. 84 days total = 12 weeks. Large Tmax = 5, small = 0.84 (12 weeks)
                dt = 1  # "sample" every 1 day.
                t_span = [t0, Tmax]
                t_eval = np.arange(t0, Tmax + dt, dt)
                sol = solve_ivp(odefun, t_span, y0, t_eval=t_eval)
                t = sol.t
                t_pseud_index = round((
                                                  12 * 7) / dt + 1)  # time to pseudoprogression is 5.7 weeks -- not 12 weeks. Add one for python weirdness
                if model_type == 'polyclonal':
                    subclone_sol = sol.y[1:, ]
                    effector_sol = sol.y[0,]
                elif model_type == 'monoclonal':
                    subclone_sol = sol.y[len(m):, ]
                    effector_sol = sol.y[0:len(m), ]

                TMB_over_time = [sum(subclone_sol[:, iii]) for iii in range(len(t))]
                effector_over_time = [sum(effector_sol[:, iii]) for iii in range(len(t))]
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
                while round(12 * iw * 7 + 1) <= Tmax:
                    if iw == 1:
                        t_index = round(12 * 7 + 1)  # Check at 12 weeks for the very first time point
                    elif (iw * 9 * 7 + 1) > 365:  # past first year, check every 12 weeks.
                        t_index = round(12 * iw * 7 + 1)
                    else:
                        t_index = round(9 * iw * 7 + 1)
                    # resp_t_inds.append(t_index)  # append the indices for checking
                    # resp_TMB_val_subset.append(TMB_over_time[t_index])  # values checked
                    cur_resp = getResponse(TMB_over_time[:t_index],
                                           t[:t_index])  # getResponse(resp_TMB_val_subset, resp_t_inds)
                    best_resp_score = min(response_score_dict[best_response], response_score_dict[cur_resp])
                    if best_resp_score == 0:
                        best_response = 'CR'
                    elif best_resp_score == 1:
                        best_response = 'PR'
                    elif best_resp_score == 2:
                        best_response = 'SD'
                    elif best_resp_score == 3:
                        best_response = 'PD'
                    else:
                        print('Something went wrong with finding best response over time.')
                    if iw == 1:
                        response_12w = cur_resp
                    if cur_resp == 'PD' and pdflag == False:  # once it's PD, no use in checking further ***
                        time_to_PD = t[t_index]
                        pdflag = True
                        response_tmax = cur_resp  # terminate therapy and use this as response_tmax
                        break
                    iw += 1
                if pdflag == False:
                    # Finally, check exactly at Tmax, if PD not achieved yet
                    t_index = Tmax
                    response_tmax = getResponse(TMB_over_time[:t_index],
                                                t[:t_index])  # getResponse(resp_TMB_val_subset, resp_t_inds)
                    cur_resp = response_tmax
                    if pdflag == False:  # if not progressed yet, check if this could be our best score yet.
                        best_resp_score = min(response_score_dict[best_response], response_score_dict[cur_resp])
                        if best_resp_score == 0:
                            best_response = 'CR'
                        elif best_resp_score == 1:
                            best_response = 'PR'
                        elif best_resp_score == 2:
                            best_response = 'SD'
                        elif best_resp_score == 3:
                            best_response = 'PD'
                        time_to_PD = t[-1]  # if not progressed, set PFS to tmax

                if time_to_PD > (24 * 7 + 1):  # compute DCR as in Le et al. 2023
                    diseaseControl = 1
                else:
                    diseaseControl = 0

                # Check immune-specific response criteria. Relate to PD.
                min_increment = 4 * 7
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
                if make_plots==True:
                    if kk == 1:
                        os.makedirs(path_base + 'tumor_plots/', exist_ok=True)
                    plt.rcParams.update({'font.size': 14})
                    plt.rcParams['font.family'] = ['Arial']
                    plt.rc('legend', fontsize=11)

                    plt.figure(figsize=(5, 4))
                    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
                    if num_subclones > 0:
                        if AxR_vals[0] == 0:
                            plt.rc('axes', prop_cycle=(
                                cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
                            plt.plot(t / 30.4368, subclone_sol.T / TMB_over_time[0])
                            if len(subclone_sol) > 1:
                                plt.plot(t / 30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)
                            plt.legend(
                                ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in
                                                     range(len(subclone_sol) - 1)] + [
                                    'Total TB'])
                        else:
                            plt.rc('axes', prop_cycle=(
                                cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
                            plt.plot(t / 30.4368, subclone_sol.T / TMB_over_time[0])
                            if len(subclone_sol) > 1:
                                plt.plot(t / 30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)
                            plt.legend(['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol))] + ['Total TB'])
                    else:
                        plt.plot(t / 30.4368, subclone_sol.T / TMB_over_time[0],
                                 'fuchsia')  # Fuchsia: non-neoantigen containing subclone (Founder)
                        plt.legend(['Total TB (no neoantigenic clones)'])
                    plt.xlabel('Months since start of therapy')
                    plt.ylabel('Relative tumor volume')
                    plt.xticks([0, 10, 20, 30, 40, 50])
                    plt.ylim([0, 3])  # max(2, max(TMB_over_time)/TMB_over_time[0])])
                    plt.title(LTR)
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_largeTmax_subclone_fig_' + str(kk) + '.png',
                                format='png')
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_largeTmax_subclone_fig_' + str(kk) + '.svg',
                                format='svg')

                    first_year_ind = int(365)
                    fig, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.set_xlabel('Days since start of therapy')
                    ax1.set_ylabel('Relative tumor volume')
                    ax1.set_ylim([0, 2])
                    ax1.plot(t[:first_year_ind], TMB_over_time[:first_year_ind] / TMB_over_time[0], 'k--', linewidth=2)

                    ax2 = ax1.twinx()
                    color = 'tab:blue'
                    ax2.plot(t[:first_year_ind], effector_over_time[:first_year_ind] / effector_over_time[0], color=color,
                             linewidth=2)
                    ax2.set_ylabel('Relative T cell population', color=color)
                    ax2.set_ylim([0, 100])
                    ax2.tick_params(axis='y', labelcolor=color)
                    plt.title(LTR)
                    fig.tight_layout()
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_smallTmax_effector_fig_' + str(kk) + '.png',
                                format='png')
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_smallTmax_effector_fig_' + str(kk) + '.svg',
                                format='svg')

                    fig, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.set_xlabel('Months since start of therapy')
                    ax1.set_ylabel('Relative tumor volume')
                    ax1.set_ylim([0, 2])
                    ax1.plot(t / 30.4368, TMB_over_time / TMB_over_time[0], 'k--', linewidth=2)

                    ax2 = ax1.twinx()
                    color = 'tab:blue'
                    ax2.plot(t / 30.4368, effector_over_time / effector_over_time[0], color=color, linewidth=2)
                    ax2.set_ylabel('Relative T cell population', color=color)
                    ax2.set_ylim([0, 100])
                    ax2.tick_params(axis='y', labelcolor=color)
                    plt.title(LTR)
                    fig.tight_layout()
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_largeTmax_effector_fig_' + str(kk) + '.png',
                                format='png')
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_largeTmax_effector_fig_' + str(kk) + '.svg',
                                format='svg')

                    plt.figure(figsize=(5, 4))
                    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.92)
                    if num_subclones > 0:
                        if AxR_vals[0] == 0:
                            plt.rc('axes', prop_cycle=(
                                cycler('color', ['fuchsia', 'dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
                            plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0])
                            if len(subclone_sol) > 1:
                                plt.plot(t[:t_pseud_index], TMB_over_time[:t_pseud_index] / TMB_over_time[0], 'k--',
                                         linewidth=2)
                                plt.legend(
                                    ['No neoantigen'] + ['Neoantigen ' + str(iii + 1) for iii in
                                                         range(len(subclone_sol) - 1)] + [
                                        'Total TB'])
                        else:
                            plt.rc('axes', prop_cycle=(
                                cycler('color', ['dodgerblue', 'cyan', 'mediumblue', 'mediumorchid', 'indigo'])))
                            plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0])
                            if len(subclone_sol) > 1:
                                plt.plot(t[:t_pseud_index], TMB_over_time[:t_pseud_index] / TMB_over_time[0], 'k--',
                                         linewidth=2)
                            plt.legend(['Neoantigen ' + str(iii + 1) for iii in range(len(subclone_sol))] + ['Total TB'])
                    else:
                        plt.plot(t[:t_pseud_index], subclone_sol.T[:t_pseud_index] / TMB_over_time[0],
                                 'fuchsia')  # Fuchsia: non-neoantigen containing subclone (Founder)
                        plt.legend(['Total TB (no neoantigenic clones)'])
                    plt.xlabel('Days since start of therapy')
                    plt.ylabel('Relative tumor volume')
                    plt.ylim([0, 2])
                    plt.title(response_label_dict[best_response])
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_smallTmax_subclone_fig_' + str(kk) + '.png',
                                format='png')
                    plt.savefig(path_base + 'tumor_plots/' + model_type + '_smallTmax_subclone_fig_' + str(kk) + '.svg',
                                format='svg')

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
                print('Simulated therapy on ' + ms_stat + ' tumor # ' + str(kk) + ' out of ' + str(
                    maxruns) + ' tumors; best response: ' + best_response)
                all_TMB_timeseries.append(TMB_timeseries)
                all_num_solved.append(num_solved)
        therapydata.index = np.arange(1, len(therapydata) + 1)
        dill.dump(therapydata,
                  open(path_data + model_type + "_therapydata_pandas_df.dump", 'wb'))
        print('Immunotherapy dataset with these parameters created. Saving to file...')
        print('Number of runs skipped due to nontrivial IC setup: ' + str(ICskip))
        return therapydata