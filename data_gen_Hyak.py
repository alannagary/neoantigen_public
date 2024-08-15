import numpy as np
import dill
import sys
import argparse

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


def generate_label(db_frac, mut_rate, task_id):
    if np.isclose(db_frac, 0.955):
        speed = 'fast'
    elif np.isclose(db_frac, 0.977):
        speed = 'moderate'
    elif np.isclose(db_frac, 0.989):
        speed = 'slow'
    else:
        raise ValueError('db_frac must have one of the values 0.955, 0.977, or 0.989 for labeling.')
    if mut_rate<0.002:
        sat_stab = 'MSS'
    else:
        sat_stab = 'MSI'
    lab = f"{speed}_{sat_stab}_{task_id}"
    return lab


def generate_tree(db_frac, mut_rate, task_id):
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
    tree_lab = generate_label(args.db_frac, args.mut_rate, args.task_id)
    dill.dump(root_dict,
              #open('../data/' + args.job_id + '/dilltree_' + tree_lab + '.dump', 'wb'))
              open(f"../data/{args.job_id}/dilltree_{tree_lab}.dump", 'wb'))
    return root, root_dict


sys.setrecursionlimit(10 ** 6)

if __name__ == "__main__":
    generate_tree(args.db_frac, args.mut_rate, args.task_id)
