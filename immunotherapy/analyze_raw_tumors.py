from neoantigen_functions import *
import pandas as pd
import dill

if __name__ == "__main__":

    path_base = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_oct_24/'
    path_addenda = ['MMRD',
                    'MMRP']
    all_deltas = [0.955, 0.955]
    u_c_list = [0.0098, 0.0011]
    ms_stat_list = ['MSI', 'MSS']
    b = 0.25
    maxruns = 10000


    d = {'ms_stat': [], 'speed': [], 'numclonalmuts': [], 'totalnummuts': [], 'largestcloneperc': [],
         'numneoantaboveAlpha_0.1': [], 'numneoantaboveAlpha_0.25': [], 'unmutatedpop': []}
    simdata = pd.DataFrame(data=d)
    legent = []
    for i in range(len(all_deltas)):
        ms_stat = ms_stat_list[i]
        cur_path = path_base + path_addenda[i]
        delta = all_deltas[i]
        if np.isclose(delta, 0.955):
            speed = 'fast'
        elif np.isclose(delta, 0.977):
            speed = 'moderate'
        elif np.isclose(delta, 0.989):
            speed = 'slow'
        legent.append(ms_stat + ' ' + speed)
        phylos, unmut = listify_data(cur_path, ms_stat, speed, maxruns, excel_true=False)
        num_clonal_muts = []
        total_num_muts = []
        largest_clone_perc = []
        for k in range(len(phylos)):
            tumor = phylos[k]
            if len(tumor)==0:
                tumordf = pd.DataFrame({'ms_stat': [ms_stat], 'speed': [speed],
                                    'totalnummuts': [0],
                                    'numclonalmuts': [0],
                                    'largestcloneperc': [0],
                                    'numneoantaboveAlpha_0.1': [0],
                                    'numneoantaboveAlpha_0.25': [0],
                                    'unmutatedpop': [int(1e5)]})
            else:
                tumordf = pd.DataFrame({'ms_stat': [ms_stat], 'speed': [speed],
                                    'totalnummuts': [len(tumor)],
                                    'numclonalmuts': [sum([tumor[k] == 1e+5 for k in range(len(tumor))])],
                                    'largestcloneperc': [max(tumor)/(1e+5)*100],
                                    'numneoantaboveAlpha_0.1': len([i for i in tumor if i>=0.1*int(1e+5)]),
                                    'numneoantaboveAlpha_0.25': len([i for i in tumor if i>=0.25*int(1e+5)]),
                                    'unmutatedpop': [unmut[k]]})
            simdata = simdata.append(tumordf)


    dill.dump(simdata, open(path_base + "simdata_" + str(maxruns) + "_pandas_df.dump", 'wb'))

