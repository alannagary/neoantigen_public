from neoantigen_functions import *

### First, please set up your environment using the requirements.txt file in the repository.

### Next, please indicate the paths to the following files:
path_to_AxRdata = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/demo/AxR_data.txt' # Where is your AxR_data.txt file?
path_base = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/demo/' # Where do you want to run the demo and store the data, plots, etc.?


### Modify key features of the demo below:
num_tumors = 10 # How many of each type of tumors (equal numbers MMRD and MMRP) would you like to generate? Note: exact stochastic simulation is slow! Each takes about 10 minutes on my computer.
re_generate_tumors = False # If you've already generated some tumors, do you want a new set?
set_model_type = 'monoclonal' # Would you like to model each T cell population individually (monoclonal), or a polyclonal single population of T cells (not recommended)?
make_plots = True # Do you want to see plots of each MMRD simulated immunotherapy trajectory? (Note: plots are only produced for MMRD tumors by default)


### Generate tumors:
db_frac = 0.2387 / 0.25 # Ratio of death/birth rates; affects the rate of tumor growth.
tumor_types = ['MMRD', 'MMRP']
mut_rates = [9.8e-3, 1.1e-3] # Mutation rate. We use MMRD mutation rate of 9.8e-3, and MMRP mutation rate of 1.1e-3.
if re_generate_tumors:
    for j in range(len(tumor_types)):
        for i in range(num_tumors):
            generate_tree_simple(db_frac, mut_rates[j], task_id=i, path=path_base)
            print(tumor_types[j] + ' tumor #' + str(i) + ' generated and saved.')
else:
    print('Using pre-generated tumors.')


### Simulate immunotherapy:
therapydata = simulate_immunotherapy(
        make_plots = make_plots,
        all_deltas = [db_frac, db_frac],
        ms_stat_list = ['MMRD', 'MMRP'],
        model_type = set_model_type,
        path_to_ARdata = path_to_AxRdata,
        path_to_MMRD_data = path_base + 'data/',
        path_to_MMRP_data = path_base + 'data/',
        path_base = path_base,
        debug = False)
# Note: you should find your plots in the "tumor_plots" folder.

### Evaluate your dataset:
MMRDdata = therapydata[therapydata['ms_stat']=='MMRD']
MMRPdata = therapydata[therapydata['ms_stat']=='MMRP']

print('MMRD DCR = ' + str(sum(MMRDdata['diseaseControl'])/len(MMRDdata['diseaseControl'])))
print('MMRD ORR = ' + str((sum(MMRDdata['best_response']=='PR')+sum(MMRDdata['best_response']=='CR'))/len(MMRDdata['best_response'])))

ttpData = [i for i in MMRDdata['time_to_progression'] if i>0]
print('MMRD Median PFS = ' + str(np.median(ttpData)/30.4368))
print('MMRD 36-month PFS = ' + str(sum([i > 36*30.4368 for i in ttpData])/len(ttpData) * 100))

print('MMRD PD best responses: ' + str(sum(MMRDdata['best_response']=='PD')/len(MMRDdata['best_response']) * 100) + '%')
print('MMRD SD best responses: ' + str(sum(MMRDdata['best_response']=='SD')/len(MMRDdata['best_response']) * 100) + '%')
print('MMRD PR best responses: ' + str(sum(MMRDdata['best_response']=='PR')/len(MMRDdata['best_response']) * 100) + '%')
print('MMRD CR best responses: ' + str(sum(MMRDdata['best_response']=='CR')/len(MMRDdata['best_response']) * 100) + '%')

print('MMRD PD 12w responses: ' + str(sum(MMRDdata['response_12w']=='PD')/len(MMRDdata['response_12w']) * 100) + '%')
print('MMRD SD 12w responses: ' + str(sum(MMRDdata['response_12w']=='SD')/len(MMRDdata['response_12w']) * 100) + '%')
print('MMRD PR 12w responses: ' + str(sum(MMRDdata['response_12w']=='PR')/len(MMRDdata['response_12w']) * 100) + '%')
print('MMRD CR 12w responses: ' + str(sum(MMRDdata['response_12w']=='CR')/len(MMRDdata['response_12w']) * 100) + '%')

print('MMRP PD 12w responses: ' + str(sum(MMRPdata['response_12w']=='PD')/len(MMRPdata['response_12w']) * 100) + '%')
print('MMRP SD 12w responses: ' + str(sum(MMRPdata['response_12w']=='SD')/len(MMRPdata['response_12w']) * 100) + '%')
print('MMRP PR 12w responses: ' + str(sum(MMRPdata['response_12w']=='PR')/len(MMRPdata['response_12w']) * 100) + '%')
print('MMRP CR 12w responses: ' + str(sum(MMRPdata['response_12w']=='CR')/len(MMRPdata['response_12w']) * 100) + '%')

MMRD_DRs = MMRDdata[MMRDdata['LTR']=='Durable Response']
MMRD_ARs = MMRDdata[MMRDdata['LTR']=='Acquired Resistance']
MMRD_NRs = MMRDdata[MMRDdata['LTR']=='No Response']

print('MMRD DR: ' + str(sum(MMRDdata['LTR']=='Durable Response')/len(MMRDdata['best_response']) * 100) + '%')
print('MMRD AR: ' + str(sum(MMRDdata['LTR']=='Acquired Resistance')/len(MMRDdata['best_response']) * 100) + '%')
print('MMRD NR: ' + str(sum(MMRDdata['LTR']=='No Response')/len(MMRDdata['best_response']) * 100) + '%')

print('MMRP DR: ' + str(sum(MMRPdata['LTR']=='Durable Response')/len(MMRPdata['best_response']) * 100) + '%')
print('MMRP AR: ' + str(sum(MMRPdata['LTR']=='Acquired Resistance')/len(MMRPdata['best_response']) * 100) + '%')
print('MMRP NR: ' + str(sum(MMRPdata['LTR']=='No Response')/len(MMRPdata['best_response']) * 100) + '%')



