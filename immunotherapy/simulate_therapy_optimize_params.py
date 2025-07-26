from scipy.optimize import fmin
from neoantigen_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import dill

# Information you may need to adjust:
path_to_ARdata = 'C:/Users/Alanna/Desktop/Research_Code/Desktop_research/AxR_data.txt' # Set your path to the AxR data file, AxR_data.txt
path_to_gridsearch = 'C:/Users/Alanna/Desktop/Research_Code/neoantigens/hyak_data/updated_code_oct_24/keynote_177/' # Set your path to the directory for your plots
regenerate_gridsearch = False # Do you want to re-generate the grid search for the initial value search? Warning: very slow if set to True!
regenerate_optimization = False # Do you want to re-optimize over 5000 MMR-D tumors? Warning: very slow if set to True!
model_type = 'monoclonal' # Acceptable: polyclonal or monoclonal. If polyclonal, you will need to regenerate gridsearch and optimization.

# Load in AxR data for later use
AxR = []
with open(path_to_ARdata, 'r') as f:
    content = f.readlines()
    for i in content:
        num = float(''.join(list(i)[:-1]))
        if num>=1:
            AxR.append(num)
# Plot neoantigen immunogenicity (Supp. Fig 4a)
axr_df = pd.DataFrame({'axr': AxR})
color_palette = ['orchid', 'purple']
sns.set_palette(sns.color_palette(color_palette))
plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.95)
ax = sns.histplot(data=axr_df, x='axr', stat='percent', log_scale=True, bins=20)
plt.xlabel('Neoantigen Quality (AxR)')
plt.ylabel('Percent')
plt.savefig(path_to_gridsearch + "AxR_hist.png")
plt.savefig(path_to_gridsearch + "AxR_hist.svg", format='svg')

# Set up gridsearch to find initial value
maxruns = 5000
def simLoss(x0):
    return simulateTherapy(x0[0], x0[1], AxR, maxruns, model_type)
n=15 # Number of grid boxes in each parameter. Note: n=15 is default. You will need to regenerate gridsearch and optimization to change this value.
m_mult_array = 5*np.logspace(-8, -6, n)
k_mult_array = 5*np.logspace(-9, -6, n)
if regenerate_gridsearch==True:
    ans_array = np.zeros((len(m_mult_array), len(k_mult_array)))
    iter = 0
    for m_mult in range(len(m_mult_array)):
        for k_mult in range(len(k_mult_array)):
            m_init_guess = m_mult_array[m_mult]
            k_init_guess = k_mult_array[k_mult]
            ans_array[m_mult, k_mult] = simLoss([m_init_guess, k_init_guess])
            iter += 1
            print(str(round(iter*100/(n**2), 2)) + ' percent done')
    dill.dump(ans_array, open(path_to_gridsearch + "gridsearch_" + model_type + "_" + str(n) + "logspace_nparray_maxruns_" + str(maxruns) + ".dump", 'wb'))


# Plot heatmaps (Supplementary Fig 4b):
ans_array = dill.load(open(path_to_gridsearch + "gridsearch_" + model_type + "_" + str(n) + "logspace_nparray_maxruns_" + str(maxruns) + ".dump", 'rb'))
plt.figure(figsize=(6, 5))
plt.subplots_adjust(left=0.2, right=1, bottom=0.2, top=0.94)
plt.title('Initial value grid search')
ax = sns.heatmap(data=ans_array, xticklabels=[np.format_float_scientific(k_mult_array[i], precision=1) for i in range(len(k_mult_array))],
            yticklabels=[np.format_float_scientific(m_mult_array[i], precision=1) for i in range(len(m_mult_array))], linewidth=0.5,
            vmin=np.min(ans_array), vmax=np.max(ans_array), cmap = 'magma') #vmax=np.min(ans_array), cmap='magma')
plt.xlabel('$\Gamma_k$')
plt.ylabel('$\Gamma_m$')
ax.invert_yaxis()
plt.savefig(path_to_gridsearch + "init_val_grid_search_" + model_type + "_logspace_nparray_monoclonal_maxruns_" + str(maxruns) + ".png")
plt.savefig(path_to_gridsearch + "init_val_grid_search_" + model_type + "_logspace_nparray_monoclonal_maxruns_" + str(maxruns) + ".svg")

maxrange = 20 # set the upper limit for the second heatmap, as (min(objective function value), min(objective function value) + maxrange)
tick_location = np.linspace(np.min(ans_array), np.min(ans_array)+maxrange, 5)
tick_labels = [str(round(i)) for i in tick_location]
tick_labels[-1] = '$\geq$' + tick_labels[-1]
plt.figure(figsize=(6, 5))
plt.subplots_adjust(left=0.2, right=1, bottom=0.2, top=0.94)
plt.title('Initial value grid search, second view')
ax = sns.heatmap(data=ans_array, xticklabels=[np.format_float_scientific(k_mult_array[i], precision=1) for i in range(len(k_mult_array))],
            yticklabels=[np.format_float_scientific(m_mult_array[i], precision=1) for i in range(len(m_mult_array))], linewidth=0.5,
            vmin=np.min(ans_array), vmax=np.min(ans_array)+maxrange, cmap = 'magma', cbar_kws={'ticks':tick_location, 'format':mticker.FixedFormatter(tick_labels)}) #vmax=np.min(ans_array), cmap='magma')
plt.xlabel('$\Gamma_k$')
plt.ylabel('$\Gamma_m$')
ax.invert_yaxis()
plt.savefig(path_to_gridsearch + "maxrange_init_val_grid_search_" + model_type + "_logspace_nparray_monoclonal_maxruns_" + str(maxruns) + ".png")
plt.savefig(path_to_gridsearch + "maxrange_init_val_grid_search_" + model_type + "_logspace_nparray_monoclonal_maxruns_" + str(maxruns) + ".svg")


# Final optimization run:
m_ind, k_ind = np.unravel_index(np.argmin(ans_array), ans_array.shape)
print('Initial guess for Gamma_m: ' + str(m_mult_array[m_ind]))  # This is the initial guess for Gamma_m.
print('Initial guess for Gamma_k: ' + str(k_mult_array[k_ind]))  # This is the initial guess for Gamma_k.
if regenerate_optimization==True:
    maxruns = 5000
    def simLoss(x0):
        return simulateTherapy(x0[0], x0[1], AxR, maxruns,model_type)
    opt_mk = fmin(simLoss, x0=(m_mult_array[m_ind], k_mult_array[k_ind]))  # set initial value to be the best found in grid search. Warning: slow!
    print('Re-optimized gamma_m, gamma_k values:')
    print(opt_mk)
else:
    m_proportionality_constant = 1.757e-7
    k_proportionality_constant = 3.419e-7
    print('Gamma_m = ' + str(m_proportionality_constant))
    print('Gamma_k = ' + str(k_proportionality_constant))