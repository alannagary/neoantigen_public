# neoantigen_public
Publicly available code for work with CRC neoantigens. Typical install time: < 10 minutes.

If you would like to try our code on a small set of tumors, please download the repo and run the "demo.py" script, which generates 10 MMRD and 10 MMRP tumors, simulates immunotherapy, and provides descriptive statistics about the results. Note that re-generating tumors requires an exact stochastic simulation (Gillespie algorithm) and is quite slow; you can expect about 10 minutes per tumor. We provide a sample set of tumors for you in the "demo_data" folder.

If you would like to fully reproduce the results in our paper, you will need to run the following scripts:
For tumor generation: use data_gen_hyak.py, preferably on a high-performance cluster (we used Hyak at UW, as the name suggests) that can generate tumors in parallel. This will take a fair amount of compute time for a tumor grown to a size of 100,000 cells, likely around 10-60 minutes per tumor; additionally, some tumors will stochastically die out before reaching full size, meaning that some runs will be "wasted" or unusable.
For model fitting: use simulate_immunotherapy_optimize_params. This script walks you through the process of the grid search to find initial values, plus the final optimization run. The final optimization run takes a significant amount of time; expect 24 hours on a laptop or less on a higher-powered machine.

For simulated immunotherapy: we use simulate_immunotherapy_manual_ICs.py for this. The first time running this script for a new batch of tumors will take a fair amount of time to run but should complete for 10,000 tumors in under 24 hours on a standalone machine.
