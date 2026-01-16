# neoantigen_public
Publicly available code for work with CRC neoantigens. Typical install time: < 10 minutes.

If you would like to try our code on a small set of tumors, please download the repo and run the "demo.py" script, which generates 10 MMRD and 10 MMRP tumors, simulates immunotherapy, and provides descriptive statistics about the results. Note that re-generating tumors requires an exact stochastic simulation (Gillespie algorithm) and is quite slow; you can expect about 10 minutes per tumor. We provide a sample set of tumors for you in the "demo_data" folder.

If you would like to fully reproduce the results in our paper, you will need to run the following scripts:
For tumor generation: use data_gen_hyak.py, preferably on a high-performance cluster (we used Hyak at UW, as the name suggests) that can generate tumors in parallel. This will take a fair amount of compute time.
For model fitting: use simulate_immunotherapy_optimize_params. This script walks you through the process of the grid search to find initial values, plus the final optimization run.
For simulated immunotherapy: we use simulate_immunotherapy_manual_ICs.py for this.
