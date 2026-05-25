# neoantigen_public
Publicly available code for work with CRC neoantigens.
Sholokhova A, Kaveh K, Bozic I. Neoantigen evolution and response to checkpoint inhibitor immunotherapy in colorectal cancer. Nature Communications 17:4543  (2026).

[![DOI](https://zenodo.org/badge/842756305.svg)](https://doi.org/10.5281/zenodo.18676798)

Typical install time: < 10 minutes.

If you would like to try our code on a small set of tumors, please download the repo and run the "demo.py" script, which generates 10 MMRD and 10 MMRP tumors, simulates immunotherapy, and provides descriptive statistics about the results. Note that re-generating tumors requires an exact stochastic simulation (Gillespie algorithm) and is quite slow; you can expect about 10 minutes per tumor. We provide a sample set of tumors for you in the "demo_data" folder.

If you would like to fully reproduce the results in our paper using the exact same data we generated, please use the pickled .dump file, titled "monoclonaltherapydata_5000_pandas_df.dump" which is located in the "Paper_Data" folder. You may load this dataset in a Python script using pandas, e.g. by:
therapydata = pd.read_pickle(open(your_path_here + "monoclonaltherapydata_5000_pandas_df.dump", 'rb'))
The script "simulate_immunotherapy_manual_ICs.py" can be used to reproduce the figures and results described in the paper on this dataset. Please remember to edit the paths at the beginning of the file to point the script in the right location.

If you would like to fully reproduce the results in our paper by generating a NEW dataset, you will need to run the following scripts:
For tumor generation: use data_gen_hyak.py, preferably on a high-performance cluster (we used Hyak at UW, as the name suggests) that can generate tumors in parallel. This will take a fair amount of compute time for a tumor grown to a size of 100,000 cells, likely around 10-60 minutes per tumor; additionally, some tumors will stochastically die out before reaching full size, meaning that some runs will be "wasted" or unusable.
For model fitting: use simulate_immunotherapy_optimize_params. This script walks you through the process of the grid search to find initial values, plus the final optimization run. The final optimization run takes a significant amount of time; expect 24+ hours on a reasonably modern laptop.

For simulated immunotherapy: we use simulate_immunotherapy_manual_ICs.py for this. The first time running this script for a new batch of tumors will take a fair amount of time to run but should complete for 10,000 tumors in under 24 hours on a standalone machine.

Any questions can be raised as an "Issue" or otherwise directed to me (Alanna): alannasholokhova@gmail.com
