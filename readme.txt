This is the codes for E2b_NMR experiment.

Use scripts in /programmes to analyse data in /data

IMPORTANT:  

    - clean.py is IREVERSIBLE
    - different t2 fitting requires different peak finding parameters. 
    - for t2, DO NOT use run all /directory path anymore, which will use same parameters to fit all data and leads to bad results. only do analysis one file at a time for any change.


About /programmes:
    Most commly used programmes: 

        t1_exp_fit.py (for t1 calculation and automatically analyze trend with temperature/concentration when analyzing a whole directory of files)
        t2_exp_fit.py (for t2 calculation)
        t2_trend.py (for analyzing trend of t2 with temperature/concentration)
    Other scripts:
        clean.py (for cleaning csv data. NOTICE THIS STEP is IREVERSIBLE)
        T2_plot_only.py (for plotting and selecting best plot for analysis)

About /data:
    All the data from experiments. Some data not used.
About /log:
    Lab logs.

About /figures:
    results for t2 are saved in /figures/t2/results
    results for t1 are hidden in the corresponding subdirectories (figure title and file name not implemented yet)


        
