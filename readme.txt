This is the codes for E2b_NMR experiment.

Use scripts in /programmes to analyse data in /data

IMPORTANT:  
    - When analyzing t1 for Glycerol concetration and mineral, use T1_exp_fit_Gly_conc_and_mineral.py
    - for other t1, use t1_exp_fit.py
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
        T1_exp_fit_Gly_conc_and_mineral.py (for analyzing t1 of Gly concentration and mineral oil data)
        plot_T2_vs_tau.py (for analyzing delay time effect to T2, used only at the beginning)

About /data:
    All the data from experiments. Some data not used.
About /log:
    Lab logs.

About /figures:
    results for t2 are saved in /figures/t2/results
    results for t1 are hidden in the corresponding subdirectories (figure title and file name not implemented yet)


        
