<b>Modified code</b>

This program is a modified version of the Transient Phase Space program written by Evan Keane
and distributed via https://github.com/FRBs/Transient_Phase_Space . It is released under GNU Public
License: check LICENSE to see more details. You will need to provide the luminosity_nuW.txt file in
the data directory gach_rud/; this can be generated using S_fluence_wrt_time/plot_brightness_wrt_time.py .
Original README follows.

<b>Transient phase space plot</b>

This is some code to make a plot that some people have asked me
for. If you use this code, it would be nice if you gave me an
acknowledgemen. You could cite something appropriate like, for
example:

https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.3687P/

OR 

https://ui.adsabs.harvard.edu/abs/2018NatAs...2..865K/

but you don't have to.

For the C version just compile with something like:

`gcc phase_space.c -o phase_space`

<b> Updated plot with latest data and python code (Dec 2025)</b>

The above code was converted to python by S. Mate and it include new data from [Hurley-Walker et al. (2022)](http://doi.org/10.1038/s41586-021-04272-x), [Hurley-Walker et al. (2023)](http://doi.org/10.1038/s41586-023-06202-5); [Caleb et al. (2022)](http://doi.org/10.1038/s41550-022-01688-x), [Caleb et al. (2024)](http://doi.org/10.1038/s41550-024-02277-w); [Dong et al. (2024)](http://doi.org/10.1038/s41550-025-02491-0); [de Ruiter et al. (2024)](http://doi.org/10.3847/2041-8213/adfa8e). Please cite them when using the figure.