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

gcc phase_space.c -o phase_space
