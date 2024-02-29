# srdop
Spatially resolved dynamics of organic polaritons

gmanp.py
--------
Class containing methods to construct and work with lambda basis defined in
thesis. Called by c2HTC.py / does not need to be edited.
- Note saves calculated structure tensors to .pkl files in ./data/tensors to
  save having to compute each time

c2HTC.py
--------
Main file for calculating dynamics for organic polariton transport using second-order cumulants.
Basic example is run using ./c2HTC.py (or python c2HTC.py).

- Parameters defined at bottom of script in a dictionary called 'params'
- main call is htc.evolve(tf_fs=tf_fs) which computes dynamics from 0 to time
  tf_fs in fs
- results are stored in numpy arrays in dictionary htc.observables
- HTC.omega(K) defines dispersion of model
- HTC.eoms() defines equations of motion as in cumulant_in_code.pdf note
- HTC.incoherent_state() defines the initial Gaussian exciton population
- Key computational parameters are number of modes Nk and number of vibrational
  levels; anything more than Nnu=3 or 4 is going to take a lot of RAM (see Fig.
6.12 of thesis). Thesis used Nk=301, Nnu=4, but lower during testing.


Requirements
------------
- numpy, scipy, opt_einsum, matplotlib+seaborn (plotting), pretty_traceback
  (optional)
