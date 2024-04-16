# srdop
Spatially resolved dynamics of organic polaritons

gmanp.py
--------
Class containing methods to construct and work with lambda basis defined in
thesis. Called by c2HTC.py / does not need to be edited.
- Note saves calculated structure tensors to .pkl files in ./data/tensors to
  avoid recomputing on later runs

c2HTC.py
--------
Main file for calculating dynamics using second-order cumulants.
Basic example is run using ./c2HTC.py (or python c2HTC.py).

cumulant_in_code.pdf
--------------------
Summary of notations for cumulant equations and coefficients used in the code
(which isn't in the thesis explicitly).


Requirements
------------
- python>=3.11
- numpy, scipy, opt_einsum, matplotlib, mpmath, pretty_traceback (optional),
  progressbar, sparse
