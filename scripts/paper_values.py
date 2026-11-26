"""
Published estimates from Favara & Imbs (2015), AER 105(3), typed in from the paper.
02_iv_regression.py compares our estimates with these numbers (Worksheet A.3).
Format: {column: {"coef": coefficient on the variable of interest, "se": standard error,
                   "N": observations, "counties": number of counties, "F": first-stage F (IV only)}}
"""

# Table 4: reduced form, log change in house prices on the deregulation index
# (county and year FE, weights w1, SEs clustered by state)
TABLE4 = {
    "(1)": {"coef": 0.0103, "se": 0.004, "N": 12214, "counties": 1021},
    "(2)": {"coef": 0.0242, "se": 0.005, "N": 10627, "counties": 887},
    "(3)": {"coef": 0.0122, "se": 0.002, "N": 9735, "counties": 886},
}

# Table 5: IV, log change in house prices on instrumented credit growth of commercial banks
# (county FE, year dummies partialled out, weights w1, HAC SEs, Bartlett kernel, bandwidth 3)
TABLE5 = {
    "(1) Number of loans": {"coef": 0.141, "se": 0.069, "N": 11107, "counties": 1018, "F": 10.13},
    "(2) Loan volume": {"coef": 0.134, "se": 0.062, "N": 11107, "counties": 1018, "F": 10.31},
    "(3) Loan to income": {"coef": 0.120, "se": 0.057, "N": 11107, "counties": 1018, "F": 8.59},
}
