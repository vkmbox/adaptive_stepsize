import sys
from sympy import solve, core
from sympy.abc import x

import logging

def cubic_real_positive_smallest(a3, a2, a1, a0, default, do_logging):
    if do_logging:
        logging.info("##--==On step 2 cubic coeffs: a3={}, a2={}, a1={}, a0={} ==--".format(a3, a2, a1, a0))
    roots = solve(a3*x**3 + a2*x**2 + a1*x + a0, x)
    if do_logging:
        logging.info("##--==On step 2 cubic roots: {} ==--".format(roots))
    res = sys.float_info.max
    for root in roots:
        if type(root) is core.numbers.Float:
            buff = float(root)
            if 0.0 < buff and buff < res:
                res = buff
    if res == sys.float_info.max:
        res = default
    
    return res
