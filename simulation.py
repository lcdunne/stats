# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:33:58 2022

@author: L
"""

# Create factorial designs
import numpy as np
import pandas as pd
from itertools import product

# Simulating Chi-Square Observation Structures ------------------------------- #
def create_crosstab_design(spec):
    n_factors = len(spec)
    
    if n_factors > 2:
        raise ValueError("Crosstab dimensions cannot exceed 2; received {len(spec)} dimensions.")

    if n_factors == 1:
        # Just a single factor
        factor = list(spec)[0]
        levels = spec[factor]
        idx = pd.Index(levels, name=factor)
        pad = [0 for _ in levels]
        return pd.DataFrame(pd.Series(pad, index=idx), columns=['observed'])
    
    R, C = spec.keys()
    rows = pd.Series(spec[R], name=R)
    cols = pd.Series(spec[C], name=C)
    return pd.DataFrame(index=rows, columns=cols).fillna(0)

def add_data_to_crosstab(crosstab, data, expand=False):
    # Data must be in the form of [[r1c1, r1c2, ...], [r2c1, r2c2, ...]] if 
    # dims are r x c, else [r1, r2]
    if crosstab.shape[1] == 1:
        crosstab += np.array(data).reshape(-1, 1)
    else:
        crosstab += np.array(data)
    if expand:
        pass
    return crosstab

def expand_table(data):
    # Expands a contingency table of counts to long-format
    long = data.unstack().reset_index()
    long = long.loc[long.index.repeat(long[0])].drop(0, axis=1).reset_index(drop=True)
    if len(data.columns) == 1:
        return long.drop('level_0', axis=1)
    return long

# Factorial designs --------------------------------------------------------- #
def create_factorial_design(*cells, n, repeated=False, spec=None, casenames=None):
    """
    Creates ANOVA-style design table for data simulation.
    
    TODO: If both varargs and spec passed, ensure same dims.
    
    Parameters
    ----------
    *cells : int
        Denotes the number of levels for the specified cell.
    n : int, optional
        Denotes the number of observations. When `repeated` is `False`, this 
        value corresponds to the number of independent samples *in each 
        category. When `repeated` is True, it corresponds to the number of 
        observations in the entire sample. The default is 1.
    repeated : bool, optional
        Specifies whether or not the design is repeated measures. The default 
        is False.
    spec : dict, optional
        A dictionary specification of factor name(s) (keys) and corresponding 
        list(s) of level name(s) (values), respectively. The default is None.
    casenames : str, optional
        A string used to relabel the cases. If None, cases column will be set 
        to `case`. The default is None.

    Raises
    ------
    ValueError
        In case `cells` varargs were not specified *and* a `names` dict was 
        not specified.

    Returns
    -------
    df : pandas.core.DataFrame
        Dataframe with the structured design.
    
    Example
    --------
    To create a 2-factor independent design with 2 and 2 levels, respectively:

    >>> create_design(2, 2, n=1)
       x0   x1
    case      
    1   0   0
    2   0   1
    3   1   0
    4   1   1

    """
    if not any([cells, spec]):
        raise ValueError("Must specify one of cells or names, but received none.")

    if all([cells, spec]):
        spec_cells = tuple(len(x) for x in spec.values())
        if not spec_cells == cells:
            raise ValueError(f"Length mismatch between cells {cells} and specification {spec_cells}.")

    if not spec:
        spec = {f"x{xi}": list(range(k)) for xi, k in enumerate(cells)}

    if not cells:
        cells = tuple(len(levels) for levels in spec.values())

    structure = sorted(list(product(*[range(cell) for cell in cells]))*n)

    if repeated:
        cases = pd.Index(list(range(1, n+1)) * np.product(cells), name='case')
    else:
        cases = pd.Index(list(range(1, len(structure)+1)), name='case')

    labels = {col: {k: lev for k, lev in enumerate(spec[col])} for col in spec.keys()}
    df = pd.DataFrame(structure, columns=spec.keys()).replace(labels)
    df.insert(0, 'case', cases)

    if casenames is not None:
        df = df.rename(columns={'case': casenames})

    return df.sort_values(by=list(df.columns)).reset_index(drop=True)

def dummify(data, order_mapper, idvar=None):
    # # Order mapper has keys equal to the relevant columns to dummify
    # to_dummify = list(order_mapper.keys())
    # dumdata = data[to_dummify].replace(order_mapper)
    # return pd.get_dummies(dumdata)

    data = data.copy()

    to_dummify = list(order_mapper.keys())
    dumdata = pd.get_dummies(data.replace(order_mapper), columns=to_dummify)
    dumdata = dumdata.drop(labels=[f"{c}_0" for c in to_dummify], axis=1)

    repmap = {}
    for dum in to_dummify:
        for key, value in order_mapper[dum].items():
            if value == 0:
                continue
            repmap['_'.join([dum, str(value)])] = '_'.join([dum, key.lower()])

    return dumdata.rename(columns=repmap)

# Linear Modeling ----------------------------------------------------------- #
def make_betas(*coefs):
    return np.array(*[coefs])

def simulate(X, betas, errdev=1):
    N = len(X)
    X = np.c_[np.ones(N), X] # Add intercept
    e = np.random.normal(0, errdev, N) # Errors
    return X @ betas + e

def bound_data(x, lower, upper):
    # Because real data is often bounded, we replace values lying outside a certain range.
    y = x.copy()
    y[y < lower] = lower
    y[y > upper] = upper
    return y

if __name__ == '__main__':
    print()