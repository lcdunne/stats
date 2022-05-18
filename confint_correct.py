def make_correction(data, id_var, factors, dv):
    # Correct data for within-subjects so that CIs are for within-subs diffs.
    J = len(factors)
    X = data.set_index([id_var] + factors)[dv].unstack(list(range(1, J + 1)))
    Y = (X.T + X.mean().mean() - X.mean(axis=1)).T # Apply the correction
    Y = Y.unstack().reset_index()[[id_var] + factors + [0]]
    return Y.rename(columns={0: dv +'_adj'})
