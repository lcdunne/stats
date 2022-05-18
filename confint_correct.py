def _make_correction(data, id_var, within, y):
    # Correct data for within-subjects so that CIs are for within-subs diffs.
    J = len(within)
    X = data.pivot(index=id_var, columns=within, values=dv)
    Y = (X.T + X.mean().mean() - X.mean(axis=1)).T
    return Y.reset_index().melt(id_vars=id_var, value_name=y)
