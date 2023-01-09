# Test setup: same dataset as in the notebook example

# FIXME: This is just the informal tests that I was using whilst pulling the
#        initial version together.  It needs to be replaced with a proper pytest
#        setup when I come back in January.

test_data = pd.read_csv("/home/cdsw/rbeis-understanding/data.csv")
test_imp_var = "white"
test_aux_vars = ["interim_id", "gor9d", "work_status_group", "dvhsize"]
test_df = 1
test_weights = {k: 1.0 for k in test_aux_vars}


def t1():
    _add_impute_col(test_data, test_imp_var)


def t2():
    _assign_igroups(test_data, test_aux_vars)


def t3():
    _calc_distances(test_data, test_aux_vars, test_df, test_weights)


def t4():
    _calc_donors(test_data)


def t4a():
    t1()
    t2()
    t3()
    t4()
    return test_data


def t5(i):
    return _impute_igroup(
        test_data,
        _freq_to_exp(test_data, _get_freq_dist(test_data, "white", [0, 1], i), i),
        [0, 1],
        i,
    )


def t6():
    impute(
        test_data,
        "dvsex",
        [1, 2],
        ["interim_id", "uac", "hh_id_fake"],
        {"interim_id": 1, "uac": 2, "hh_id_fake": 3},
        6,
        threshold=1,
        custom_df_map={(1, 1): 2},
        min_quantile=4,
        keep_intermediates=True,
    )
