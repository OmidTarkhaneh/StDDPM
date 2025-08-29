import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.util import ngrams
from scipy.special import rel_entr
from scipy.special import entr
from scipy.spatial import distance

from scipy.stats import energy_distance
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
import random
import os


def ks_dist(real_obs, gen_obs):
    stat, pval = ks_2samp(real_obs, gen_obs)

    return stat

def comapre_unidist_cont(CONT_FIELDS,CF_FIELD, real, synth, real_cf, synth_cf):
    """
    CONT_FIELDS : list of continuous columns
    CF_FIELD: name of the column in real_cf and synth_cf (used for computing cash flow)
    CONTINUOUS_METRICS = {"wasser": wasserstein_distance, "ks": ks_dist,"energy_d": energy_distance}
    real_cf, synth_cf: groupby(["account_id", "month", "year"]) real, and synth. and compute the sum of the raw_amount.

    """
    CONTINUOUS_METRICS = {"wasser": wasserstein_distance, "ks": ks_dist,"energy_d": energy_distance}
    univariate_cont_res = {}

    for field in CONT_FIELDS:
        univariate_cont_res[field] = {}
        for name, fn in CONTINUOUS_METRICS.items():
            univariate_cont_res[field][name] = fn(real[field], synth[field])

    univariate_cont_res['CF'] = {}
    for name, fn in CONTINUOUS_METRICS.items():
        univariate_cont_res['CF'][name] = fn(real_cf[CF_FIELD], synth_cf[CF_FIELD])
    return univariate_cont_res

def comapre_unidist_cat(real, synth, field):
    real_distribution = real[field].value_counts(normalize=True).sort_index()
    synthetic_distribution = synth[field].value_counts(normalize=True).sort_index()
    df_tcode = pd.merge(real_distribution, synthetic_distribution, left_index=True, right_index=True, how='outer')
    df_tcode.columns = ['real', 'synthetic']

    # Fill missing values with 0
    df_tcode.fillna(0, inplace=True)
    df_tcode['mid'] = (df_tcode['real'] + df_tcode['synthetic'])/2
    kl_real_M = sum(rel_entr(df_tcode['real'], df_tcode['mid']))
    kl_gen_M = sum(rel_entr(df_tcode['synthetic'], df_tcode['mid']))

    jsd = (kl_real_M + kl_gen_M)/2
    return jsd

def create_ngramcount_df(df, n, field):
    #gb = df.sort_values(by=["account_id", "datetime"]).groupby("account_id", sort=False)[field]
    gb = df.groupby("account_id", sort=False)[field]
    ngram_list = gb.apply(lambda x: list(ngrams(x, n=n)))

    counts = {}
    for ngram_seq in ngram_list:
        for ngram in ngram_seq:
            ngram = str(ngram)[1:-1]
            counts[ngram] = counts.get(ngram, 0) + 1


    df = pd.DataFrame.from_dict(counts, orient="index", columns=["counts"]).sort_values("counts", ascending=False)


    return df.reset_index().rename(columns={"index": "ngram"})

def compute_ngram_metrics(real_df, gen_df, field, n , pseudo_counts=0.0):


    n_codes_unique = len(set(real_df[field].unique()).union(set(gen_df[field].unique())))


    # create combo_df, which contains counts of all ngrams for both datasets (note: it omits any ngrams which do not occur in either dataset)
    real_ngrams = create_ngramcount_df(real_df, n, field)
    gen_ngrams = create_ngramcount_df(gen_df, n, field)
    combo_df = pd.merge(real_ngrams, gen_ngrams, on="ngram", how="outer", suffixes=("_real", "_gen")).fillna(0.0)


    N_obs_real = real_ngrams["counts"].sum()
    N_obs_gen = gen_ngrams["counts"].sum()
    N_possible_ngrams = n_codes_unique**n


    # add psudo-counts
    combo_df["counts_real"] += pseudo_counts
    combo_df["ps_real"] = combo_df["counts_real"] / (N_obs_real + N_possible_ngrams*pseudo_counts)
    combo_df["counts_gen"] += pseudo_counts
    combo_df["ps_gen"] = combo_df["counts_gen"] / (N_obs_gen + N_possible_ngrams*pseudo_counts)




    # compute jsd (note: contribution to jsd from any ngram not in either dataset is 0)
    combo_df["ps_mid"] = (combo_df["ps_real"] + combo_df["ps_gen"])/2
    kl_real_M = sum(rel_entr(combo_df["ps_real"], combo_df["ps_mid"]))
    kl_gen_M = sum(rel_entr(combo_df["ps_gen"], combo_df["ps_mid"]))

    jsd = (kl_real_M + kl_gen_M)/2


    # compute entropy for both distributions
    n_unobs = N_possible_ngrams - len(combo_df)

    entr_r = entr(combo_df["ps_real"]).sum()  # from observed

    entr_g = entr(combo_df["ps_gen"]).sum()  # from observed

    results = {"jsd":jsd,
                      "entr_r":entr_r,
                      "entr_g":entr_g,
                      "NED": entr_r - entr_g,
                      "l1":distance.minkowski(combo_df["ps_real"], combo_df["ps_gen"], p=1),
                      "l2":distance.minkowski(combo_df["ps_real"], combo_df["ps_gen"], p=2),
                      "jac": distance.jaccard(combo_df["counts_real"]>0, combo_df["counts_gen"] > 0),
                      "count_r": len(real_ngrams),
                      "coverage_r": len(real_ngrams)/N_possible_ngrams,
                      "count_g": len(gen_ngrams),
                      "coverage_g": len(gen_ngrams)/N_possible_ngrams,
                      "count_max": N_possible_ngrams,
                      "field": field,
                       "n":n,
                       "pseudo_counts":pseudo_counts}

    return combo_df, results

#joint distribution of two categorical columns
def compute_2d_categorical_metrics(real_df, gen_df, field1, field2):
    f1_opts = set(real_df[field1].unique()).union(set(gen_df[field1].unique()))
    f2_opts = set(real_df[field2].unique()).union(set(gen_df[field2].unique()))

    n_opts_total = len(f1_opts) * len(f2_opts)

    kl_r_m = 0.
    kl_g_m = 0.
    entr_r = 0.
    entr_g = 0.
    l1_d = 0.
    l2_d = 0.
    count_g = 0.
    count_r = 0.

    observed_opts = 0

    cont_metric_results = {}
    for code_1 in f1_opts:
        for code_2 in f2_opts:
            cond_r = np.logical_and(real_df[field1] == code_1, real_df[field2] == code_2)
            cond_g = np.logical_and(gen_df[field1] == code_1, gen_df[field2] == code_2)

            p_r = (np.sum(cond_r)) / (len(cond_r))
            p_g = (np.sum(cond_g)) / (len(cond_g))
            p_m = (p_r + p_g) / 2.

            if np.sum(cond_r) + np.sum(cond_g) > 0:
                observed_opts += 1


            count_r += int(np.sum(cond_r) > 0)
            count_g += int(np.sum(cond_g) > 0)

            l1_d += np.abs(p_r - p_g)
            l2_d += (p_r - p_g) ** 2


            if p_r > 0:
                kl_r_m += p_r * np.log(p_r / p_m)
                entr_r += - p_r * np.log(p_r)

            if p_g > 0:
                kl_g_m += p_g * np.log(p_g / p_m)
                entr_g += - p_g * np.log(p_g)

    # compute jaccard
    sr = set(zip(real_df[field1].to_list(), real_df[field2].to_list()))
    sg = set(zip(gen_df[field1].to_list(), gen_df[field2].to_list()))
    s_union = len(sr.union(sg))
    s_inter = len(sr.intersection(sg))
    jacc_d = (s_union - s_inter) / s_union

    # finshed l2
    l2_d = np.sqrt(l2_d)

    # coverage
    coverage_g = count_g / n_opts_total
    coverage_r = count_r / n_opts_total

    #jsd
    jsd = (kl_r_m + kl_g_m) / 2


    result = {'jsd': jsd,
                    'entr_r': entr_r,
                    'entr_g': entr_g,
                    'l1': l1_d,
                    'l2': l2_d,
                    'jac': jacc_d,
                    'count_r': count_r,
                    'coverage_r': coverage_r,
                    'count_g': count_g,
                    'coverage_g': coverage_g,
                    'count_max': n_opts_total}
    return result



