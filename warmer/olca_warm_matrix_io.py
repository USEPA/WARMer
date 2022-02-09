# olca_warm_io.py (WARMer)

"""
Import WARM OpenLCA model's technology & intervention matrices
Last Updated: Wednesday, November 11, 2021
"""
import pandas as pd
import numpy as np
from olca_data_unpack import classify_prcs
from mapping import map_warmer_envflows
from pathlib import Path

modulepath = Path(__file__).parent

def read_olca2(filename):
    """
    Import matrix (numeric values) or matrix index (labels only)
    :param filename: string
    """
    if 'index' in filename:
        opt_h = 'infer'  # index files have headers
    else:
        opt_h = None  # necessary for matrices

    df = (pd.read_csv(modulepath/'data'/'warm_olca_mtx'/filename, header=opt_h)
           .dropna(how='all', axis='columns'))  # drop entirely-nan columns
    return df

def normalize_mtcs(mtx_a, mtx_b):
    """
    Normalize A-matrix (square), dividing elements in each column by values on
    diagonal, such that diagonal elements all equal 1. Then perform element-
    wise division on columns of B-matrix by the same array of diagonal A values.
    :param mtx_a: pd dataframe, unlabeled A-matrix
    :param mtx_b: pd dataframe, unlabeled b-matrix
    """
    a = mtx_a.to_numpy()
    b = mtx_b.to_numpy()
    a_diag = a.diagonal()
    a_norm = a / a_diag[None,:]  # column-wise division by diagonal elements
    b_norm = b / a_diag[None,:]
    a_norm = pd.DataFrame(a_norm)
    b_norm = pd.DataFrame(b_norm)
    return a_norm, b_norm

def append_mtx_IDs(mtx_a, mtx_b, idx_a, idx_b):
    """
    Append process & flow ID values to rows & columns of matrices
    based on order of entries in each matrix and index df (and thereby file)
    :param mtx_a: pd dataframe, unlabeled A-matrix
    :param mtx_b: pd dataframe, unlabeled b-matrix
    :param idx_a: pd dataframe, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    mtx_a.columns = idx_a['process ID']
    mtx_a.insert(loc=0, column='from_process_ID', value=idx_a['process ID'])
    mtx_b.columns = idx_a['process ID']  # i.e., why this fxn needs all 4 df's
    mtx_b.insert(loc=0, column='to_flow_ID', value=idx_b['flow ID'])
    return mtx_a, mtx_b

def melt_mtx(mtx_i, opt):
    """
    Unpivot matrices to long format (i.e., rows are exchanges)
    :param mtx_i: pd dataframe, labeled matrix
    :param opt: string {'a','b'}
    """
    if opt=='a':  # target melt keys for A & b matrices
        k = ['from_process_ID', 'to_process_ID']
    elif opt=='b':
        k = ['to_flow_ID', 'from_process_ID']
    else:
        print(f'WARNING invalid "opt" string: {opt}')
        return None

    df = (pd.melt(mtx_i, id_vars=k[0], var_name=k[1], value_name='Amount')
          .query('Amount != 0'))  # drop empty exchanges
    return df

def label_exch_dfs(df_a, df_b, idx_a, idx_b):
    """
    Merge full sets of index label fields into df's of exchanges.
    :param df_a: pd dataframe, A-matrix exchanges w/ IDs
    :param df_b: pd dataframe, B-matrix exchanges w/ IDs
    :param idx_a: pd dataframe, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    iat, iaf, ibt = map(prep_idx_cols,  # prepend colnames w/ "to_" and "from_"
                        [idx_a, idx_a, idx_b],
                        ['to', 'from', 'to'])

    df_a_m = (df_a.merge(iaf, how='left', on='from_process_ID')
                  .merge(iat, how='left', on='to_process_ID'))
    if inspect_df_merge(df_a, df_a_m) == False:
        print('WARNING inspect labeled df_a for merge errors')

    df_b_m = (df_b.merge(iaf, how='left', on='from_process_ID')
                  .merge(ibt, how='left', on='to_flow_ID'))
    if inspect_df_merge(df_b, df_b_m) == False:
        print('WARNING inspect labeled df_b for merge errors')
    return df_a_m, df_b_m

def prep_idx_cols(df_idx, opt):
    """
    Remove whitespace and prepend index columns w/ "to_" and "from_" prefixes
    :param df_idx: pd dataframe, labels
    :param opt: string {'to','from'}
    """
    if opt not in {'to','from'}:
        print(f'WARNING invalid "opt" string: {opt}')
        return None
    df = df_idx.copy(deep=True)
    df.columns = opt + '_' + df.columns.str.replace(' ', '_').astype(str)
    return df

def inspect_df_merge(df_pre, df_post):
    """
    Quick post-merge assertions
    :param df_pre: pandas df, pre-merge
    :param df_post: pandas df, post-merge
    """
    eq_len = (len(df_pre)==len(df_post))  # ensure number of rows
    na_free = []
    if 'from_process_ID' in df_post.columns:  # check 'on' columns for na's
        na_free.append(not any(df_post.from_process_ID.isna()))
    if 'to_process_ID' in df_post.columns:    # check 'on' columns for na's
        na_free.append(not any(df_post.to_process_ID.isna()))
    if 'to_flow_ID' in df_post.columns:       # check 'on' columns for na's
        na_free.append(not any(df_post.to_flow_ID.isna()))
    na_free = all(na_free)
    all_good = all([na_free, eq_len])
    return all_good

def filter_processes(df_a, df_b, filterfile=None):
    """
    Filter processes in A and B simultaneously.
    For A, keep only foreground processes and their direct inputs
        (i.e., foreground + "background_map") by querying exchanges
        that target (via to_prcs_class) foreground processes.
    For B, keep remaining A "to_" processes.
    An optional, extra filter list .txt file of (foreground) processes can be
        passed in via the filterfile argument, which looks in warmer/data/.
    :param df_a: pd dataframe, A-matrix exchanges w/ all labels
    :param df_b: pd dataframe, B-matrix exchanges w/ all labels
    :param filterfile: string, filename
    """
    df_a_f = df_a.query('to_prcs_class == "foreground"')
    if filterfile is not None:  # must check & perform before df_b filtering
        prcs_keep = open(modulepath/'data'/filterfile).read().splitlines()
        df_a_f = df_a_f.query('to_process_name in @prcs_keep')
    df_b_f = df_b.query('from_process_ID in @df_a_f.to_process_ID')
    return df_a_f, df_b_f

def pivot_to_labeled_mtcs(df_a, df_b, idx_a, idx_b):
    """
    Convert dfs of labeled exchanges back into matrices
    :param df_a: pd dataframe, A-matrix exchanges w/ all labels
    :param df_b: pd dataframe, B-matrix exchanges w/ all labels
    :param idx_a: pd dataframe, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    mtx_a = (df_a.loc[:,['from_process_ID','to_process_ID','Amount']]
                 .pivot(index='from_process_ID', columns='to_process_ID',
                        values='Amount')
                  .fillna(0)      # reinstate filtered out 0-flows
                  .reset_index()  # convert former headers (now index) to col
                  .rename(columns={'from_process_ID': 'process ID'}))
    mtx_a_t = (idx_a.merge(mtx_a, how='right', on='process ID')
                    .transpose()
                    .reset_index()
                    .rename(columns={'index': 'process ID'}))
    mtx_a_t = (idx_a.merge(mtx_a_t, how='right', on='process ID'))

    mtx_b = (df_b.loc[:,['from_process_ID','to_flow_ID','Amount']]
                 .pivot(index='to_flow_ID', columns='from_process_ID',
                        values='Amount')
                  .fillna(0)      # reinstate filtered out 0-flows
                  .reset_index()  # convert former headers (now index) to col
                  .rename(columns={'to_flow_ID': 'flow ID'}))
    mtx_b_t = (idx_b.merge(mtx_b, how='right', on='flow ID')
                    .transpose()
                    .reset_index()
                    .rename(columns={'index': 'process ID'}))
    mtx_b_t = (idx_a.drop(columns='prcs_class')
                    .merge(mtx_b_t, how='right', on='process ID'))

    # manual position adjustments of field name labels
    mtx_a_t.iloc[0:11,10] = mtx_a_t.iloc[0:11,0]
    mtx_a_t.iloc[0:11,0] = np.nan
    mtx_a_t.iloc[10,0:11] = mtx_a_t.columns[0:11]
    mtx_a_lab = mtx_a_t.transpose()

    mtx_b_t.iloc[0:5,9] = mtx_b_t.iloc[0:5,0]
    mtx_b_t.iloc[0:5,0] = np.nan
    mtx_b_t.iloc[4,0:10] = mtx_b_t.columns[0:10]
    mtx_b_lab = mtx_b_t.transpose()
    return mtx_a_lab, mtx_b_lab

if __name__ == '__main__':
    a_raw, b_raw, idx_a, idx_b = map(
        read_olca2, ['A.csv', 'B.csv', 'index_A.csv', 'index_B.csv'])

    idx_a = classify_prcs(idx_a)  # assign prcs_class before merges
    # [later] assign add'l process classification here; sort in pivot_... fxn

    mtx_a, mtx_b = normalize_mtcs(a_raw, b_raw)

    mtx_a, mtx_b = append_mtx_IDs(mtx_a, mtx_b, idx_a, idx_b)

    df_a, df_b = map(melt_mtx, [mtx_a, mtx_b], ['a', 'b'])

    df_a, df_b = label_exch_dfs(df_a, df_b, idx_a, idx_b)

    df_b = map_warmer_envflows(df_b)

    df_a, df_b = filter_processes(df_a, df_b)
    # Notes:
        # 1. Some 145 processes in A-mtx are just technosphere mixing processes
            # with all 0-valued elementary flows & no entries in df_b
        # 2.  Of the 14 elementary flows in idx_b, only 10 remain after
            # keeping only non-0 exchanges. Removed flows include:
                # Ethane, hexafluoro-, HFC-116
                # Methane, tetrafluoro-, R-14
                # Other means of transport (no truck, train or s...
                # Carbon (biotic)
    mtx_a_lab, mtx_b_lab = pivot_to_labeled_mtcs(df_a, df_b, idx_a, idx_b)

    ## Sample matrix generation (via .txt filter list)
    df_a_eg, df_b_eg = filter_processes(df_a, df_b, 'sample_processes.txt')
    a_eg, b_eg = pivot_to_labeled_mtcs(df_a_eg, df_b_eg, idx_a, idx_b)
    a_eg.to_csv(modulepath/'data'/'eg_a_mtx.csv', index=False, header=False)
    b_eg.to_csv(modulepath/'data'/'eg_b_mtx.csv', index=False, header=False)


    # [last] handler function that calls all others sequentially?


    ## generate processmapping.csv
    # newcols = ['MatchCondition','ConversionFactor',
    #            'TargetListName','TargetProcessName','TargetUnit','LastUpdated']
    # a_index = a_index.reindex(columns = a_index.columns.tolist() + newcols)
    # a_index.to_csv(modulepath/'processmapping'/'processmapping.csv', index=False)
