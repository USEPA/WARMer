# olca_warm_io.py (WARMer)

"""
Import WARM OpenLCA model's technology & intervention matrices
Last Updated: Wednesday, November 11, 2021
"""
import numpy as np
import pandas as pd
import uuid
from olca_data_unpack import classify_processes
from mapping import map_warmer_envflows, map_useeio_processes
from pathlib import Path

modulepath = Path(__file__).parent
warm_version = 'WARMv15'

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
    df.columns = df.columns.astype(str).str.replace(' ', '_')

    # correct economic flow units in WARMv15: USD 2002 should be USD 2007
    if filename == 'index_B.csv':
        df['flow_unit'] = df['flow_unit'].replace('USD 2002', 'USD 2007')

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
    Append process_ID & flow_ID values to rows & columns of matrices
    based on order of entries in each matrix and index df (and thereby file)
    :param mtx_a: pd dataframe, unlabeled A-matrix
    :param mtx_b: pd dataframe, unlabeled b-matrix
    :param idx_a: pd dataframe, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    mtx_a.columns = idx_a['process_ID']
    mtx_a.insert(loc=0, column='from_process_ID', value=idx_a['process_ID'])
    mtx_b.columns = idx_a['process_ID']  # i.e., why this fxn needs all 4 df's
    mtx_b.insert(loc=0, column='to_flow_ID', value=idx_b['flow_ID'])
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

    df = pd.melt(mtx_i, id_vars=k[0], var_name=k[1], value_name='Amount')
    return df


def label_exch_dfs(df_a, df_b, idx_a, idx_b):
    """
    Merge full sets of index label fields into df's of exchanges.
    :param df_a: pd dataframe, A-matrix exchanges w/ IDs
    :param df_b: pd dataframe, B-matrix exchanges w/ IDs
    :param idx_a: pd dataframe, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    df_a_m = (df_a.merge(idx_a.add_prefix('from_'),
                         how='left', on='from_process_ID')
                  .merge(idx_a.add_prefix('to_'),
                         how='left', on='to_process_ID'))
    if inspect_df_merge(df_a, df_a_m) == False:
        print('WARNING inspect labeled df_a for merge errors')

    df_b_m = (df_b.merge(idx_a.add_prefix('from_'),
                         how='left', on='from_process_ID')
                  .merge(idx_b.add_prefix('to_'),
                         how='left', on='to_flow_ID'))
    if inspect_df_merge(df_b, df_b_m) == False:
        print('WARNING inspect labeled df_b for merge errors')
    return df_a_m, df_b_m

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
        that target (via to_process_fgbg) foreground processes.
    For B, keep remaining A "to_" processes.
    An optional, extra filter list .txt file of (foreground) processes can be
        passed in via the filterfile argument, which looks in warmer/data/.
    :param df_a: pd dataframe, A-matrix exchanges w/ all labels
    :param df_b: pd dataframe, B-matrix exchanges w/ all labels
    :param filterfile: string, filename
    """
    df_a_f = df_a.query('to_process_fgbg == "foreground"')
    if filterfile is not None:  # must check & perform before df_b filtering
        subset_keep = pd.read_csv(modulepath/'data'/filterfile, header=None)
        df_a_f = df_a_f.merge(subset_keep, how='right',
                              left_on = 'to_process_name',
                              right_on = 1)
        df_a_f.loc[:, 'to_process_ID'] = df_a_f[0]
        df_a_f.drop(columns = [0,1], inplace=True)

        # Update IDs for consumed flows as well
        df_a_f = df_a_f.merge(subset_keep, how = 'left',
                              left_on = 'from_process_name',
                              right_on = 1)
        df_a_f.loc[~df_a_f[0].isna(), 'from_process_ID'] = df_a_f[0] + '/US'
        df_a_f.drop(columns = [0,1], inplace=True)

        df_b_f = df_b.merge(subset_keep, how='inner',
                            left_on = 'from_process_name',
                            right_on = 1)
        df_b_f.loc[:, 'from_process_ID'] = df_b_f[0]
        df_b_f.drop(columns = [0,1], inplace=True)
        df_b_f.sort_values(by='from_process_ID', inplace=True)
    else:
        df_b_f = df_b.query('from_process_ID in @df_a_f.to_process_ID')
    return df_a_f, df_b_f

def map_agg(df_b, idx_b):
    """
    Map elementary flows to FEDEFL format, derive a new idx_b w/ temporary IDs,
    and aggregate flows in df_b with newly-overlapping categorical variables
    :param df_b: pd dataframe, B-matrix exchanges w/ all labels
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    df = map_warmer_envflows(df_b)

    # derive new flow_ID_temp from FEDEFL-mapped (name + category)
    df['ID_str'] = df['to_flow_name'] + '_' + df['to_flow_category'].fillna('')
    df['to_flow_ID_temp'] = df['ID_str'].apply(
        lambda x: str(uuid.uuid3(uuid.NAMESPACE_OID, x)))
    df = df.drop(columns=['to_flow_ID', 'ID_str'])
    # extract new idx_b df from mapped df_b
    mask = df.columns.intersection('to_' + idx_b.columns).tolist()
    idx = df[['to_flow_ID_temp'] + ['FlowUUID'] + mask].drop_duplicates()
    idx.columns = idx.columns.str.replace('to_','')

    # aggregate df_b flows
    aggcols = df.columns.tolist()
    aggcols.remove('Amount')  # aggcols = [col for col in aggcols if col not in ['Amount','to_flow_ID']]
    df = df.groupby(aggcols, as_index=False, dropna=False, observed=True).sum()
    return df, idx

def pivot_to_labeled_mtcs(df_a, df_b, idx_a, idx_b):
    """
    Convert dfs of labeled exchanges back into matrices
    :param df_a: pd dataframe, A-matrix exchanges w/ all labels
    :param df_b: pd dataframe, B-matrix exchanges w/ all labels
    :param idx_a: pd dataframe, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    idx_a, idx_b = sort_idcs(idx_a, idx_b)

    mtx_a = (df_a.loc[:,['from_process_ID','to_process_ID','Amount']]
                 .pivot(index='from_process_ID', columns='to_process_ID',
                        values='Amount')
                  .fillna(0))      # reinstate filtered out 0-flows
    # reorder mtx_a process_ID's by sorted idx_a process_ID
    mtx_a = (mtx_a.reindex(index=idx_a.index.intersection(mtx_a.index))
                  .transpose())
    mtx_a = (mtx_a.reindex(index=idx_a.index.intersection(mtx_a.index))
                  .transpose()
                  .rename_axis('process_ID')  # formerly from_process_ID
                  .reset_index())
    mtx_a_i = (idx_a.merge(mtx_a, how='right', on='process_ID')
                    .transpose()
                    .rename_axis('process_ID')  # formerly to_process_ID
                    .reset_index())
    mtx_a_i = (idx_a.merge(mtx_a_i, how='right', on='process_ID'))
    # manual field name label position adjustments
    a = len(idx_a.columns) + 1  # to account for process_ID becoming the index
    mtx_a_i.iloc[0:a,a-1] = mtx_a_i.iloc[0:a,0]
    mtx_a_i.iloc[0:a,0] = np.nan
    mtx_a_i.iloc[a-1,0:a] = mtx_a_i.columns[0:a]
    mtx_a_lab = mtx_a_i.transpose()

    mtx_b = (df_b.loc[:,['from_process_ID','to_flow_ID_temp','Amount']]
                  .pivot(index='from_process_ID', columns='to_flow_ID_temp',
                        values='Amount')
                  .fillna(0))      # reinstate filtered out 0-flows
    # reorder mtx_b from_process_ID's by sorted idx_a process_ID
    mtx_b = (mtx_b.reindex(index=idx_a.index.intersection(mtx_b.index))
                  .rename_axis('process_ID')  # formerly from_process_ID
                  .reset_index())
    mtx_b_i = (idx_a.drop(columns='process_fgbg')
                    .merge(mtx_b, how='right', on='process_ID')
                    .transpose()
                    .rename_axis('flow_ID_temp')  # formerly to_flow_ID_temp
                    .reset_index())
    mtx_b_lab = (idx_b.merge(mtx_b_i, how='right', on='flow_ID_temp'))
    # manual field name label position adjustments
    b = len(idx_b.columns)
    mtx_b_lab.iloc[0:a-1,b-1] = mtx_b_lab.iloc[0:a-1,1]
    mtx_b_lab.iloc[0:a-1,1] = np.nan
    mtx_b_lab.iloc[a-1,0:b] = mtx_b_lab.columns[0:b]
    mtx_b_lab = mtx_b_lab.drop(columns='flow_ID_temp')
    return mtx_a_lab, mtx_b_lab

def sort_idcs(idx_a, idx_b):
    """
    Sort idx_a and idx_b columns, plus idx_a rows to prep for matrix labeling.
    DO NOT CALL BEFORE append_mtx_IDs().
    :param idx_a: pd dataframe, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd dataframe, labels for B-matrix rows
    """
    # column sorting
    idx_a, idx_b = map(sort_idx_cols, [idx_a, idx_b])

    # row sorting
    idx_a = idx_a.sort_values(by=['process_class','process_name'])
    idx_a['process_class'] = idx_a['process_class'].astype(str)
        # categorical back to string enables manual matrix label adjustments
    idx_a = idx_a.set_index('process_ID')
    return idx_a, idx_b

def sort_idx_cols(df_idx):
    """
    Multi-level sorting of index df columns, by (1) "process" before "flow",
    and (2) substring order defined by order_cols
    :param df_idx: pd dataframe, matrix labels
    """
    cols = df_idx.columns
    p = cols[cols.str.startswith('process')].str.replace('process_','')
    f = cols[cols.str.startswith('flow')].str.replace('flow_','')
    order_cols = pd.Index(['ID', 'ID_temp', 'category', 'class', 'fgbg',
                           'location', 'type', 'name', 'unit'])
    p_o = ('process_' + order_cols.intersection(p)).tolist()
    if 'FlowUUID' in cols:
        p_o = ['FlowUUID'] + p_o
    f_o = ('flow_' + order_cols.intersection(f)).tolist()
    df_idx = df_idx[p_o + f_o]
    return df_idx


def format_for_export(df, opt):
    if opt == 'a':
        col_dict = {'to_process_ID': 'ProcessID',
                    'to_process_name': 'ProcessName',
                    'to_flow_unit': 'ProcessUnit',
                    'to_process_location': 'Location',
                    'Amount': 'Amount',
                    'from_process_ID': 'FlowID',
                    'from_process_name': 'Flow',
                    'from_flow_unit': 'FlowUnit',
                    # 'from_process_fgbg': 'FlowMap'  # use when fbgb is needed
                    }
        df.loc[(df['to_process_ID'] == df['from_process_ID'].str.split('/').str[0]) &
               (df['Amount']==1), 'Amount'] = 0
        # Invert all signs in A matrix
        df['Amount'] = df['Amount'] * -1
    else: # opt == 'b'
        col_dict = {'from_process_ID': 'ProcessID',
                    'from_process_name': 'ProcessName',
                    'from_process_category': 'ProcessCategory',
                    'from_process_location': 'Location',
                    'Amount': 'Amount',
                    'to_flow_name': 'Flowable',
                    'to_flow_category': 'Context',
                    'to_flow_unit': 'Unit',
                    'FlowUUID': 'FlowUUID',
                    }

    df_mapped = df[list(col_dict.keys())]
    df_mapped = df_mapped.rename(columns=col_dict)
    df_mapped['Location'] = df_mapped['Location'].fillna('US')
    df_mapped.dropna(subset=['Amount'], inplace=True)
    return df_mapped

if __name__ == '__main__':
    model_version = None
    # model_version = 'm1'
    file_stub = f'{warm_version}'
    if model_version:
        file_stub = file_stub + f'_{model_version}'

    a_raw, b_raw, idx_a, idx_b = map(
        read_olca2, ['A.csv', 'B.csv', 'index_A.csv', 'index_B.csv'])

    idx_a = classify_processes(idx_a, 'fgbg')  # assign before merges
    idx_a = classify_processes(idx_a, 'class')  # assign before merges

    mtx_a, mtx_b = normalize_mtcs(a_raw, b_raw)

    mtx_a, mtx_b = append_mtx_IDs(mtx_a, mtx_b, idx_a, idx_b)

    df_a, df_b = map(melt_mtx, [mtx_a, mtx_b], ['a', 'b'])

    df_a, df_b = label_exch_dfs(df_a, df_b, idx_a, idx_b)

    df_a, df_b = filter_processes(df_a, df_b)
    ## Notes:
    # 1. Some 145 processes in A-mtx are technosphere mixing processes,
        # with all 0-valued elementary flows and no entries in df_b
    # 2. Of the 14 elementary flows in idx_b, only 11 remain after
        # removing 0-value exchanges. Removed flows include:
            # Ethane, hexafluoro-, HFC-116
            # Methane, tetrafluoro-, R-14
            # Other means of transport (no truck, train or s...

    df_b, idx_b = map_agg(df_b, idx_b)

    mtx_a_lab, mtx_b_lab = pivot_to_labeled_mtcs(df_a, df_b, idx_a, idx_b)

    df_a = map_useeio_processes(df_a)

    ## Sample dataframe export (via .csv filter list)
    if model_version == 'm1':
        filename = 'model_1_processes.csv'
    elif model_version == 'm4':  #??? or is this the m3 file?
        filename = 'choose_processes.csv'
    else:
        filename = None
    df_a_eg, df_b_eg = filter_processes(df_a, df_b, filename)

    # a_eg, b_eg = pivot_to_labeled_mtcs(df_a_eg, df_b_eg, idx_a, idx_b)

    if model_version:
        writepath = modulepath.parent/'model_build'/'data'
        (format_for_export(df_a_eg, 'a')
             .query('Amount != 0')  # drop empty exchanges
             .to_csv(writepath/f'{file_stub}_tech.csv', index=False))
        (format_for_export(df_b_eg, 'b')
             .drop(columns='ProcessCategory')
             .query('Amount != 0')  # drop empty exchanges
             .to_csv(writepath/f'{file_stub}_env.csv', index=False))
    else:
        writepath = modulepath/'data'/'flowsa_inputs'
        # (format_for_export(df_a, 'a')
        #     .to_csv(writepath/f'{file_stub}_tech.csv', index=False))
        (format_for_export(df_b, 'b')
            .to_csv(writepath/f'{file_stub}_env.csv', index=False))

    ## Generate processmapping.csv
    # newcols = ['MatchCondition','ConversionFactor',
    #            'TargetListName','TargetProcessName','TargetUnit','LastUpdated']
    # a_index = a_index.reindex(columns = a_index.columns.tolist() + newcols)
    # a_index.to_csv(modulepath/'processmapping'/'processmapping.csv', index=False)
