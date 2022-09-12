# olca_warm_io.py (WARMer)

"""
Import and transform WARM OpenLCA model's technology & intervention matrices
"""

import uuid
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from warmer.mapping import map_warmer_envflows, map_processes
import warmer.controls

modulepath = Path(__file__).parent
warm_version = 'WARMv15'

def read_olca2_mtx(filename):
    """
    Import matrix (numeric values) or matrix index (labels only) file
    exported from an openLCA v2.0.0
    :param filename: string
    """
    if 'index' in filename:
        opt_h = 'infer'  # (default) index files have headers
    else:
        opt_h = None  # necessary for matrices
    df = (pd.read_csv(modulepath/'data'/f'{warm_version}_olca_mtcs'/filename,
                      header=opt_h)
            .dropna(how='all', axis='columns'))  # drop entirely-nan cols
    df.columns = df.columns.astype(str).str.replace(' ', '_')
    # correct WARMv15 economic flow units: USD 2002 should be USD 2007
    if filename == 'index_B.csv':
        df['flow_unit'] = df['flow_unit'].replace('USD 2002', 'USD 2007')
    return df

def classify_processes(df, opt='class'):
    """
    Classify and label WARM db processes via a dictionary (selected via 'opt')
    of regex patterns and ordered label keys stored in warmer/processmapping/
    :param df: pandas dataframe of olca processes
    :param opt: string {'class','fgbg'}
    """
    with open(modulepath/'processmapping'/f'WARMv15_{opt}_regex.yaml', 'r') as f:
        rgx = yaml.safe_load(f)

    is_a_index = 'process_name' in df.columns
    if is_a_index:  # adapt A_index.csv header to olca.processes
        df['name'] = df['process_name']

    cond = []
    labels = list(rgx.keys())
    for key in labels:
        cond.append(df['name'].str.contains('|'.join(rgx[key])))

    df[f'process_{opt}'] = np.select(cond, labels)
    if opt=='class':
        df['process_class'] = pd.Categorical(df['process_class'],
                                             categories=labels, ordered=True)
    if is_a_index: df = df.drop(columns='name')
    return df

def normalize_mtcs(mtx_a, mtx_b):
    """
    Normalize A-matrix (square), dividing elements in each column by values on
    diagonal, such that diagonal elements all equal 1. Then perform element-
    wise division on columns of B-matrix by the same array of diagonal A values.
    :param mtx_a: pd.DataFrame, unlabeled A-matrix
    :param mtx_b: pd.DataFrame, unlabeled b-matrix
    """
    a = mtx_a.to_numpy()
    b = mtx_b.to_numpy()
    a_diag = a.diagonal()
    a_norm = a / a_diag[None,:]  # column-wise division by diagonal elements
    b_norm = b / a_diag[None,:]
    return a_norm, b_norm

def populate_mixer_processes(mtx_a, mtx_b, idx_a):
    """
    Repopulate foreground (fg) "mixer" processes in A and B with sums of the
    tier-1 purchases and elementary flows of their constituent processes.
    WARMv15 mixer processes are identified as those fg processes that lack
    elementary flows and obtain inputs from >=2 other fg processes.
    :param mtx_a: pd.DataFrame, unlabeled A-matrix
    :param mtx_b: pd.DataFrame, unlabeled b-matrix
    :param idx_a: pd.DataFrame, labels for A-matrix rows & cols, B-matrix cols
    """
    all_zero_elem_flows = np.sum(np.abs(mtx_b), axis=0) == 0
    is_fg_prcs = idx_a['process_class'].eq('waste treatment pathway').to_numpy()
    ids_fg = idx_a.loc[is_fg_prcs, 'process_ID']
    mtx_a_diag_zero = mtx_a.copy()  # substitute 0 for 1 in diagonal elements
    np.fill_diagonal(mtx_a_diag_zero, 0)  # note: in-place operation
    mtx_a_diag_zero = append_mtx_IDs(mtx_a_diag_zero, idx_a)
    has_fg_inputs = (mtx_a_diag_zero.query('from_process_ID in @ids_fg')
                                    .sum(axis=0, numeric_only=True)
                                    .ne(0)  # not equal
                                    .to_numpy())
    mixers = (all_zero_elem_flows * is_fg_prcs * has_fg_inputs).astype(int)
    # temp = idx_a.process_name[mixers.astype(bool)]  # check idcs & process_name

    y = np.diag(mixers)
    mtx_b_pop = -1*(mtx_b @ mtx_a @ y) + mtx_b  # '@' equivalent to np.matmul()

    mtx_a_mix = mtx_a * mixers  # multiply mtx_a columns by mixers elements {0, 1}
    mtx_a_pop = (mtx_a @ mtx_a_mix) + mtx_a
    ## NOTES:
    # Many instances of 1 or -1 become 2 or -2 (respectively) due to diag elements
    # Altering mixer cols via '@' matmul instead of iteratively (by column)
    # occasionally returns slightly different values due to
        # differing decimal precision (e.g., element [462, 170], +/- 1 digit),
        # or rounding of very small val (e.g., [90, 213] ~= -7e-18) to 0

    ## By-column inspection:
    # for col in np.where(mixers)[0].tolist():
    # col = 90 # differing row vals: {90: 213, 462: 170}
    # y_a_p = mtx_a[:,col]
    # mtx_a_p = mtx_a @ y_a_p
    # cmpr = mtx_a_pop[:,col] == (mtx_a[:,col] + mtx_a_p)
    # print(f'[{col}]: {cmpr.all()}')
    # if not cmpr.all():
    #     cmpr_cols = np.concatenate(
    #         (mtx_a_pop[:,col].reshape(-1, 1),
    #          (mtx_a[:,col] + mtx_a_p).reshape(-1, 1)),
    #         axis=1)
    #     np.where(cmpr_cols[:,0] != cmpr_cols[:,1])
    return mtx_a_pop, mtx_b_pop

def append_mtx_IDs(mtx_a, idx_a, mtx_b=None, idx_b=None):
    """
    Append process_ID & flow_ID values to rows & columns of matrices
    based on order of entries in each matrix and index df (and thereby file)
    :param mtx_a: pd.DataFrame, unlabeled A-matrix
    :param mtx_b: pd.DataFrame, unlabeled b-matrix
    :param idx_a: pd.DataFrame, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd.DataFrame, labels for B-matrix rows
    """
    mtx_a = pd.DataFrame(mtx_a)
    mtx_a.columns = idx_a['process_ID']
    mtx_a.insert(loc=0, column='from_process_ID', value=idx_a['process_ID'])

    if mtx_b is not None and idx_b is not None:
        mtx_b = pd.DataFrame(mtx_b)
        mtx_b.columns = idx_a['process_ID']  # i.e., why this fxn needs all 4 df's
        mtx_b.insert(loc=0, column='to_flow_ID', value=idx_b['flow_ID'])
        return mtx_a, mtx_b
    else:
        return mtx_a

def melt_mtx(mtx_i, opt):
    """
    Unpivot matrices to long format (i.e., rows are exchanges)
    :param mtx_i: pd.DataFrame, labeled matrix
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
    :param df_a: pd.DataFrame, A-matrix exchanges w/ IDs
    :param df_b: pd.DataFrame, B-matrix exchanges w/ IDs
    :param idx_a: pd.DataFrame, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd.DataFrame, labels for B-matrix rows
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

def query_fg_processes(df_a, df_b, df_subset=None):
    """
    Subset processes in A and B simultaneously.
    For A, keep exchanges targeting foreground (fg) processes with a
        process_class of "waste treatment pathway"
    Optionally, providing df_subset further narrows the subset fg process
        exchanges and overwrites their IDs; df_subset must have cols
        {'process_name', 'process_ID'} of existing fg process names and new IDs
    For B, keep exchanges targeting those fg processes remaining in A.
    :param df_a: pd.DataFrame, A-matrix exchanges w/ all labels
    :param df_b: pd.DataFrame, B-matrix exchanges w/ all labels
    :param df_subset: pd.DataFrame or None, optional processes subset to keep
    """
    df_a_f = df_a.query('to_process_class == "waste treatment pathway"')
    if df_subset is not None:
        dict_keep = dict(zip(df_subset['process_name'],
                             df_subset['process_ID']))
        prcs_keep = list(dict_keep.keys())

        df_a_f = (df_a_f.query('to_process_name in @prcs_keep')
                        .reset_index(drop=True))
        # overwrite foreground IDs in 'to_process_ID' and 'from_process_ID'
        for pfx in ['to', 'from']:
            ids_new = df_a_f[f'{pfx}_process_name'].map(dict_keep)
            if pfx == 'from':
                ids_new = ids_new + '/US'   # formatting for export as FlowID
            df_a_f[f'{pfx}_process_ID'] = \
                np.where(~ids_new.isna(), ids_new, df_a_f[f'{pfx}_process_ID'])

        df_b_f = (df_b.query('from_process_name in @prcs_keep')
                      .reset_index(drop=True))
        df_b_f['from_process_ID'] = df_b_f['from_process_name'].map(dict_keep)
        df_b_f = df_b_f.sort_values(by='from_process_ID')
    else:
        df_b_f = df_b.query('from_process_ID in @df_a_f.to_process_ID')
    return df_a_f, df_b_f

def map_agg(df_b, idx_b):
    """
    Map elementary flows to FEDEFL format, derive a new idx_b w/ temporary IDs,
    and aggregate flows in df_b with newly-overlapping categorical variables
    :param df_b: pd.DataFrame, B-matrix exchanges w/ all labels
    :param idx_b: pd.DataFrame, labels for B-matrix rows
    """
    df = map_warmer_envflows(df_b)
    # assign new flow_ID_temp via FEDEFL-mapped (name + category)
    df['ID_str'] = df['to_flow_name'] + '_' + df['to_flow_category'].fillna('')
    df['to_flow_ID_temp'] = df['ID_str'].apply(
        lambda x: str(uuid.uuid3(uuid.NAMESPACE_OID, x)))
    df = df.drop(columns=['to_flow_ID', 'ID_str'])
    # derive new idx_b df from mapped df_b
    mask = df.columns.intersection('to_' + idx_b.columns).tolist()
    idx = df[['to_flow_ID_temp'] + ['FlowUUID'] + mask].drop_duplicates()
    idx.columns = idx.columns.str.replace('to_','')
    # aggregate df_b flows
    aggcols = df.columns.tolist()
    aggcols.remove('Amount')
    df = df.groupby(aggcols, as_index=False, dropna=False, observed=True).sum()
    return df, idx

def sort_idx_cols(df_idx):
    """
    Multi-level sorting of index df columns, by (1) "process" before "flow",
    and (2) substring order defined by order_cols
    :param df_idx: pd.DataFrame, matrix labels
    """
    cols = df_idx.columns
    p = cols[cols.str.startswith('process')].str.replace('process_','')
    f = cols[cols.str.startswith('flow')].str.replace('flow_','')
    order_cols = pd.Index(['ID', 'ID_temp', 'category', 'class',
                           'location', 'type', 'name', 'unit'])
    p_o = ('process_' + order_cols.intersection(p)).tolist()
    if 'FlowUUID' in cols:
        p_o = ['FlowUUID'] + p_o
    f_o = ('flow_' + order_cols.intersection(f)).tolist()
    df_idx = df_idx[p_o + f_o]
    return df_idx

def sort_idcs(idx_a, idx_b):
    """
    Sort idx_a and idx_b columns, plus idx_a rows to prep for matrix labeling.
    DO NOT CALL BEFORE append_mtx_IDs().
    :param idx_a: pd.DataFrame, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd.DataFrame, labels for B-matrix rows
    """
    # column sorting
    idx_a, idx_b = map(sort_idx_cols, [idx_a, idx_b])
    # row sorting
    idx_a = idx_a.sort_values(by=['process_class','process_name'])
    idx_a['process_class'] = idx_a['process_class'].astype(str)
        # categorical back to string enables manual matrix label adjustments
    idx_a = idx_a.set_index('process_ID')
    return idx_a, idx_b

def pivot_to_labeled_mtcs(df_a, df_b, idx_a, idx_b):
    """
    Convert dfs of labeled exchanges back into matrices
    :param df_a: pd.DataFrame, A-matrix exchanges w/ all labels
    :param df_b: pd.DataFrame, B-matrix exchanges w/ all labels
    :param idx_a: pd.DataFrame, labels for A-matrix rows & cols, B-matrix cols
    :param idx_b: pd.DataFrame, labels for B-matrix rows
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
    mtx_b_i = (idx_a.drop(columns='process_class')
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

def format_tables(df, opt, opt_map):
    """
    Reorder columns and map in new headers, plus drop NA Amount rows and fill
    Location NAs w/ 'US'
    :param df: pd.DataFrame, df_a or df_b
    :param opt: str, {'a', 'b'} switch for df_a or df_b headers
    :param opt_map: str, switch for including 'FlowUUID' header in df_b
    """
    if opt == 'a':
        col_dict = {'to_process_ID': 'ProcessID',
                    'to_process_name': 'ProcessName',
                    'to_flow_unit': 'ProcessUnit',
                    'to_process_location': 'Location',
                    'Amount': 'Amount',
                    'from_process_ID': 'FlowID',
                    'from_process_name': 'Flow',
                    'from_flow_unit': 'FlowUnit',
                    }

        # Drop all 0 exchanges prior to setting diagonal to 0
        df = df.query('Amount != 0').reset_index(drop=True)

        # Find and set mtx_a diagonal exchanges to 0, then invert all signs
        df['Amount'] = -1 * np.where(
            ((df['to_process_ID'] == df['from_process_ID'].str.rstrip('/US')) &
             (df['Amount'] == 1)),
            0, df['Amount'])

    elif opt == 'b':
        col_dict = {'from_process_ID': 'ProcessID',
                    'from_process_name': 'ProcessName',
                    'from_process_category': 'ProcessCategory',
                    'from_process_location': 'Location',
                    'Amount': 'Amount',
                    'to_flow_name': 'Flowable',
                    'to_flow_category': 'Context',
                    'to_flow_unit': 'Unit',
                    }
        if opt_map in {'all', 'fedefl'}:
            col_dict['FlowUUID'] = 'FlowUUID'

    df_mapped = (df.filter(col_dict.keys())
                   .rename(columns=col_dict)
                   .dropna(subset=['Amount'])
                   .fillna({'Location': 'US'}))
    return df_mapped

def get_exchanges(opt_fmt='tables', opt_mixer='pop', opt_map=None,
                  query_fg=True, df_subset=None, mapping=None, controls=None):
    """
    Load WARM baseline scenario matrix files, reshape tables,
    append 'idx' labels, and apply other transformations before returning
    product (A) and elemental (B) flow exchanges in table or matrix format
    :param opt_fmt: str, {'tables', 'matrices'}
    :param opt_mixer: str, switch to enable/disable mixer process flow replacements
    :param opt_map: str, {'all','fedefl','useeio'}
    :param query_fg: bool, True calls query_fg_processes
    :param df_subset: pd.DataFrame, see query_fg_processes
    :param mapping: pd.DataFrame, process mapping file
    :param controls: list, subset of warmer.controls.controls_dict
    """
    if opt_fmt not in {'tables', 'matrices'}:
        print(f'"{opt_fmt}" not a valid format option')
        return None

    a_raw, b_raw, idx_a, idx_b = map(
        read_olca2_mtx, ['A.csv', 'B.csv', 'index_A.csv', 'index_B.csv'])

    idx_a = classify_processes(idx_a)  # assign before populate_mixer_processes
    if opt_map in {'all', 'useeio'}:
        idx_a = classify_processes(idx_a, opt='fgbg')

    mtx_a, mtx_b = normalize_mtcs(a_raw, b_raw)
    if opt_mixer == 'pop':
        mtx_a, mtx_b = populate_mixer_processes(mtx_a, mtx_b, idx_a)
    mtx_a, mtx_b = append_mtx_IDs(mtx_a, idx_a, mtx_b, idx_b)

    df_a, df_b = map(melt_mtx, [mtx_a, mtx_b], ['a', 'b'])
    df_a, df_b = label_exch_dfs(df_a, df_b, idx_a, idx_b)

    # Call elementary and/or product flow controls before mapping steps
    if not controls:
        controls = []
    for c in controls:
        if c in warmer.controls.controls_dict.keys():
            func = getattr(warmer.controls, warmer.controls.controls_dict[c])
            df_a, df_b = func(df_a, df_b)
        else:
            print(f'control {c} does not exist.')

    if query_fg:
        df_a, df_b = query_fg_processes(df_a, df_b, df_subset)
    if opt_map in {'all', 'useeio'}:
        df_a = map_processes(df_a, mapping)
    if opt_map in {'all', 'fedefl'}:
        df_b, idx_b = map_agg(df_b, idx_b)

    if opt_fmt == 'tables':
        df_a, df_b = map(format_tables, [df_a, df_b], ['a','b'], [opt_map, opt_map])
        return df_a, df_b
    elif opt_fmt == 'matrices':
        mtx_a_lab, mtx_b_lab = pivot_to_labeled_mtcs(df_a, df_b, idx_a, idx_b)
        return mtx_a_lab, mtx_b_lab

if __name__ == '__main__':
    df_a, df_b = get_exchanges()

    # # Generate processmapping.csv
    # idx_a = read_olca2_mtx('index_A.csv')
    # newcols = ['MatchCondition','ConversionFactor',
    #             'TargetListName','TargetProcessName','TargetUnit','LastUpdated']
    # idx_a = idx_a.reindex(columns = idx_a.columns.tolist() + newcols)
    # idx_a.to_csv(modulepath/'processmapping'/'processmapping.csv', index=False)
