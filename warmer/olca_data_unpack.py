# https://greendelta.github.io/olca-ipc.py/olca/ipc.html

# import time

import olca
import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path

modulepath = Path(__file__).parent
datapath = modulepath/'data'

## flatten nested olca data classes
def unpack_olca_tuple_to_df(iterable, cols = ['all']):
    """
    Extract olca object __dict__ values by mapping vars() over collection
    :param iterable: obj (e.g., tuple, list, pd.Series) containing olca objs
    :param cols: optional, list specifying column(s) to return
    """
    # [next] identify nan elements before map()
    df = pd.DataFrame(map(vars, iterable))
    if cols != ['all']: df = df[cols]

    # # extract column of objects and rename field
    # temp = (unpack_olca_tuple_to_df(df_flow['category'], ['name','category_path'])
    #         .rename(columns={'name':'category_name'}))
    return df

# def unpack_olca_dict_col(col):
#     """
#     Extract olca object nested data from df columns
#     :param col: pandas dataframe column (i.e., numpy array)
#     """
#     if col.dtype == 'O':
#         if is_col_olca(col):

#             df = pd.DataFrame(vars(col), index=[0])
#             ## still working through this
#         else:
#             print(f'{col.name} contains objects (e.g., lists of olca), \
#                   but not individual olca objects')
#     return df

def unpack_list_simple_col(col):
    """
    Attempt to unpack column single-element lists; return warning if >1 element
    :param col: pandas dataframe column (i.e., numpy array)
    """
    if is_list_col(col):
        if all(col.map(len)==1):  # all length-1 lists
            col = col.apply(lambda x: x[0])  # just grab the single elements
            return col
        else:
            print(f'{col.name} lists have >1 element each; extract as table')
    return col

def is_list_col(col):
    """
    Confirm (T/F) whether column consists entirely of list objects.
    If expecting T but returns F check for None's, etc.
    """
    tf = all(col.apply(lambda x: isinstance(x, list)))
    return tf

def is_col_olca(col):
    """
    Identify columns of olca schema objects (dicts), which contain nested data
    :param col: pandas dataframe column (i.e., numpy array)
    """
    tf = all(col.apply(lambda x: x.__module__ == 'olca.schema'))
    # if all(col.apply(lambda x: type(x).__bases__[0] == olca.schema.Entity)):
    if not tf: print('Column does not uniformly contain olca schema objects')
    return tf

def unpack_exchanges(df_proc):
    """
    Extract exchange data from a dataframe of olca data classes
    """
    exch_list = []
    for index, row in df_proc.iterrows():
        for ex in row['exchanges']:
            y = ex.to_json()
            exc_dict = exch_dict(row, y)
            exch_list.append(exc_dict)
    df = pd.DataFrame(exch_list)
    return df


def exch_dict(proc, exch):
    process = proc['name']
    try: category = proc['category'].name
    except AttributeError: category = ''
    flow = exch['flow']['name']
    flow_type = exch['flow']['flowType']
    amount = exch['amount']
    return {'process': process, 'category': category,
            'flow': flow, 'flow_type': flow_type,
            'amount': amount}


def classify_prcs(df):
    """
    Classify WARM db processes as foreground, background_map, or background_deep
    :param df: pandas dataframe of olca processes
    """   
    with open(modulepath/'processmapping'/'WARMv15_prcs_regex.yaml', 'r') as f:
        rgx = yaml.safe_load(f)
    
    a_index = 'process name' in df.columns
    if a_index:  # adapt A_index.csv header to olca.processes
        df['name'] = df['process name']   
        
    df['fg'] = df['name'].str.contains('|'.join(rgx['foreground']))    
    df['bg_map'] = df['name'].str.contains('|'.join(rgx['background_map']))    
    df['bg_deep'] = df['name'].str.contains('|'.join(rgx['background_deep']))    
    
    df['prcs_class'] = np.select(
        [df['fg'], df['bg_map'], df['bg_deep']],  # bool cols as cond
        ['foreground', 'background_map', 'background_deep'])  # label vals
    
    df = df.drop(columns=['fg','bg_map','bg_deep'])
    if a_index: df = df.drop(columns='name')
    return df


if __name__ == "__main__":  # revert to "==" later
    get_data = False
    get_params = False
    
    if get_data:  # get olca data
        # Match to IPC Server value: in openLCA -> Tools > Developer Tools > IPC Server
        client = olca.Client(8080)
    
        # print(vars(olca.schema))  # view all exportable data
        # from WARMv15 db: 799 flows, 2140 processes, 69635 parameters
    
        # for p in processes[:5]: print(f"{p.name} - {p.flow_type}")
        # f = client.get('Flow', 'Carbon dioxide')
        # temp = tuple(client.get_all(olca.ProductSystem))
        
        flows = tuple(client.get_all(olca.Flow)) # wrap in tuple to make it iterable
        pickle.dump(flows, open(datapath/'flows.pickle', 'wb'))
    
        processes = tuple(client.get_all(olca.Process))
        pickle.dump(processes, open(datapath/'processes.pickle', 'wb'))
    
        if get_params: # takes 16min+ to get these
            parameters = tuple(client.get_all(olca.Parameter))
            pickle.dump(parameters, open(datapath/'parameters.pickle', 'wb'))
    
    
    if not get_data:  # retrieve from pickle
        # flows = pickle.load(open(datapath/'flows.pickle', 'rb'))
        processes = pickle.load(open(datapath/'processes.pickle', 'rb'))
        # parameters = pickle.load(open(datapath/'parameters.pickle', 'rb'))
    
    # for WARM db, olca.X classes that have a get_all() method...
        # cannot retrieve: AllocationFactor, FlowType, FlowMap, FlowMapEntry, FlowMapRef,
            # FlowPropertyFactor, FlowPropertyType, FlowType, ProcessLink
        # can retrieve: Category, FlowProperty, ModelType, ProductSystem, UnitGroup

    ## convert tuples to dfs
    # drop flow rows (513, 514) b/c irregular formatting & lack of info
    # flows_exclude = ['Baseline scenario','Alternative scenario']  # rows (513, 514)
    # df_flow = (unpack_olca_tuple_to_df(flows)
    #             .query('name not in @flows_exclude')
    #             .reset_index())

    df_prcs = unpack_olca_tuple_to_df(processes)  # plenty of columns to expand here too
    df_exch = unpack_exchanges(df_prcs)
    
    df_prcs_class = classify_prcs(df_prcs)

    if get_params: df_param = unpack_olca_tuple_to_df(parameters)

    olca_obj = df_prcs.exchanges[0][0]

    unpack_test = unpack_olca_tuple_to_df(df_prcs.process_type)


if False:  # set True for script development

    dir(olca_obj)
    vars(olca_obj)
    # vars(olca_obj).__iter__
    # vars(olca_obj) == olca_obj.__dict__
    # iter(vars(prcs_smpl.exchanges[1][0]))  # can't construct df from just this
    temp = pd.DataFrame(vars(olca_obj), index=[0])

    # function to convert each to a csv?
    df_prcs.name[1]  # combustion
    df_prcs.exchanges[1]  # 15 exchanges
    vars(df_prcs.exchanges[1][0])  # first exchange variables
    vars(vars(df_prcs.exchanges[1][0])['flow'])
    vars(vars(vars(df_prcs.exchanges[1][0])['flow'])['flow_type'])
        # extraneous; elementary vs. product flows are easy to spot
    vars(vars(df_prcs.exchanges[1][0])['flow_property'])
        # need ['name'] only to reconstruct flat olca export
    vars(df_prcs.exchanges[1][14])  # input: T/F determines inputs vs outputs


    ###########################################################################
    # unpack flow_properties' single-element list values
    df_flow.flow_properties = unpack_list_simple_col(df_flow.flow_properties)
    df_flow['flow_properties'] = df_flow['flow_properties'].apply(lambda x: x[0])

    # another approach to expanding olca obj columns
    temp = (df_flow['category'].apply(lambda x: pd.Series(vars(x)))
            .loc[:,['name','category_path']]
            .rename(columns={'name':'category_name'}))

    # extract column of objects and rename field
    a = (unpack_olca_tuple_to_df(df_flow['category'], ['name','category_path'])
            .rename(columns={'name':'category_name'}))
    # unit test: # df_flow['category'][i].__dict__['name'] == a['category_name'][i]
    b = pd.concat([df_flow,a],axis=1)
    c = df_flow.copy(deep=True).join(a, how='inner')  # unit test: len() same before/after
    d = df_flow.join(a, how='outer')  # unit test: len() same before/after
    e = d.eq(c)
    f = d.fillna(0).eq(c.fillna(0))
    
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))
