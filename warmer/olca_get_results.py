# -*- coding: utf-8 -*-
"""
Extract LCIA contribution trees for select WARMv15 OpenLCA processes
Last Updated: 2022.07.13
"""
import pickle
from pathlib import Path

import numpy as np
import olca
import pandas as pd

from warmer.olca_warm_matrix_io import classify_processes

modulepath = Path(__file__).parent
datapath = modulepath/'data'

def get_dump_load(data_type, get=False, client=None, get_opt='all',
                  dump=False, load=False):
    """
    Get olca data objects from anfrom pathlib import Path open olca.Client() object, dump them to a pickle,
    and/or load them from a pickle via olca.<data_type> attribute strings
    :param data_type: str, olca schema attribute
    :param client: object, connected olca.Client()
    :param get: bool, use client.get_{get_opt} method to retreive data?
    :param get_opt: str, 'all' (default), or 'descriptors' for less info
    :param dump: bool, dump data to pickle?
    :param load: bool, load data from pickle?
    :return: list of olca.schema.Ref objects
    """
    # get_opt='descriptors' returns just {id, name, categoryPath} for data_type='Process'
    # takes 16min+ to get olca.Parameter
    if get:
        data = list(
                getattr(client, f'get_{get_opt}')(
                    getattr(olca, data_type)))
    # TODO: store/access via appdirs
    filename = f'WARMv15_olca_{data_type.lower()}_{get_opt}.pickle'
    if load:
        with open(datapath/filename, 'rb') as f:
            data = pickle.load(f)
    if dump:
        with open(datapath/filename, 'wb') as f:
            pickle.dump(data, f)

    # Note: pd.testing.assert_frame_equal() is not True for otherwise identical
    # df's w/ cols of olca objects b/c hashes are re-assigned when loaded/generated
    return data

def calc_lcia_cntbs(id_prodsys, client, impact='WARM (MTCO2E)'):
    """
    For a given product system, calculate LCIA results (default imapct GHGs) and
    return a list of process contributions
    :param id_prodsys: str, product system UUID
    :param client: object, connected olca.Client()
    :return: list of olca.ipc.ContributionItem objects
    """
    id_impact_method = {
        'WARM (MTCO2E)': 'bf2194d5-e72d-482c-8ade-f576d78dc5fc',
        'WARM (Energy)': 'a10dd50b-68f5-4899-b1a8-b82a048947fe',
        'WARM (Jobs)': '70514047-a8d7-48cf-a974-697b3fefc87a',
        'WARM (Labor Hours)': '4c5c02ad-07bf-4053-b95f-d5b61db3b033',
        'WARM (MTCE)': '238efe76-cfda-4110-a6b7-95a3fea10466',
        'WARM (Taxes)': '9d9ae107-6190-4221-a601-526a464ee4fb',
        'WARM (Wages)': '13d9c770-934d-403a-9496-a933f8a9a017',}
    setup = olca.CalculationSetup()
    setup.calculation_type = olca.CalculationType.CONTRIBUTION_ANALYSIS
        # Options: .SIMPLE_CALCULATION, .CONTRIBUTION_ANALYSIS, .UPSTREAM_ANALYSIS
        # Cannot get lcia process/flow contributions if using SIMPLE_CALCULATION
    setup.product_system = olca.ref(olca.ProductSystem, id_prodsys)
    setup.impact_method = olca.ref(olca.ImpactMethod, id_impact_method[impact])
    setup.amount = 1.0
    result = client.calculate(setup)
    # print(f'type(result): {type(result)}')  # always SimpleResult
    result_ctb = client.lcia_process_contributions(
        result,
        client.lcia(result)[0].impact_category)
    # .lcia_flow_contributions() gives amounts by each GHG rather than by process
    client.dispose(result)  # helps avoid memory leaks
    return result_ctb

def unpack_dict_col(df, col):
    """
    Convert df column of single-level dictionaries to multiple columns;
    if list of dictionaries, flatten or explode first
    :param df: pandas dataframe
    :param col: str, target column header
    """
    is_list_element = df[col].apply(lambda x: isinstance(x, list))
    if all(is_list_element):
        if all(df[col].map(len)==1):
            df[col] = df[col].apply(lambda x: x[0])
        else:
            # df[col].map(len).value_counts()
            print(f'INFO: {col} list elements have !=1 elements each')
            df = df.explode(col).reset_index(drop=True)
    elif any(is_list_element):
        print('Mixture of list and non-list elements; inspect df')
        return None

    df = pd.concat([df.drop(columns=col),
                    (pd.json_normalize(df[col], sep='_') #, record_prefix=col)
                       .add_prefix(f'{col}_'))],
                   axis='columns')

    ## Two dict flattening approaches:
        # pd.json_normalize(df[col])
        # df[col].apply(pd.Series)
    # def kwargs_view(**kwargs):
        # print(kwargs)
    # kwargs_view(**df)
    # df2 = df.assign(**df[col].apply(pd.Series)).drop(columns=col)
    # df3 = df.assign(**pd.json_normalize(df[col])).drop(columns=col)
    # pd.testing.assert_frame_equal(df2, df3)  # True
    # pd.testing.assert_frame_equal(df2, df)   # False; duplicate '@type' col preserved
    return df


def convert_lcia_units(df, from_unit, to_unit, conv_factor):
    """Convert units in the indicators based on conversion factor."""
    df['impactFactors_value'] = np.where(
        df['impactFactors_unit_name'] == from_unit,
        df['impactFactors_value'] * conv_factor,
        df['impactFactors_value'])

    df['impactFactors_unit_name'] = np.where(
        df['impactFactors_unit_name'] == from_unit, to_unit,
        df['impactFactors_unit_name'])
    return df

if __name__ == "__main__":  # revert to "==" later
    # Match to IPC Server value: in openLCA -> Tools > Developer Tools > IPC Server
    client = olca.Client(8080)

    # %% Impact Methods & Factors
    impact_methods = get_dump_load('ImpactMethod', client=client,
                                   get=True, get_opt='all')
    df_imth = (pd.DataFrame([x.to_json() for x in impact_methods])
                 .drop(columns=['@id', '@type', 'lastChange'])
                 .pipe(unpack_dict_col, 'impactCategories')
                 .filter(['impactCategories_@id', 'name', 'description'])
                 .rename(columns={'impactCategories_@id': '@id',
                                  'name': 'method_name',
                                  'description': 'method_description'}))
    # 'ImpactFactor' raises exception; using 'ImpactCategory' instead:
    impact_cats = get_dump_load('ImpactCategory', client=client,
                                get=True, get_opt='all')
    df_icat = (pd.DataFrame([x.to_json() for x in impact_cats])
                 .pipe(unpack_dict_col, 'impactFactors')
                 .replace('', np.nan)
                 .dropna(axis='columns', how='all')
                 # fix WARMv15 economic flow units: USD 2002 should be 2007
                 .replace('USD 2002', 'USD 2007')
                     # TODO: reinstall fedefl@develop to accomodate USD 2007
                 .merge(df_imth, how='right', on='@id'))

    # Replace incorrect category description for Labor Hours
    df_icat.loc[df_icat['method_name'] == 'WARM (Labor Hours)',
                ('name', 'referenceUnitName')] = ('Labor Hours', 'Labor Hours')

    # Replace incorrect units for MTCE
    df_icat.loc[df_icat['method_name'] == 'WARM (MTCE)',
                ('name', 'referenceUnitName')] = ('GHGs (MTCE)', 'MTCE')

    df_icat['method_name'] = 'WARMv15'

    # convert units to align with flow mapping used by lcia formatter
    df_icat = convert_lcia_units(df_icat, 't', 'kg', .001)
    df_icat = convert_lcia_units(df_icat, 'btu', 'MJ', 947.8)

    dict_lciafmt_cols = {
        'method_name': 'Method',
        'name': 'Indicator',    # only valid when excluding MTCE method
        'referenceUnitName': 'Indicator unit',
        'impactFactors_flow_name': 'Flowable',
        'impactFactors_flow_@id': 'Flow UUID',
        'impactFactors_flow_categoryPath': 'Context',
        'impactFactors_value': 'Characterization Factor',
        'impactFactors_unit_name': 'Unit'}

    df_icat_expt = (df_icat.filter(dict_lciafmt_cols.keys())
                           .rename(columns=dict_lciafmt_cols)
                           .to_csv(datapath/'WARMv15_lcia_methods.csv',
                                   index=False))


    # from warmer.mapping import map_warmer_envflows
    # field_dict = {'SourceName': '',
    #               'FlowableName': 'impactFactors_flow_name',
    #               'FlowableUnit': 'impactFactors_unit_name',
    #               'FlowableContext': 'impactFactors_flow_categoryPath',
    #               'FlowableQuantity': 'impactFactors_value',
    #               'UUID': 'impactFactors_flow_@id'}
    # temp = map_warmer_envflows(df_icat, field_dict=field_dict)
        # FIXME: returns `TypeError: unhashable type: 'list'`


    # %% LCIA Contribution Trees script
    # Get/load WARMv15 olca db process data:
    # processes = get_dump_load('Process', client=client, get=True, get_opt='descriptors', dump=True)
    processes = get_dump_load('Process', get_opt='descriptors', load=True)

    # Choose waste treatment processes for which to generate product systems
    prcs_keep = (pd.read_csv(datapath/'model_1_processes.csv',
                             header=None, usecols=[1])
                   .squeeze()
                   .to_list())
    df_prcs = (pd.DataFrame([p.to_json() for p in processes])
                 .query('name in @prcs_keep')
                 .reset_index(drop=True))
        # Note: using pd.DataFrame(map(vars, processes)) gives additional
        # metadata (not needed here) but doesn't unpack all object-cols to dicts

    # For each foreground process, get a prod-sys

    # Tip: make & open a copy of warm_v15_... before setting to True
    create_prodsys_collection = False
    if create_prodsys_collection:
        # Iterate over process id's to create a collection of prod-sys's
        [client.create_product_system(process_id=x) for x in df_prcs['@id']]
        prodsys = get_dump_load('ProductSystem', client=client,
                                get_opt='all', get=True, dump=True)
    else:
        prodsys = get_dump_load('ProductSystem', get_opt='all', load=True)

    df_psys = (pd.DataFrame([p.to_json() for p in prodsys])
                 .query('name in @prcs_keep')  # in case pickles have extra entries
                 # .query('~name.str.contains("Scenario")', engine='python')
                 .reset_index(drop=True))

    # Iterate over prod-sys's to calcualte and aggregate process contributions
    lcia_cntb = [calc_lcia_cntbs(p, client) for p in df_psys['@id']]
    lcia_dict = [[r.to_json() for r in psys] for psys in lcia_cntb]
    df_lcia = (pd.DataFrame({'prodsys_name': df_psys['name'],
                            'result': lcia_dict})
                 .explode('result')
                 .reset_index(drop=True)
                 .pipe(unpack_dict_col, 'result')
                 # TODO: affirm subsequent methods work w/ new col name prefix
                 .drop(columns=['@type', 'share', 'rest', 'item_@type']))
                 # .filter(regex='.*(?<!type)$'))  # drop '@type' cols
                 # .filter(items=['amount','unit','name'])  # [later] decide what we want
    df_lcia.columns = df_lcia.columns.str.replace('item_','process_')
    df_lcia = classify_processes(df_lcia, 'fgbg')  # uses 'process_name' col
    df_AlCan = df_lcia.query('prodsys_name=="MSW recycling of Aluminum Cans"') \
                [['amount','process_name','process_fgbg','unit',]]

    mask = df_lcia.query('process_fgbg == "background_deep"')['prodsys_name'].tolist()
    df_lcia_hasDeep = df_lcia.query('prodsys_name in @mask')


    # client.update(olca.ProductSystem, df_prcs['@id'][1])
    # raise SystemExit(0)  # stop script execution

    # result_dict = result.to_json()
    # df_impacts = (pd.DataFrame.from_dict(result_dict['impactResults'])
    #                 .pipe(unpack_dict_col, 'impactCategory'))
    # df_flows = (pd.DataFrame.from_dict(result_dict['flowResults'])
    #               .pipe(unpack_dict_col, 'flow'))


    # %% [old] Inspection & Flow + Param retreival
    # for p in processes[:5]: print(f"{p.name} - {p.flow_type}")
    # f = client.get('Flow', 'Carbon dioxide')
    # temp = get_dump_load('ProductSystem', client=client, get=True)

    # print(vars(olca.schema))  # view all exportable data
    # from WARMv15 db: 799 flows, 2140 processes, 69635 parameters

    # processes = get_dump_load('Process', client=client, get=True, dump=True)
    # flows = get_dump_load('Flow', client=client, get=True, dump=True)

    # get_params = False
    # if get_params: # takes 16min+ to get these
        # parameters = get_dump_load('Parameter', client=client, get=True, dump=True)
