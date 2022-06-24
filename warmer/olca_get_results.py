# -*- coding: utf-8 -*-
"""
Import WARM OpenLCA model's technology & intervention matrices
Last Updated: 2022.06.23
"""
import pickle

import olca
import pandas as pd

from pathlib import Path

modulepath = Path(__file__).parent
datapath = modulepath/'data'

def unpack_dict_col(df, col):
    """
    Convert df column of single-level dictionaries to multiple columns
    :param df: pandas dataframe
    :param col: str, target column header
    """
    df = pd.concat([df.drop(columns=col),
                    pd.json_normalize(df[col])],
                   axis='columns')

    ## Two dict flattening options:
        # pd.json_normalize(df[col])
        # df[col].apply(pd.Series)
    # def kwargs_view(**kwargs):
        # print(kwargs)
    # kwargs_view(**df)
    # df2 = df.assign(**df[col].apply(pd.Series)).drop(columns=col)
    # df3 = df.assign(**pd.json_normalize(df[col])).drop(columns=col)
    # pd.util.testing.assert_frame_equal(df2, df3)  # True
    # pd.util.testing.assert_frame_equal(df2, df)   # False; duplicate '@type' col preserved
    return df

if __name__ == "__main__":  # revert to "==" later
    # Match to IPC Server value: in openLCA -> Tools > Developer Tools > IPC Server
    client = olca.Client(8080)
    setup = olca.CalculationSetup()
    ## Options: .SIMPLE_CALCULATION, .CONTRIBUTION_ANALYSIS, .UPSTREAM_ANALYSIS
    # setup.calculation_type = olca.CalculationType.SIMPLE_CALCULATION
        # CANNOT retreive lcia process/flow contributions if using this method
    setup.calculation_type = olca.CalculationType.CONTRIBUTION_ANALYSIS
    # setup.calculation_type = olca.CalculationType.UPSTREAM_ANALYSIS

    setup.product_system = olca.ref(
        olca.ProductSystem,
        'c29d0465-53b3-4ee9-a3e1-c448db457aac'    # Grains AD
        # '2ce732ce-b508-498b-8322-63eba3278a81'    # Grains composting
        # 'ecc77138-09d5-31ea-8f63-531eba66b991',   # Baseline (doesn't work...)
    )
    setup.impact_method = olca.ref(
        olca.ImpactMethod,
        'bf2194d5-e72d-482c-8ade-f576d78dc5fc'  # WARM (MTCO2E)
        # 'a10dd50b-68f5-4899-b1a8-b82a048947fe'  # WARM (Energy)
    )
    setup.amount = 1.0
    result = client.calculate(setup)
    # print(f'type(result): {type(result)}')

    # client.lcia_flow_contributions() gives amounts by each GHG but not by processes
    result_lcia = client.lcia_process_contributions(
        result,
        client.lcia(result)[0].impact_category)
    lcia_dict = [x.to_json() for x in result_lcia]
    df_lcia = (pd.DataFrame(lcia_dict)
                 .pipe(unpack_dict_col, 'item'))

    # TODO: classify_processes() on df_prcs; query only 'fg' processes; keep only uuid & names
    # TODO: (script) iterate over filtered df_prcs to create prod-sys's w/i fresh warm_v15 db
    # TODO: (script) iterate over new db processes to calcualte & retreive contributions
    processes = pickle.load(open(datapath/'processes.pickle', 'rb'))
    # processes = tuple(client.get_all(olca.Process))
    # pickle.dump(processes, open(datapath/'processes.pickle', 'wb'))
    df_prcs = pd.DataFrame(map(vars, processes))

    # result_dict = result.to_json()
    # df_impacts = (pd.DataFrame.from_dict(result_dict['impactResults'])
    #                 .pipe(unpack_dict_col, 'impactCategory'))
    # df_flows = (pd.DataFrame.from_dict(result_dict['flowResults'])
    #               .pipe(unpack_dict_col, 'flow'))


    client.dispose(result)  # helps avoid memory leaks



    # print(vars(olca.schema))  # view all exportable data
    # from WARMv15 db: 799 flows, 2140 processes, 69635 parameters

    # for p in processes[:5]: print(f"{p.name} - {p.flow_type}")
    # f = client.get('Flow', 'Carbon dioxide')
    # temp = tuple(client.get_all(olca.ProductSystem))

    # flows = tuple(client.get_all(olca.Flow)) # wrap in tuple to make it iterable
    # pickle.dump(flows, open(datapath/'flows.pickle', 'wb'))

    # processes = tuple(client.get_all(olca.Process))
    # pickle.dump(processes, open(datapath/'processes.pickle', 'wb'))

    # get_params = False
    # if get_params: # takes 16min+ to get these
    #     parameters = tuple(client.get_all(olca.Parameter))
        # pickle.dump(parameters, open(datapath/'parameters.pickle', 'wb'))