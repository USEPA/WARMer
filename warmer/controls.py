# controls.py (WARMer)

"""
Functions to enable modifications to WARM data
"""
import pandas as pd

controls_dict = {'electricity': 'control_displaced_electricity_emissions',
                 'forest': 'remove_forest_carbon_storage',
                 'fertilizer': 'control_avoided_fertilizer_emissions',
                 }

def control_displaced_electricity_emissions(df_a, df_b):

    e_gen = 'Electricity generation, at grid, National'
    ghg_u = 'GHGs, unspecified'

    ## TODO: derive 1.376E+03 kWh/MTCO2E factor from e_gen elementary flows
    # plus factors from new olca_get_results.get_impact_factors fxn
    # df_ghgs_to_e = df_b.query('from_process_name == @e_gen and '
    #                           'Amount != 0')

    # For combustion processes w/ a "GHGs, unspecified" output flow and no
    # elec. output (avoided product), substitute the former for the latter
    df_b_e = df_b.query('from_process_name.str.contains("combustion") and '
                        'to_flow_name == @ghg_u and '
                        # 'to_flow_name == "GHGs, unspecified"'
                        # 'Amount < 0'
                        'Amount != 0')
    prcs_ghg_u = set(df_b_e['from_process_ID'])
    # NOTE: elec. gen. outputs are given as negative inputs w/i olca mtx exports
    df_a_e = df_a.query('from_process_name == @e_gen and '
                        'Amount != 0')
    prcs_e_gen = set(df_a_e['to_process_ID'])

    df_target = df_a.query('to_process_ID in @prcs_ghg_u and '
                           'to_process_ID not in @prcs_e_gen and '
                           'Amount != 0')
    prcs_target = set(df_target['to_process_ID'])

    ## TODO: clarify sign handling here and in olca_warm_matrix_io
    # Construct new e_gen exchanges for target processes & concat into df_a
    # Must invert sign when converting output flow to avoided input
    flows_e = dict(zip(df_b_e['from_process_ID'],
                       df_b_e['Amount']*1376*-1))

    df_elec_to = (df_a.query('to_process_ID in @prcs_target and '
                             'Amount != 0')
                      .filter(regex='^to_')
                      .drop_duplicates()
                      .reset_index(drop=True))
    df_elec_from = (df_a.query('from_process_name == @e_gen and '
                                'Amount != 0')
                        .filter(regex='^from_')
                        .drop_duplicates())
    df_elec_from = (pd.concat([df_elec_from]*len(df_elec_to))
                      .reset_index(drop=True))

    df_elec = pd.concat([df_elec_from, df_elec_to], axis='columns')
    df_elec['Amount'] = df_elec['to_process_ID'].map(flows_e)
    df_a = pd.concat([df_a, df_elec])

    # Drop "GHGs, unspecified" and "Energy, unspecified" (avoided consumption)
    # flows from target processes
    unspec = set([ghg_u, 'Energy, unspecified'])
    df_b = df_b.query('not (from_process_ID in @prcs_target and '
                      'to_flow_name in @unspec)')
    return df_a, df_b

def remove_forest_carbon_storage(df_a, df_b):
    # Remove Carbon/resource/biotic
    df_b = df_b.query(
        'not ('
        'to_flow_name == "Carbon" and '
        'to_flow_category.str.contains("resource/biotic") and '
        'Amount != 0)')
    return df_a, df_b

def control_avoided_fertilizer_emissions(df_a, df_b):
    # Remove carbon dioxide from Anaerobic digestion
    df_b = df_b.query(
        'not ('
        'to_flow_name == "Carbon dioxide" and '
        'to_flow_category.str.contains("air/unspecified") and '
        'from_process_name.str.contains("Anaerobic digestion") and '
        'Amount != 0)')
    ## TODO Add negative fertilizer output as avoided product
    # Extract Mass_N_Offset & Mass_P_Offset (or *._Applied?) parameters
    # via warmer.olca_get_results, construct avoided fertilizer product flows,
    # and merge these into df_a
    return df_a, df_b
