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
    # Then incorporate avoided fertilizer product flows
    df_a = insert_FW_AD_fert_flows(df_a)
    return df_a, df_b

def insert_FW_AD_fert_flows(df_a):
    """
    Create and insert avoided product flows for N and P fertilizers
    derived from the land application of digestate, produced by
    "Anaerobic digestion of Food Waste; Dry digestion, Cured"
    """
    # prepare new fert flow/process values (ordered; N then P)
    # Note: avoided products are defined as negative inputs in the A-matrix
    fert_offset_masses = [x for x in get_FW_AD_fert_offsets()]
    N_P = ['N', 'P']
    ferts = ['Anhydrous Ammonia', 'Triple Superphosphate (45% P2O5)']
    vals_fert_from = {
        'from_process_ID':      [f'fw-ad-fert-offset-{x}-prcs' for x in N_P],
        'from_flow_ID':         [f'fw-ad-fert-offset-{x}-flow' for x in N_P],
        'from_process_name':    [f'Fertilizer production, {x}' for x in ferts],
        'from_flow_name':       [f'Fertilizer, {x}' for x in ferts],
        'Amount':               fert_offset_masses,}
        # 'from_process_name': [(f'Product manufacturing, from {x}, '
        #                         'using 100% virgin inputs') for x in ferts],
        # 'from_flow_name': [(f'Product manufactured, from {x}, '
        #                      'using 100% virgin inputs') for x in ferts],
    # get FW AD "to_.*" cols b/c of above Note
    prcs_FW_AD = 'Anaerobic digestion of Food Waste; Dry digestion, Cured'
    row_fert_to = (df_a.query('to_process_name == @prcs_FW_AD')
                       .filter(regex='^to_')
                       .drop_duplicates())
    df_fert_to = (pd.concat([row_fert_to]*len(fert_offset_masses))
                    .reset_index(drop=True))
    # borrow prcs_fert_mimic's "from_.*" cols for new fertilizer processes
    prcs_fert_mimic = 'Product manufacturing, from Ag Gypsum, using 100% virgin inputs'
    row_fert_from = (df_a.query('from_process_name == @prcs_fert_mimic')
                         .filter(regex='^from_')
                         .drop_duplicates())
    # then overwrite select cols w/ new fertilizer values
    df_fert_from = (pd.concat([row_fert_from]*len(fert_offset_masses))
                      .reset_index(drop=True)
                      .assign(**vals_fert_from))

    df_fert = pd.concat([df_fert_from, df_fert_to], axis='columns')
    df_a = pd.concat([df_a, df_fert])
    return df_a

def get_FW_AD_fert_offsets():
    """
    Convert the N and P fertilizer offset parameters (Mass_*_Offset; elemental masses)
    from WARMv15's "Anaerobic digestion of Food Waste; Dry digestion, Cured" process into
    Anhydrous Ammonia (NH3) and Triple Superphosphate (45% P2O5) equivalent avoided product flows.
    These fertilizer equivalents are chosen for elemental N & P purity, and to align with
    available price data from USDA ERS: https://www.ers.usda.gov/data-products/fertilizer-use-and-price/
    """
    sh_tn_to_kg = 907.18  # kg per short ton
    mass_N_offset = 1.19542224 / sh_tn_to_kg  # kg elemental N in fertilizer / kg "Food Waste, digested"
    mass_P_offset = 1.41780000 / sh_tn_to_kg  # kg elemental P in fertilizer / kg "Food Waste, digested"
    # atomic masses
    mass_a = {'H': 1.008,
              'N': 14.007,
              'O': 15.999,
              'P': 30.974}
    # mass fractions for elemental --> molecular fertilizer conversions
    pct_mass_N_in_AA = mass_a['N'] / (mass_a['N'] + 3*mass_a['H'])      # kg N / kg NH3
    pct_mass_P_in_P2O5 = mass_a['P'] / (2*mass_a['P'] + 5*mass_a['O'])  # kg P / kg P2O5
    pct_mass_P2O5_in_TSP = 0.45                                         # kg P2O5 / kg TSP
    pct_mass_P_in_TSP = pct_mass_P_in_P2O5 * pct_mass_P2O5_in_TSP       # kg P / kg TSP
    # final masses of AA and TSP
    mass_AA_offset = mass_N_offset / pct_mass_N_in_AA       # kg NH3 / kg "Food Waste, digested"
    mass_TSP_offset = mass_N_offset / pct_mass_P_in_TSP     # kg TSP / kg "Food Waste, digested"

    # TODO: extract Mass_*_Offset params via olca_get_results.py, then use
        # this approach for all AD processes
    # TODO: also extract the Leachate_treatment & Treated_freshwater params;
        # confirm if 0-valued across all processes (manually inspected for FW_AD);
        # if not, rework CO2 emissions flows (can't simply drop; need proportional reduction)
    return [mass_AA_offset, mass_TSP_offset]
