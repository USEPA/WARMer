# controls.py (WARMer)

"""
Functions to enable modifications to WARM data
"""

controls_dict = {'electricity': 'control_displaced_electricity_emissions',
                 'forest': 'remove_forest_carbon_storage',
                 'fertilizer': 'control_avoided_fertilizer_emissions',
                 }


def control_displaced_electricity_emissions(df_a, df_b):
    # Remove GHGs Unspecified from select processes
    b = df_b.query('from_process_name.str.contains("combustion") and '
                   'to_flow_name == "GHGs, unspecified"', engine='python')
    b = b.query('Amount < 0')
    df_b = df_b.drop(b.index)
    
    # TODO Add negative electricity output (avoided product) 1.376E+03 kWh/MTCO2E
    elec = df_a.query('from_process_name == "Electricity generation, at grid, National"')
    
    return df_a, df_b
    
def remove_forest_carbon_storage(df_a, df_b):
    # Remove Carbon/resource/biotic
    b = df_b.query('to_flow_name == "Carbon" and '
                   'to_flow_category.str.contains("resource/biotic")', engine='python')
    b = b.query('Amount < 0')
    df_b = df_b.drop(b.index)
    
    return df_a, df_b
    
def control_avoided_fertilizer_emissions(df_a, df_b):
    # Remove carbon dioxide from Anaerobic digestion
    b = df_b.query('to_flow_name == "Carbon dioxide" and '
                   'to_flow_category.str.contains("air/unspecified") and '
                   'from_process_name.str.contains("Anaerobic digestion")',
                   engine='python')
    b = b.query('Amount < 0')
    df_b = df_b.drop(b.index)
    
    # TODO Add negative fertilizer output as avoided product

    return df_a, df_b
