# mapping.py (WARMer)

"""
Functions related to mapping flows and processes
"""

from esupy.mapping import apply_flow_mapping

def map_warmer_envflows(df):

    # Align warmer fields with mapping, value represents field in target df
    field_dict = {'SourceName': '',
                  'FlowableName':'to_flow_name',
                  'FlowableUnit':'to_flow_unit',
                  'FlowableContext':'to_flow_category',
                  'FlowableQuantity':'Amount',
                  'UUID':'FlowUUID'}

    df = apply_flow_mapping(df, 'WARM', flow_type='ELEMENTARY_FLOW',
                            keep_unmapped_rows=True, field_dict = field_dict,
                            ignore_source_name = True)

    return df
