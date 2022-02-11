# mapping.py (WARMer)

"""
Functions related to mapping flows and processes
"""

from pathlib import Path
import pandas as pd
from esupy.mapping import apply_flow_mapping

modulepath = Path(__file__).parent

def map_warmer_envflows(df):
    """Update elementary flow data from FEDEFL."""
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


def map_useeio_processes(df):
    """Update process data based on process mapping file and apply conversions."""
    mapping = pd.read_csv(modulepath/'processmapping'/'processmapping.csv')
    mapping['ConversionFactor'] = mapping['ConversionFactor'].fillna(1)
    mapping_cols = ['TargetProcessName', 'TargetUnit',
                    'ConversionFactor', 'process ID']
    df = df.merge(mapping[mapping_cols],
                  how='left', left_on = ['from_process_ID'],
                  right_on = ['process ID'])
    criteria = (df['from_process_class'] == 'background_map') & \
               (df['TargetProcessName'].notnull())
    print(f'mapping {sum(criteria)} processes')
    df.loc[criteria, 'from_process_name'] = df['TargetProcessName']
    df.loc[criteria, 'from_flow_unit'] = df['TargetUnit']
    df.loc[criteria, 'Amount'] = df['Amount'] * df['ConversionFactor']
    df.drop(columns=mapping_cols, inplace=True)
    return df
