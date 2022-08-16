# mapping.py (WARMer)

"""
Functions related to mapping flows and processes
"""

import yaml
from pathlib import Path

import numpy as np
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

    # Mapping data for economic flows (e.g. 'Jobs') are marked with 'n.a.'
    # for some fields. Replace those with None.
    df = df.replace('n.a.', np.nan)

    return df


def map_useeio_processes(df):
    """Update process data based on process mapping file and apply conversions."""
    mapping = pd.read_csv(modulepath/'processmapping'/'processmapping.csv')
    mapping['ConversionFactor'] = mapping['ConversionFactor'].fillna(1)
    mapping_cols = ['TargetProcessName', 'TargetUnit', 'TargetProcessID',
                    'ConversionFactor', 'process ID']
    df = df.merge(mapping[mapping_cols],
                  how='left', left_on = ['from_process_ID'],
                  right_on = ['process ID'])
    if 'from_process_fgbg' in df.columns:
        criteria = (df['from_process_fgbg'] == 'background_map') & \
                   (df['TargetProcessName'].notnull())
    elif 'from_process_class' in df.columns:
       criteria = (df['from_process_class'] == 'special') & \
                   (df['TargetProcessName'].notnull())
    print(f'mapping {sum(criteria)} processes')
    df.loc[criteria, 'from_process_name'] = df['TargetProcessName']
    df.loc[criteria, 'from_process_ID'] = df['TargetProcessID']
    df.loc[criteria, 'from_flow_unit'] = df['TargetUnit']
    df.loc[criteria, 'Amount'] = df['Amount'] * df['ConversionFactor']
    df.drop(columns=mapping_cols, inplace=True)
    return df


def identify_price_flowsa(sector, fba_dict):
    """Extract and filter FBA datasets based on passed fba_dict. Divide economic
    output by total flow to obtain price."""
    import flowsa
    df = flowsa.getFlowByActivity(datasource = fba_dict['SourceName'],
                                  year = fba_dict['Year'])

    for k,v in fba_dict.items():
        df = df.loc[df[k]==v]

    if len(df) > 1:
        print('WARNING more than one record available')

    x = df[['FlowAmount','Unit']].groupby(by = 'Unit').sum('FlowAmount').reset_index()

    output = flowsa.getFlowByActivity(datasource = 'BEA_GDP_GrossOutput',
                                      year = fba_dict['Year'])
    output = output.loc[output['ActivityProducedBy'] == sector
                        ].reset_index(drop=True)

    value = output['FlowAmount'][0] / x['FlowAmount'][0]
    unit = f"USD / {x['Unit'][0]}"

    return {sector: {unit:value}}


def generate_prices():
    """Generate prices for all sectors stored in price_data.yaml"""
    with open(modulepath/'data'/'price_data.yaml', 'r') as f:
            method_dict = yaml.safe_load(f)
    price_dict = {}
    for sector, fba_dict in method_dict.items():
        price_dict.update(identify_price_flowsa(str(sector), fba_dict))
    return price_dict


if __name__ == "__main__":
    d = generate_prices()