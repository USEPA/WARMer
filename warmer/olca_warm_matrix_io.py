# olca_warm_io.py (WARMer)

"""
Import WARM OpenLCA model's technology & intervention matrices
Last Updated: Wednesday, November 11, 2021
"""
import pandas as pd
import numpy as np
from olca_data_unpack import classify_prcs
from pathlib import Path

modulepath = Path(__file__).parent

def olca_mtx_read(filename):
    """
    Import baseline & alternative intervention matrices
    """
    df = pd.read_csv(modulepath/'data'/'warm_olca_mtx'/filename, header=None)
    # df.columns = df.columns.astype(str)  # defaults to int when header=None
    return df

def olca_tech_clean(df):
    """
    Concatenate flows + categories (first 2 cols) into delimited strings;
    insert transposed copy of row labels as column headers to explicitly 
    assign implied identities of square matrix columns;
    then melt to long format, split using delimiter, and filter out 0's.
    """
    df['0'] = df['0'] + ';' + df['1'].fillna('NA')  # mitigates nan in 1st row
    df = df.drop('1', axis=1)
        # [later] UNIT TEST to ensure raw cols do not contain semi-colon, 
        # e.g. any(df['0'].str.contains(';')) expects FALSE; repeat for ['1']

    # insert concat vals as column headers, then melt
    df.columns = np.append('from_flow', df['0'])
    df = pd.melt(df, id_vars='from_flow', var_name = 'to_flow')
    
    # separate concatenated columns, plus extract units 
    post_melt_parse(df, 'from')
    post_melt_parse(df, 'to')
    
    # reorder columns
    df = df[['from_flow','from_flow_unit','from_flow_cat',
            'to_flow','to_flow_unit','to_flow_cat','value']]

    # # filter out 0 values (i.e., empty flows)
    # df = df.query('value != 0')
    
    return df

def olca_intv_clean(df):
    """
    Concatenate midpoints + contexts (first 2 cols) into delimited strings;
    transpose and repeat concatenation w/ flows + categories;
    convert first row to column headers;
    then melt to long format, split using delimiter, and filter out 0's.
    """
    df['0'] = df['0'] + ';' + df['1']
    df = df.drop('1', axis=1).transpose()
    df.columns = df.columns.astype(str)  # b/c int index values became headers
    df['0'] = df['0'] + ';' + df['1'].fillna('NA')
    df = df.drop('1', axis=1).fillna('from_flow')  # replace final NA w/ header

    # insert concat vals as column headers, then melt
    df.columns = df.iloc[0]
    df = df[1:]  # drop first row
    df = pd.melt(df, id_vars='from_flow', var_name = 'to_flow')  
    
    # separate concatenated columns, plus extract units 
    post_melt_parse(df, 'from')
    post_melt_parse(df, 'to')
    
    # reorder columns
    df = df[['from_flow','from_flow_unit','from_flow_cat',
            'to_flow','to_flow_unit','to_flow_cat','value']]

    # filter out 0 values (i.e., empty flows)
    df['value'] = pd.to_numeric(df['value'])
    df = df.query('value != 0')
    
    return df

def post_melt_parse(df, fromto):
    """
    Separate semi-colon delimited str cols used in melt steps,
    plus extract "[unit]" from flow labels
    """
    flow = f'{fromto}_flow'
    
    df[[flow, flow+'_cat']] = \
        df[flow].str.split(';', expand=True)
        
    df[[flow, flow+'_unit']] = \
        df[flow].str.extract('(.*)\s\[(.*)\]', expand=True)

    return df

if __name__ == '__main__':
    tech_raw = olca_mtx_read('A.csv')
    intv_raw = olca_mtx_read('B.csv')
    
    a_index = pd.read_csv(modulepath/'data'/'warm_olca_mtx'/'index_A.csv')
    a_index = classify_prcs(a_index)
    a_index['MappingRequired'] = np.where(a_index.prcs_class=='background_map',
                                          True, False)
    newcols = ['MatchCondition','ConversionFactor',
               'TargetListName','TargetProcessName','TargetUnit','LastUpdated']
    a_index = a_index.reindex(columns = a_index.columns.tolist() + newcols)
    a_index.to_csv(modulepath/'processmapping'/'processmpaping.csv', index=False)

    
    # tech = olca_tech_clean(tech_raw)
    # intv = olca_intv_clean(intv_raw)
    
    # # comparing old & new (comprehensive) flow matrices
    # tech_raw_old = olca_mtx_read('technology_matrix_old.csv')
    # intv_raw_old = olca_mtx_read('intervention_matrix_old.csv')

    # tech_old = olca_tech_clean(tech_raw_old)
    # intv_old = olca_intv_clean(intv_raw_old)
    
    # temp_tech = tech.compare(tech_old)
    # temp_intv = intv.compare(intv_old)
    # temp_tech_raw = tech_raw.compare(tech_raw_old)
    # temp_intv_raw = intv_raw.compare(intv_raw_old)
    
