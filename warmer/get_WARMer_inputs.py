# get_WARMer_inputs.py (HIO)
"""
Script to generate WARMer model inputs
"""
import pandas as pd
from pathlib import Path

# from warmer.olca_warm_matrix_io import get_exchanges, warm_version
from olca_warm_matrix_io import get_exchanges, warm_version

modulepath = Path(__file__).parent
file_stub = f'{warm_version}'

if __name__ == '__main__':
    # dict_ff = {'m1': 'model_1_processes.csv'}
    # file_ff = dict_ff.get(model_name)

    ## FBS input
    model_name = None
    writepath = modulepath.parent/'flowsa'

    df_a, df_b = get_exchanges(opt_map=None)

    df_b.to_csv(writepath/f'{file_stub}_env.csv', index=False)

    ## Model 1
    model_name = 'm1'
    df_m1_prcs = pd.read_csv(modulepath/'model_1_processes.csv')

    df_a, df_b = get_exchanges(df_subset=df_m1_prcs, opt_mixer=None)

    writepath = modulepath.parent/'model_build'/'data'
    (df_a.query('Amount != 0')  # drop empty exchanges
         .to_csv(writepath/f'{file_stub}_{model_name}_tech.csv', index=False))
    (df_b.drop(columns='ProcessCategory')
         .query('Amount != 0')  # drop empty exchanges**
         .to_csv(writepath/f'{file_stub}_{model_name}_env.csv', index=False))

        # Of the 14 types of elementary flows in df_b, only 11 remain after
        # removing all-0-valued exchanges. Removed flows include:
            # Ethane, hexafluoro-, HFC-116
            # Methane, tetrafluoro-, R-14
            # Other means of transport (no truck, train or ...)
