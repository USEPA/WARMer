from setuptools import setup, find_packages

setup(
    name='WARMer',
    version='0.1.0',
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=['pandas>=1.4.0',
                      'numpy>=1.20.1',
                      ],
    extras_require={'olca': 'olca-ipc==0.0.10',
                    'fedelem_map':
                        ['fedelemflowlist @ git+https://github.com/USEPA/Federal-LCA-Commons-Elementary-Flow-List.git@develop#egg=fedelemflowlist',
                         'esupy @ git+https://github.com/USEPA/esupy.git@develop#egg=esupy']},
    url='https://github.com/USEPA/WARMer.git',
    author='Andrew Beck, Ben Young, Jorge Vendries, and Wesley Ingwersen',
    author_email='ingwersen.wesley@epa.gov',
    description='Extract and process data from WARM openLCA database'
)
