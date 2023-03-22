# WARMer
<!-- badges: start -->
[![Python CI/CD tests](https://github.com/USEPA/WARMer/actions/workflows/python_CI-CD.yaml/badge.svg)](https://github.com/USEPA/WARMer/actions/workflows/python_CI-CD.yaml)
<!-- badges: end -->

The **WARM** transform**er** (**WARMer**) is a Python package that extracts and processes data from the [USEPA Waste Reduction Model (WARM) v15 openLCA database](https://www.epa.gov/warm/versions-waste-reduction-model-warm#15) into tabular and matrix formats. 

Current status: **Alpha**

## Installation
Requires Python >= 3.8 with pip.

A simple installation with basic functionality, assuming this repository is downloaded or cloned
```
python -m pip install warmer
```

To connect to a live OpenLCA database, the extra `olca` dependency must be installed:  
```
python -m pip install warmer[olca]
```

To perform [fedelem](https://github.com/USEPA/Federal-LCA-Commons-Elementary-Flow-List) flow mapping, the extra `fedelem_map` must be installed:  
```
python -m pip install warmer[fedelem_map]
```

To install everything at once, use the following command:  
```
python -m pip install SomePackage[olca, fedelem_map]
```


## Disclaimer

The United States Environmental Protection Agency (EPA) GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. EPA has relinquished control of the information and no longer has responsibility to protect the integrity , confidentiality, or availability of the information. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by EPA. The EPA seal and logo shall not be used in any manner to imply endorsement of any commercial product or activity by EPA or the United States Government.

