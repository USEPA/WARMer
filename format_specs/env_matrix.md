# Format specifications for environmental data

To match the style of the `B` matrix:

Rows are `flowable/context/unit` and columns are sector in the format of `code_loc`

To build the `B` matrix, the environmental data frame requires the following fields:

 Field | Type | Required |  Note |
----------- |  ---- | ---------| -----  |
ProcessID | str | Y | ProcessID of the source process
ProcessName | str | N |
Location | str | Y | Two-digit code, e.g., `US`
Amount | float | Y | Per unit of reference flow
Flowable | str | Y | Name of flowable
Context | str | Y | Compartments separated by /; e.g., `emission/air`
Unit | str | Y | Unit abbreviation; e.g., `kg`
FlowUUID | str| Y | Unique hexadecimal ID for each Flowable
FlowListFormat | str | Y | Name and version of the flowlist data format; e.g., `openLCA v2.0` or `FEDEFL v1.0.9`


See [useeior SatelliteTables](https://github.com/USEPA/useeior/blob/master/format_specs/Model.md#satellitetables) for full available specs
