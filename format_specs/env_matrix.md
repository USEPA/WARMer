# Format specifications for environmental data

To match the style of the `B` matrix:

Rows are `flowable/context/unit` and columns are sector in the format of `code_loc`

To build the `B` matrix, the environmental data frame requires the following fields:

 Field | Type | Required |  Note |
----------- |  ---- | ---------| -----  |
Flow | str | Y | FEDEFL nomenclature
Context | str | Y | FEDEFL nomenclature
Unit | str | Y | FEDEFL nomenclature
Amount | float | Y | Per unit of reference flow
Sector | str | Y | Matching the `BaseIOLevel`
Location | str | Y | two-digit code

See [useeior SatelliteTables](https://github.com/USEPA/useeior/blob/master/format_specs/Model.md#satellitetables) for full available specs