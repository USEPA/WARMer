# Format specifications for technological flow data

To match the style of the `A` matrix:

Rows and columns are sectors in the format of `code_loc`

To build the `A` matrix, the technological data frame requires the following fields:

 Field | Type | Required |  Note |
----------- |  ---- | ---------| -----  |
Flow | str | Y | Process_ID of the flow being consumed
Amount | float | Y | Per unit of reference flow
Unit | str | Y | FEDEFL nomenclature
Process_ID | str | Y |
Location | str | Y | two-digit code, e.g., `US`

