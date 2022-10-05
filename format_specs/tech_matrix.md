# Format specifications for technological flow data

To match the style of the `A` matrix:

Rows and columns are sectors in the format of `code_loc`

To build the `A` matrix, the technological data frame requires the following fields:

| Field         | Type  | Required | Note |
|:--------------|:-----:|:--------:|:-----|
| ProcessID     | str   | Y        | ProcessID of the consuming process |
| ProcessName   | str   | N        |      |
| ProcessUnit   | str   | N        | Unit abbreviation; e.g., `kg` |
| Location      | str   | Y        | Two-digit code, e.g., `US` |
| Amount        | float | Y        | Normalized per unit of reference flow |
| FlowID        | str   | Y        | ProcessID of the flow being consumed; if an IO sector use [code_loc format](https://github.com/USEPA/useeior/blob/master/format_specs/Model.md#sector-meta) |
| Flow          | str   | N        | ProcessName of the flow being consumed |
| FlowUnit      | str   | Y        | Unit abbreviation; e.g., `kg` |
