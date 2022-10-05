# Format specifications for environmental data

| Field           | Type  | Required |  Note |
|:----------------|:-----:|:--------:|:------|
| ProcessID       | str   | Y        | ProcessID of the source process |
| ProcessName     | str   | N        | Name of the source process |
| ProcessCategory | str   | N        | Name of the category that the source process resides in |
| Location        | str   | Y        | Two-digit code, e.g., `US` |
| Amount          | float | Y        | Per unit of reference flow |
| Flowable        | str   | Y        | Name of flowable; see [Flow List Format](https://github.com/USEPA/Federal-LCA-Commons-Elementary-Flow-List/blob/master/format%20specs/FlowList.md) |
| Context         | str   | Y        | See [Flow List Format](https://github.com/USEPA/Federal-LCA-Commons-Elementary-Flow-List/blob/master/format%20specs/FlowList.md) |
| Unit            | str   | Y        | Unit abbreviation; e.g., `kg` |
| FlowUUID        | str   | Y        | Unique hexadecimal ID for flow within `FlowListName` |
| FlowListName    | str   | Y        | Name and version of the flowlist; see `SourceFlowList` in [Flowmapping format](https://github.com/USEPA/Federal-LCA-Commons-Elementary-Flow-List/blob/master/format%20specs/FlowMapping.md) |
