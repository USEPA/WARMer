## SQL queries for data access in openLCA

Generate list of processes with reference flows:

process_map.csv

select

processes.name as ProcessName,

flows.name as ReferenceFlow,

loc.name as Location,

cat1.name as Category,

cat2.name as Category2,

cat3.name as Category3,

cat4.name as Category4,

cat5.name as Category5 from tbl_processes processes

left outer join tbl_categories cat1 on processes.f_category = cat1.id

left outer join tbl_categories cat2 on cat1.f_category = cat2.id

left outer join tbl_categories cat3 on cat2.f_category = cat3.id

left outer join tbl_categories cat4 on cat3.f_category = cat4.id

left outer join tbl_categories cat5 on cat4.f_category = cat5.id

left outer join tbl_exchanges exchanges on processes.F_QUANTITATIVE_REFERENCE = exchanges.id

left outer join tbl_flows flows on exchanges.f_flow = flows.id

left outer join tbl_locations loc on processes.F_location = loc.id


