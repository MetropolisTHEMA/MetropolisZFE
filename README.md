# Application of Metropolis to Paris' Zone à Faibles Émissions

## Notes

Command to get a CSV from a GeoPackage:

`ogr2ogr -f "CSV" FILE.csv FILE.gpkg -sql "SELECT person_id, trip_index, departure_time,
arrival_time, mode, preceding_purpose, following_purpose, ST_X(ST_StartPoint(geom)) as 'x0',
ST_Y(ST_StartPoint(geom)) as 'y0', ST_X(ST_EndPoint(geom)) as 'x1', ST_Y(ST_EndPoint(geom)) as 'y1'
FROM FILE`
