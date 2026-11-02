library(sf)
library(terra)

carpeta <- "C:/DOCUMENTS/EARTH_SCIENCES/CLIMA/2027_Alvarez-Garreton/ANALYSIS/codigos_ivan/ms_recovery-main/data/rivers"
setwd(carpeta)
rios <- st_read("HydroRIVERS_v10_sa.shp")

# Límites geográficos (ejemplo; reemplazalos por los tuyos)
lon_min <- -74
lon_max <- -66
lat_min <- -43
lat_max <- -17

zona <- st_as_sfc(st_bbox(
  c(xmin = lon_min, ymin = lat_min,
    xmax = lon_max, ymax = lat_max),
  crs = st_crs(4326)
))

# Transformar solo si el CRS de los ríos es diferente
rios <- st_transform(rios, 4326)

# Seleccionar líneas que intersectan el rectángulo
rios_filtrados <- rios[st_intersects(rios, zona, sparse = FALSE)[, 1], ]

st_write(rios_filtrados, "rios.gpkg", delete_dsn = TRUE)


# Rasters ORIGINALES de Sudamérica (los pesados); ajustá los nombres
dir_full <- rast("hyd_sa_dir_15s.tif")
aca_full <- rast("hyd_sa_aca_15s.tif")

zona_r <- ext(lon_min, lon_max, lat_min, lat_max)

crop(dir_full, zona_r, filename = "sa_dir_15s_crop.tif",
     overwrite = TRUE, datatype = "INT1U", gdal = "COMPRESS=DEFLATE")
crop(aca_full, zona_r, filename = "sa_aca_15s_crop.tif",
     overwrite = TRUE, gdal = "COMPRESS=DEFLATE")