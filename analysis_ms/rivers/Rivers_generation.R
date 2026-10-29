library(sf)
library(terra)


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


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
carpeta <- "C:/DOCUMENTS/EARTH_SCIENCES/CLIMA/2027_Alvarez-Garreton/R_nue1/GIS"
setwd(carpeta)

crs_dem <- st_crs(32719)   # UTM 19S (pozos y DEM)
crs_r   <- st_crs(4326)    # WGS84 (flow direction y ACA)

# --- shape de ríos (HydroRIVERS / HydroSHEDS 1): AJUSTAR NOMBRES ---
archivo_rios <- "rios.gpkg"
col_id   <- "HYRIV_ID"     # id del tramo
col_next <- "NEXT_DOWN"    # id del tramo aguas abajo (0 = desembocadura)
col_area <- "UPLAND_SKM"   # área de drenaje; sirve para elegir el cauce principal
col_q    <- "DIS_AV_CMS"   # caudal medio anual (m3/s)

# --- criterios de "río" ---
#  fuente "raster": ACC de HydroSHEDS 1 (celdas), convertido a km2 al leerlo
#  fuente "vector": caudal del shape (m3/s)
criterios <- data.frame(
  etiqueta = c("A500", "A1000", "A2000", "Q0p5", "Q1p5", "Q2p5"),
  fuente   = c("raster", "raster", "raster", "vector", "vector", "vector"),
  umbral   = c(500, 1000, 1500, 0.5, 1.5, 2.5),
  stringsAsFactors = FALSE)

radio_z <- 100   # m: radio para tomar el mínimo del DEM en el punto del río

# ============================================================
# 1. DEM (UTM 19S, 12.5 m)
# ============================================================
tifs <- list.files(carpeta, pattern = "\\.jp2$", full.names = TRUE, ignore.case = TRUE)


dems <- lapply(tifs, function(f) {
  r <- rast(f)
  crs(r) <- "EPSG:32719"
  NAflag(r) <- 0
  r
})

# altura puntual (pozos)
extraer_multi <- function(xy, dems) {
  z <- rep(NA_real_, nrow(xy))
  for (r in dems) {
    falta <- which(is.na(z))
    if (length(falta) == 0) break
    pts <- vect(xy[falta, , drop = FALSE], crs = "EPSG:32719")
    z[falta] <- terra::extract(r, pts, ID = FALSE)[, 1]
  }
  z
}

# altura mínima del DEM dentro de un círculo (puntos de río)
extraer_min <- function(xy, dems, radio = 100) {
  z <- rep(NA_real_, nrow(xy))
  for (r in dems) {
    falta <- which(is.na(z))
    if (length(falta) == 0) break
    circ <- terra::buffer(vect(xy[falta, , drop = FALSE], crs = "EPSG:32719"),
                          width = radio)
    v <- terra::extract(r, circ, fun = min, na.rm = TRUE, ID = FALSE)[, 1]
    v[is.infinite(v)] <- NA
    z[falta] <- v
  }
  z
}

# ============================================================
# 2. POZOS
# ============================================================
pozos <- st_read("POZOS1.gpkg")
pozos <- st_zm(pozos)

# Transformar todos los puntos de WGS84 a UTM 19S
pozos <- st_transform(pozos, 32719)
stopifnot(st_crs(pozos) == crs_dem)
pozos <- pozos[!st_is_empty(pozos), ]

cajas <- lapply(dems, function(r) st_as_sfc(st_bbox(
  c(xmin = xmin(r), xmax = xmax(r), ymin = ymin(r), ymax = ymax(r)), crs = crs_dem)))
caja_dem <- st_union(do.call(c, cajas))

pozos <- st_filter(pozos, caja_dem)
stopifnot(nrow(pozos) > 0)

# pozos <- pozos[1:20, ]   # <- descomentar para probar con pocos pozos

# ============================================================
# 3. RASTERS DE FLUJO (HydroSHEDS 1) Y RÍOS
# ============================================================
# La carpeta debe tener SOLO los DIR y ACC de HydroSHEDS 1
fd_files  <- list.files(carpeta, pattern = "DIR.*\\.tif$", full.names = TRUE, ignore.case = TRUE)
aca_files <- list.files(carpeta, pattern = "ACC.*\\.tif$", full.names = TRUE, ignore.case = TRUE)  # ACC = nº de celdas aguas arriba
stopifnot(length(fd_files) > 0, length(aca_files) > 0)

fdir <- vrt(fd_files)
aca  <- vrt(aca_files)
if (crs(fdir) == "") crs(fdir) <- "EPSG:4326"
if (crs(aca)  == "") crs(aca)  <- "EPSG:4326"
stopifnot(compareGeom(fdir, aca, stopOnError = FALSE))

# Shape de ríos (solo se usa para los criterios de caudal)
rios <- st_zm(st_read(archivo_rios))
if (!col_area %in% names(rios)) col_area <- col_q   # para elegir el cauce principal
stopifnot(all(c(col_id, col_next, col_q) %in% names(rios)))
rios_u <- st_transform(rios, crs_dem)   # para distancias
rios_r <- st_transform(rios, crs_r)     # para rasterizar en la grilla de fdir

# Caudal máximo por celda (NA donde no hay río).
# Se rasteriza solo la zona del shape, alineada a la grilla de fdir (que cubre todo
# Sudamérica), y por bloques de filas pequeños (evita std::bad_alloc).
# fila/columna en fdir = fila/columna en r_q + off_r / off_c
bb    <- st_bbox(rios_r)
xr    <- terra::res(fdir)[1];  yr <- terra::res(fdir)[2]
c_ini <- max(1, floor((unname(bb["xmin"]) - xmin(fdir)) / xr) + 1 - 2)
c_fin <- min(ncol(fdir), ceiling((unname(bb["xmax"]) - xmin(fdir)) / xr) + 2)
r_ini <- max(1, floor((ymax(fdir) - unname(bb["ymax"])) / yr) + 1 - 2)
r_fin <- min(nrow(fdir), ceiling((ymax(fdir) - unname(bb["ymin"])) / yr) + 2)
off_r <- r_ini - 1;  off_c <- c_ini - 1

tpl <- rast(xmin = xmin(fdir) + off_c * xr, xmax = xmin(fdir) + c_fin * xr,
            ymax = ymax(fdir) - off_r * yr, ymin = ymax(fdir) - r_fin * yr,
            resolution = c(xr, yr), crs = "EPSG:4326")
stopifnot(nrow(tpl) == r_fin - r_ini + 1, ncol(tpl) == c_fin - c_ini + 1)

bloque <- 250
margen <- 0.01   # grados: evita perder tramos en el borde de cada bloque
archivo_q <- "tmp_rios_q.tif"
unlink(c(archivo_q, paste0(archivo_q, ".aux.xml")))   # borra restos de un intento fallido
if (file.exists(archivo_q))                            # si Windows lo tiene bloqueado, otro nombre
  archivo_q <- paste0("tmp_rios_q_", format(Sys.time(), "%H%M%S"), ".tif")
r_q <- rast(tpl)
writeStart(r_q, archivo_q, overwrite = TRUE,
           datatype = "FLT4S", gdal = c("COMPRESS=DEFLATE"))
for (r0 in seq(1, nrow(tpl), by = bloque)) {
  nr <- min(bloque, nrow(tpl) - r0 + 1)
  yt <- yFromRow(tpl, r0) + yr / 2
  yb <- yFromRow(tpl, r0 + nr - 1) - yr / 2
  blk <- rast(xmin = xmin(tpl), xmax = xmax(tpl), ymin = yb, ymax = yt,
              ncols = ncol(tpl), nrows = nr, crs = "EPSG:4326")
  caja <- st_as_sfc(st_bbox(c(xmin = xmin(tpl), xmax = xmax(tpl),
                              ymin = yb - margen, ymax = yt + margen), crs = crs_r))
  hay <- lengths(st_intersects(rios_r, caja)) > 0
  if (any(hay)) {
    rb <- rasterize(vect(rios_r[hay, col_q]), blk, field = col_q, fun = "max", touches = TRUE)
    v  <- values(rb, mat = FALSE)
  } else {
    v <- rep(NA_real_, nr * ncol(tpl))
  }
  writeValues(r_q, v, r0, nr)
  if ((r0 - 1) %/% bloque %% 10 == 0) message("rasterizado: filas ", r0, "-", r0 + nr - 1, " de ", nrow(tpl))
}
r_q <- writeStop(r_q)

readStart(fdir); readStart(aca); readStart(r_q)

xy_pozo_utm <- st_coordinates(pozos)[, 1:2]
xy_pozo_r   <- st_coordinates(st_transform(pozos, crs_r))[, 1:2]
celdas      <- cellFromXY(fdir, xy_pozo_r)
rc0         <- rowColFromCell(fdir, celdas)

# ============================================================
# 4. FUNCIONES
# ============================================================
codigos <- c(1, 2, 4, 8, 16, 32, 64, 128)  # E, SE, S, SW, W, NW, N, NE
dr <- c( 0,  1,  1,  1,  0, -1, -1, -1)
dc <- c( 1,  1,  0, -1, -1, -1,  0,  1)

# ACC (HydroSHEDS 1) = nº de celdas aguas arriba. Se convierte a km2 con el área
# real de la celda según su latitud (la celda se achica hacia los polos)
area_celda_km2 <- function(lat) {
  rx <- terra::res(fdir)[1] * pi / 180
  ry <- terra::res(fdir)[2] * pi / 180
  6371.0088^2 * rx * (sin(lat * pi / 180 + ry / 2) - sin(lat * pi / 180 - ry / 2))
}

# Lee una ventana del ACC y la devuelve en km2 (matriz nr x nc)
leer_acc <- function(r0, nr, c0, nc) {
  M   <- matrix(readValues(aca, r0, nr, c0, nc), nrow = nr, ncol = nc, byrow = TRUE)
  lat <- yFromRow(fdir, r0:(r0 + nr - 1))
  M * area_celda_km2(lat)
}

# Lee una ventana (en índices de fdir) del raster de caudal r_q, que cubre solo
# una parte de la grilla; fuera de r_q devuelve NA (= no es río)
leer_q <- function(r0, nr, c0, nc) {
  M  <- matrix(NA_real_, nrow = nr, ncol = nc)
  ra <- max(r0, r_ini); rb <- min(r0 + nr - 1, r_fin)
  ca <- max(c0, c_ini); cb <- min(c0 + nc - 1, c_fin)
  if (ra <= rb && ca <= cb) {
    v <- readValues(r_q, ra - off_r, rb - ra + 1, ca - off_c, cb - ca + 1)
    M[(ra - r0 + 1):(rb - r0 + 1), (ca - c0 + 1):(cb - c0 + 1)] <-
      matrix(v, nrow = rb - ra + 1, byrow = TRUE)
  }
  M
}

# Sigue la dirección de flujo hasta una celda de río (ras >= umbral)
seguir <- function(row, col, ras, umbral, acc = FALSE, max_pasos = 200000, W = 1500) {
  NR <- nrow(fdir); NC <- ncol(fdir)
  cargar <- function(row, col) {
    r0 <- max(1, row - W %/% 2); c0 <- max(1, col - W %/% 2)
    nr <- min(W, NR - r0 + 1);   nc <- min(W, NC - c0 + 1)
    list(r0 = r0, c0 = c0, nr = nr, nc = nc,
         fd = matrix(readValues(fdir, r0, nr, c0, nc), nrow = nr, ncol = nc, byrow = TRUE),
         ac = if (acc) leer_acc(r0, nr, c0, nc) else leer_q(r0, nr, c0, nc))
  }
  rs <- integer(max_pasos); cs <- integer(max_pasos)
  win <- cargar(row, col)
  fin <- function(estado, i) list(estado = estado, rs = rs[seq_len(i)], cs = cs[seq_len(i)])
  
  for (i in seq_len(max_pasos)) {
    lr <- row - win$r0 + 1; lc <- col - win$c0 + 1
    if ((lr <= 1 && win$r0 > 1) || (lr >= win$nr && win$r0 + win$nr - 1 < NR) ||
        (lc <= 1 && win$c0 > 1) || (lc >= win$nc && win$c0 + win$nc - 1 < NC)) {
      win <- cargar(row, col)
      lr <- row - win$r0 + 1; lc <- col - win$c0 + 1
    }
    rs[i] <- row; cs[i] <- col
    a <- win$ac[lr, lc]; d <- win$fd[lr, lc]
    if (is.na(d))                  return(fin("NA / fuera del raster", i))
    if (!is.na(a) && a >= umbral)  return(fin("ok", i))   # NA en ras = no es río
    if (d == 0)                    return(fin("sumidero", i))
    k <- match(d, codigos)
    if (is.na(k))                  return(fin("codigo invalido", i))
    row <- row + dr[k]; col <- col + dc[k]
    if (row < 1 || row > NR || col < 1 || col > NC) return(fin("borde del raster", i))
  }
  fin("max_pasos", max_pasos)
}

# ---------- fuente RASTER (ACA) ----------

# Celdas del MISMO río que pasa por (row0, col0): cauce principal aguas arriba y aguas abajo
rio_mismo_r <- function(row0, col0, u, max_pasos = 20000, W = 1001) {
  NR <- nrow(fdir); NC <- ncol(fdir)
  win <- NULL
  cargar <- function(rw, cl) {
    r0 <- max(1, rw - W %/% 2); c0 <- max(1, cl - W %/% 2)
    nr <- min(W, NR - r0 + 1); nc <- min(W, NC - c0 + 1)
    win <<- list(r0 = r0, c0 = c0, nr = nr, nc = nc,
                 fd = matrix(readValues(fdir, r0, nr, c0, nc), nrow = nr, ncol = nc, byrow = TRUE),
                 ac = leer_acc(r0, nr, c0, nc))
  }
  asegurar <- function(rw, cl) {
    if (is.null(win) ||
        max(1, rw - 1) < win$r0 || min(NR, rw + 1) > win$r0 + win$nr - 1 ||
        max(1, cl - 1) < win$c0 || min(NC, cl + 1) > win$c0 + win$nc - 1) cargar(rw, cl)
  }
  val <- function(rw, cl) c(fd = win$fd[rw - win$r0 + 1, cl - win$c0 + 1],
                            ac = win$ac[rw - win$r0 + 1, cl - win$c0 + 1])
  
  # aguas arriba (cauce principal: vecina que drena hacia la celda con mayor ACA)
  rs <- row0; cs <- col0
  rw <- row0; cl <- col0
  for (p in seq_len(max_pasos)) {
    asegurar(rw, cl)
    mejor <- NULL; mejor_ac <- -Inf
    for (k in 1:8) {
      rn <- rw + dr[k]; cn <- cl + dc[k]
      if (rn < 1 || rn > NR || cn < 1 || cn > NC) next
      v <- val(rn, cn)
      opuesto <- codigos[((k - 1 + 4) %% 8) + 1]
      if (!is.na(v["fd"]) && !is.na(v["ac"]) && v["fd"] == opuesto &&
          v["ac"] >= u && v["ac"] > mejor_ac) {
        mejor <- c(rn, cn); mejor_ac <- v["ac"]
      }
    }
    if (is.null(mejor)) break
    rw <- mejor[1]; cl <- mejor[2]
    rs <- c(rs, rw); cs <- c(cs, cl)
  }
  
  # aguas abajo
  rw <- row0; cl <- col0
  for (p in seq_len(max_pasos)) {
    asegurar(rw, cl)
    v <- val(rw, cl)
    if (is.na(v["fd"]) || v["fd"] == 0) break
    k <- match(v["fd"], codigos); if (is.na(k)) break
    rw <- rw + dr[k]; cl <- cl + dc[k]
    if (rw < 1 || rw > NR || cl < 1 || cl > NC) break
    rs <- c(rs, rw); cs <- c(cs, cl)
  }
  list(rs = rs, cs = cs)
}

# (3) distancia mínima a las celdas del mismo río (raster)
cercano_mismo_r <- function(xy_p, row0, col0, u) {
  rio <- rio_mismo_r(row0, col0, u)
  xy  <- xyFromCell(fdir, cellFromRowCol(fdir, rio$rs, rio$cs))
  utm <- st_coordinates(st_transform(
    st_as_sf(as.data.frame(xy), coords = c("x", "y"), crs = crs_r), crs_dem))
  d <- sqrt((utm[, 1] - xy_p[1])^2 + (utm[, 2] - xy_p[2])^2)
  j <- which.min(d)
  list(dist = d[j], xy = utm[j, ])
}

# (4) distancia mínima a cualquier celda con aca >= u (raster)
# d_max = radio de búsqueda en metros (dist_recta es una cota segura)
cercano_cualq_r <- function(xy_p, row, col, d_max, u) {
  NR <- nrow(fdir); NC <- ncol(fdir)
  lat <- yFromRow(fdir, row)
  rx <- terra::res(fdir)[1]; ry <- terra::res(fdir)[2]
  hr <- ceiling(d_max / (ry * 111320)) + 2
  hc <- ceiling(d_max / (rx * 111320 * cos(lat * pi / 180))) + 2
  r0 <- max(1, row - hr); r1 <- min(NR, row + hr)
  c0 <- max(1, col - hc); c1 <- min(NC, col + hc)
  nr <- r1 - r0 + 1;      nc <- c1 - c0 + 1
  
  M   <- leer_acc(r0, nr, c0, nc)
  idx <- which(M >= u, arr.ind = TRUE)
  if (length(idx) == 0) return(NULL)
  
  rr <- r0 + idx[, 1] - 1
  cc <- c0 + idx[, 2] - 1
  xy  <- xyFromCell(fdir, cellFromRowCol(fdir, rr, cc))
  utm <- st_coordinates(st_transform(
    st_as_sf(as.data.frame(xy), coords = c("x", "y"), crs = crs_r), crs_dem))
  d <- sqrt((utm[, 1] - xy_p[1])^2 + (utm[, 2] - xy_p[2])^2)
  j <- which.min(d)
  list(dist = d[j], xy = utm[j, ])
}

# ---------- fuente VECTOR (shape, caudal) ----------

# Índices (dentro de los tramos filtrados) del MISMO río que pasa por el tramo k0:
# cadena aguas abajo + cauce principal aguas arriba (afluente de mayor área)
cadena_rio <- function(k0, ids, nxt, rank, hijos) {
  sel <- k0
  k <- k0
  repeat {                                   # aguas abajo
    j <- match(nxt[k], ids)
    if (is.na(j) || j %in% sel) break
    sel <- c(sel, j); k <- j
  }
  k <- k0
  repeat {                                   # aguas arriba
    h <- hijos[[as.character(ids[k])]]
    if (is.null(h)) break
    k <- h[which.max(rank[h])]
    sel <- c(sel, k)
  }
  sel
}

# Altura del DEM (mínimo en radio_z) para los puntos xy[sel, ]
alt <- function(xy, sel) {
  z <- rep(NA_real_, nrow(xy))
  if (any(sel)) z[sel] <- extraer_min(xy[sel, , drop = FALSE], dems, radio = radio_z)
  z
}

# ============================================================
# 5. CORRER PARA CADA CRITERIO
# ============================================================
n <- nrow(pozos)
pozos$z_pozo <- extraer_multi(xy_pozo_utm, dems)

guardar_rectas <- function(sel, xy_dest, dist, nombre, et) {
  if (!any(sel)) return(invisible())
  ls <- lapply(which(sel), function(i) st_linestring(rbind(xy_pozo_utm[i, ], xy_dest[i, ])))
  st_write(st_sf(id = which(sel), criterio = et, dist = dist[sel],
                 geometry = st_sfc(ls, crs = crs_dem)),
           paste0(nombre, "_", et, ".gpkg"), delete_dsn = TRUE)
}

for (cr in seq_len(nrow(criterios))) {
  et     <- criterios$etiqueta[cr]
  fuente <- criterios$fuente[cr]
  u      <- criterios$umbral[cr]
  ras    <- if (fuente == "raster") aca else r_q
  message("=== ", et, " (", fuente, ", umbral ", u, ") ===")
  
  if (fuente == "vector") {
    tramos <- rios_u[!is.na(rios_u[[col_q]]) & rios_u[[col_q]] >= u, ]
    if (nrow(tramos) == 0) { message("Ningún tramo cumple el criterio; se omite."); next }
    tr_geom <- st_geometry(tramos)
    ids     <- tramos[[col_id]]
    nxt     <- tramos[[col_next]]
    rank    <- tramos[[col_area]]
    hijos   <- split(seq_along(nxt), nxt)
  }
  
  estado     <- rep(NA_character_, n)
  dist_flujo <- rep(NA_real_, n)
  xy_fin     <- matrix(NA_real_, n, 2)
  lineas     <- vector("list", n)
  dist_mismo <- rep(NA_real_, n);  xy_mismo <- matrix(NA_real_, n, 2)
  dist_cualq <- rep(NA_real_, n);  xy_cualq <- matrix(NA_real_, n, 2)
  
  for (i in seq_len(n)) {
    if (i %% 200 == 0) message(i, " / ", n)
    if (is.na(celdas[i])) { estado[i] <- "pozo fuera del raster"; next }
    res <- seguir(rc0[i, 1], rc0[i, 2], ras, u, acc = (fuente == "raster"))
    estado[i] <- res$estado
    if (res$estado != "ok") next
    
    xy  <- xyFromCell(fdir, cellFromRowCol(fdir, res$rs, res$cs))
    utm <- st_coordinates(st_transform(
      st_as_sf(as.data.frame(xy), coords = c("x", "y"), crs = crs_r), crs_dem))
    
    xy_fin[i, ]   <- utm[nrow(utm), ]
    lineas[[i]]   <- st_linestring(rbind(xy_pozo_utm[i, ], utm))
    dist_flujo[i] <- as.numeric(st_length(lineas[[i]]))
    d_rec <- sqrt(sum((xy_pozo_utm[i, ] - xy_fin[i, ])^2))
    
    if (fuente == "raster") {
      # (3) mismo río y (4) cualquier río, recorriendo el raster ACA
      cm <- cercano_mismo_r(xy_pozo_utm[i, ], tail(res$rs, 1), tail(res$cs, 1), u)
      dist_mismo[i] <- cm$dist;  xy_mismo[i, ] <- cm$xy
      cq <- cercano_cualq_r(xy_pozo_utm[i, ], rc0[i, 1], rc0[i, 2], d_rec, u)
      if (!is.null(cq)) { dist_cualq[i] <- cq$dist; xy_cualq[i, ] <- cq$xy }
    } else {
      # (3) mismo río con la topología del shape
      k0  <- st_nearest_feature(st_sfc(st_point(xy_fin[i, ]), crs = crs_dem), tr_geom)
      sel <- cadena_rio(k0, ids, nxt, rank, hijos)
      np  <- st_nearest_points(st_sfc(st_point(xy_pozo_utm[i, ]), crs = crs_dem),
                               st_combine(tr_geom[sel]))
      co  <- st_coordinates(np)
      dist_mismo[i] <- sqrt(sum((co[1, 1:2] - co[2, 1:2])^2))
      xy_mismo[i, ] <- co[2, 1:2]
    }
  }
  print(table(estado, useNA = "ifany"))
  
  # (4) cualquier río con el shape: vectorizado, todos los pozos
  if (fuente == "vector") {
    j  <- st_nearest_feature(pozos, tr_geom)
    np <- st_nearest_points(st_geometry(pozos), tr_geom[j], pairwise = TRUE)
    dist_cualq <- as.numeric(st_length(np))
    xy_cualq   <- do.call(rbind, lapply(np, function(g) g[2, 1:2]))
  }
  
  ok <- !is.na(estado) & estado == "ok"
  dist_recta <- sqrt((xy_pozo_utm[, 1] - xy_fin[, 1])^2 +
                       (xy_pozo_utm[, 2] - xy_fin[, 2])^2)
  
  # alturas (mínimo del DEM en radio_z alrededor de cada punto de río)
  z_fin   <- alt(xy_fin,   ok)
  z_mismo <- alt(xy_mismo, ok & !is.na(dist_mismo))
  z_cualq <- alt(xy_cualq, !is.na(dist_cualq))
  
  pozos[[paste0("estado_",     et)]] <- estado
  pozos[[paste0("dist_flujo_", et)]] <- dist_flujo
  pozos[[paste0("dist_recta_", et)]] <- dist_recta
  pozos[[paste0("dist_mismo_", et)]] <- dist_mismo
  pozos[[paste0("dist_cualq_", et)]] <- dist_cualq
  pozos[[paste0("z_fin_",      et)]] <- z_fin
  pozos[[paste0("z_mismo_",    et)]] <- z_mismo
  pozos[[paste0("z_cualq_",    et)]] <- z_cualq
  pozos[[paste0("dz_",         et)]] <- pozos$z_pozo - z_fin
  pozos[[paste0("dz_mismo_",   et)]] <- pozos$z_pozo - z_mismo
  pozos[[paste0("dz_cualq_",   et)]] <- pozos$z_pozo - z_cualq
  
  message(sum(ok & is.na(z_fin)), " pozos con trayectoria ok pero sin altura en el DEM")
  
  # --- líneas por tipo de medición y criterio ---
  ok_l <- ok & !sapply(lineas, is.null)
  if (any(ok_l)) {
    st_write(st_sf(id = which(ok_l), criterio = et,
                   geometry = st_sfc(lineas[ok_l], crs = crs_dem)),
             paste0("lineas_flujo_", et, ".gpkg"), delete_dsn = TRUE)
  }
  guardar_rectas(ok,                      xy_fin,   dist_recta, "lineas_recta", et)
  guardar_rectas(ok & !is.na(dist_mismo), xy_mismo, dist_mismo, "lineas_mismo", et)
  guardar_rectas(!is.na(dist_cualq),      xy_cualq, dist_cualq, "lineas_cualq", et)
}

readStop(fdir); readStop(aca); readStop(r_q)

# ============================================================
# 6. GUARDAR EL SHAPE DE PUNTOS ÚNICO
# ============================================================
st_write(pozos, "pozos_resultado_flujo_todos.gpkg", delete_dsn = TRUE)





