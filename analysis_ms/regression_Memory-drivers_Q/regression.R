setwd("C:/DOCUMENTS/EARTH_SCIENCES/CLIMA/2027_Alvarez-Garreton/R_nue1/R_prediction/Q")
library(fBasics);library(stats);library(scatterplot3d);library(lattice);library(rsm);library(splines);library(pROC)
library(mgcv);library(FactoMineR)
library(car);library(gam);library(dplyr);library(MuMIn);library(leaps); library(caret);library(lmtest)
library(lsr);library(relaimpo);library(splines);library(reshape);library(tidyverse);library(dslabs)
library(ROCR);library(MASS);library(bestglm);library(reshape2);library(corrplot);library(ggplot2)
library(plyr);library(readr);library(klaR);library(gratia);library(patchwork); library(dplyr)
library(cowplot); library(pROC);library(combinat)
a <- read.csv("df_nuevo_0.45.csv"); head(a); dim(a)

# ---- 2. Fórmula global (lo único que se edita) ------------
fml <- q_memory_iso_ln ~ area_km2_ln + aridity_cr2met_1979_2010_ln +
  T_basin + z_basin_ln + rain + snowfall_ln +
  frac_snow_cr2met_1979_2010_ln + p_mean_cr2met_1979_2010

# ---- 3. Filas completas (las variables salen de la fórmula)
a2 <- na.omit(a[, all.vars(fml)])
nrow(a2)

# ---- 4. Modelos armados a mano (reemplaza a dredge) -------
resp <- all.vars(fml)[1]
vv   <- all.vars(fml)[-1]

# Helper: todas las combinaciones de tamaño 0..kmax
combos <- function(v, kmax = length(v)) {
  v <- intersect(v, vv)
  if (length(v) == 0) return(list(character(0)))
  unlist(lapply(0:min(kmax, length(v)),
                function(k) combn(v, k, simplify = FALSE)),
         recursive = FALSE)
}

# Grupos
opt_area  <- combos("area_km2_ln")                                  # con o sin área (2)

opt_clima <- combos(c("aridity_cr2met_1979_2010_ln",
                      "p_mean_cr2met_1979_2010",
                      "rain"), kmax = 1)                            # MÁXIMO 1 (4)

#opt_nieve <- combos(c("T_basin", "z_basin_ln", "snowfall_ln",
                   #   "frac_snow_cr2met_1979_2010_ln"), kmax = 2)   # MÁXIMO 2 (11)

# Grupo nieve/temperatura/elevación
vars_nieve <- intersect(c("T_basin", "z_basin_ln", "snowfall_ln",
                          "frac_snow_cr2met_1979_2010_ln"), vv)
prohibido  <- c("snowfall_ln", "frac_snow_cr2met_1979_2010_ln")   # no pueden ir juntas

sin_nada <- list(character(0))                                    # ninguna
solas    <- as.list(vars_nieve)                                   # una sola

pares <- combn(vars_nieve, 2, simplify = FALSE)
pares <- Filter(function(p) !all(prohibido %in% p), pares)        # saca el par prohibido

aditivos <- pares                                                 # a + b
interac  <- lapply(pares, function(p) paste(p, collapse = "*"))   # a * b (incluye a + b + a:b)

opt_nieve <- c(sin_nada, solas, aditivos, interac)
length(opt_nieve)   # 1 + 4 + 5 + 5 = 15

sapply(list(area = opt_area, clima = opt_clima, nieve = opt_nieve), length)

g <- expand.grid(a = seq_along(opt_area),
                 w = seq_along(opt_clima),
                 n = seq_along(opt_nieve))
nrow(g)    # 2 x 4 x 11 = 88

mods <- vector("list", nrow(g))
for (i in seq_len(nrow(g))) {
  terminos <- c(opt_area[[g$a[i]]], opt_clima[[g$w[i]]], opt_nieve[[g$n[i]]])
  f <- if (length(terminos) == 0) as.formula(paste(resp, "~ 1"))
  else reformulate(terminos, response = resp)
  mods[[i]] <- lm(f, data = a2)
}

# ---- 5. Tabla de resultados (equivalente a la de dredge) --
dd2 <- data.frame(
  modelo = sapply(mods, function(m) paste(attr(terms(m), "term.labels"), collapse = " + ")),
  df     = sapply(mods, function(m) attr(logLik(m), "df")),
  logLik = sapply(mods, function(m) as.numeric(logLik(m))),
  AICc   = sapply(mods, AICc),
  R2     = sapply(mods, function(m) summary(m)$r.squared),
  R2adj  = sapply(mods, function(m) summary(m)$adj.r.squared)
)
dd2$delta  <- dd2$AICc - min(dd2$AICc)
dd2$weight <- exp(-dd2$delta / 2) / sum(exp(-dd2$delta / 2))

ord  <- order(dd2$AICc)
dd2  <- dd2[ord, ]
mods <- mods[ord]

# ---- 6. Resultados ----------------------------------------
head(dd2, 20)
summary(mods[[1]])



mod    <- lm(q_memory_iso_ln~area_km2_ln  +rain                       + z_basin_ln+T_basin,data=a);summary(mod);AIC(mod)

win.graph();par(mfrow=c(2,2));termplot(mod, partial.resid = TRUE, se = TRUE)


boot <- boot.relimp(mod, b = 1000, type = "lmg", rank = TRUE, diff = TRUE, rela = TRUE)
booteval.relimp(boot)
win.graph();plot(booteval.relimp(boot))



mod <- lm(  q_memory_iso_ln ~ area_km2_ln +    rain +    T_basin * z_basin_ln,  data = a)
summary(mod)
AIC(mod)

mf <- model.frame(mod)

v_y <- "q_memory_iso_ln"
v_x <- "T_basin"
v_z <- "z_basin_ln"
v_area <- "area_km2_ln"
v_pr <- "rain"

lvl <- function(x) c(min(x), mean(x), max(x))
cols <- c("blue", "darkgreen", "red")
labs <- c("mín", "media", "máx")

# ============================================================
# FUNCIÓN DE PREDICCIÓN
# Varía xvar y fija mod_var en mod_val.
# Las demás covariables se mantienen en sus medias.
# ============================================================

curva <- function(xvar, xseq, mod_var = NULL, mod_val = NULL) {
  
  nd <- data.frame(
    area_km2_ln = mean(mf$area_km2_ln),
    rain = mean(mf$rain),
    T_basin=
      mean(mf$T_basin),
    z_basin_ln = mean(mf$z_basin_ln)
  )
  
  nd <- nd[rep(1, length(xseq)), , drop = FALSE]
  
  nd[[xvar]] <- xseq
  
  if (!is.null(mod_var)) {
    nd[[mod_var]] <- mod_val
  }
  
  predict(mod, newdata = nd)
}

# ============================================================
# GRÁFICOS
# ============================================================

win.graph(width = 12, height = 10)
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3, 1))

# ------------------------------------------------------------
# PANEL 1: FRACCIÓN DE NIEVE SEGÚN z_basin
# ------------------------------------------------------------

xs <- seq(
  min(mf[[v_x]], na.rm = TRUE),
  max(mf[[v_x]], na.rm = TRUE),
  length.out = 100
)

plot(
  mf[[v_x]], mf[[v_y]],
  pch = 16, col = "grey70",
  xlab = v_x, ylab = v_y,
  main = "Fracción de nieve según z_basin"
)

for (i in 1:3) {
  lines(
    xs,
    curva(v_x, xs, v_z, lvl(mf[[v_z]])[i]),
    col = cols[i], lwd = 2
  )
}

legend(
  "topright",
  legend = paste("z_basin", labs),
  col = cols, lwd = 2, bty = "n"
)

# ------------------------------------------------------------
# PANEL 2: z_basin SEGÚN FRACCIÓN DE NIEVE
# ------------------------------------------------------------

xs2 <- seq(
  min(mf[[v_z]], na.rm = TRUE),
  max(mf[[v_z]], na.rm = TRUE),
  length.out = 100
)

plot(
  mf[[v_z]], mf[[v_y]],
  pch = 16, col = "grey70",
  xlab = v_z, ylab = v_y,
  main = "z_basin según fracción de nieve"
)

for (i in 1:3) {
  lines(
    xs2,
    curva(v_z, xs2, v_x, lvl(mf[[v_x]])[i]),
    col = cols[i], lwd = 2
  )
}

legend(
  "topright",
  legend = paste("Fracción nieve", labs),
  col = cols, lwd = 2, bty = "n"
)

# ------------------------------------------------------------
# PANEL 3: ÁREA DE LA CUENCA
# ------------------------------------------------------------

xs3 <- seq(
  min(mf[[v_area]], na.rm = TRUE),
  max(mf[[v_area]], na.rm = TRUE),
  length.out = 100
)

plot(
  mf[[v_area]], mf[[v_y]],
  pch = 16, col = "grey70",
  xlab = v_area, ylab = v_y,
  main = "Efecto del área de la cuenca"
)

lines(
  xs3,
  curva(v_area, xs3),
  col = "blue", lwd = 2
)

# ------------------------------------------------------------
# PANEL 4: PRECIPITACIÓN MEDIA
# ------------------------------------------------------------

xs4 <- seq(
  min(mf[[v_pr]], na.rm = TRUE),
  max(mf[[v_pr]], na.rm = TRUE),
  length.out = 100
)

plot(
  mf[[v_pr]], mf[[v_y]],
  pch = 16, col = "grey70",
  xlab = v_pr, ylab = v_y,
  main = "Efecto de la precipitación media"
)

lines(
  xs4,
  curva(v_pr, xs4),
  col = "blue", lwd = 2
)




