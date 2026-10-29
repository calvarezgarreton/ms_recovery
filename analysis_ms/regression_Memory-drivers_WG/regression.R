setwd("C:/DOCUMENTS/EARTH_SCIENCES/CLIMA/2027_Alvarez-Garreton/R_nue1/R_prediction/small")
library(fBasics);library(stats);library(scatterplot3d);library(lattice);library(rsm);library(splines);library(pROC)
library(mgcv);library(FactoMineR)
library(car);library(gam);library(dplyr);library(MuMIn);library(leaps); library(caret);library(lmtest)
library(lsr);library(relaimpo);library(splines);library(reshape);library(tidyverse);library(dslabs)
library(ROCR);library(MASS);library(bestglm);library(reshape2);library(corrplot);library(ggplot2)
library(plyr);library(readr);library(klaR);library(gratia);library(patchwork); library(dplyr)
library(cowplot); library(pROC);library(combinat)
a <- read.csv("df_nuevo_0.45.csv"); head(a); dim(a)

# ---- 1. Factores (una sola vez) ---------------------------
a$perm2_f <- factor(a$perm2)
a$imp2_f  <- factor(a$imp2)
a$imp3_f  <- factor(a$imp3)

# ---- 2. Fórmula global (lo único que se edita) ------------
fml <- memory_iso_raw ~ UP_AREA_raw_ln + AI_tr_ln +
  dist_flujo_A500_ln +dist_recta_A500_ln+ dist_mismo_A500_ln+ dist_cualq_A500_ln+dist_flujo_A1000_ln +dist_recta_A1000_ln +dist_mismo_A1000_ln +
  dist_cualq_A1000_ln+dist_flujo_A2000_ln+ dist_recta_A2000_ln+ dist_mismo_A2000_ln+ dist_cualq_A2000_ln+dist_flujo_Q0p5_ln +dist_recta_Q0p5_ln+
  dist_mismo_Q0p5_ln+dist_cualq_Q0p5_ln+dist_flujo_Q1p5_ln +dist_recta_Q1p5_ln +dist_mismo_Q1p5_ln+ dist_cualq_Q1p5_ln+
  dist_recta_Q2p5_ln+ dist_mismo_Q2p5_ln +dist_cualq_Q2p5_ln+dist_flujo_Q2p5_ln+ perm2_f + imp2_f+ imp3_f+cr2sub_clean_mean_gwl_ln+ksat_ln+Total_AWC_0_200cm

# ---- 3. Filas completas (las variables salen de la fórmula)
a2 <- na.omit(a[, all.vars(fml)])
nrow(a2)

# ---- 4. Los 300 modelos armados a mano (reemplaza a dredge)
resp      <- all.vars(fml)[1]
dist_vars <- grep("^dist_", all.vars(fml), value = TRUE)

opt_dist <- c(list(character(0)), as.list(dist_vars))   # ninguna o UNA distancia (25)

vars_cont <- c("UP_AREA_raw_ln",
               "AI_tr_ln",
               "cr2sub_clean_mean_gwl_ln")

# todas las combinaciones posibles (incluida ninguna): 2^3 = 8
opt_cont <- unlist(lapply(0:length(vars_cont),
                          function(k) combn(vars_cont, k, simplify = FALSE)),
                   recursive = FALSE)
length(opt_cont)   # 8

opt_fac  <- list(character(0), "perm2_f", "imp2_f", "imp3_f")        # ninguna o UNA categórica (4)

opt_suelo <- list(character(0), "ksat_ln", "Total_AWC_0_200cm")  # ninguna o UNA (3)

g <- expand.grid(d = seq_along(opt_dist), c = seq_along(opt_cont),
                 f = seq_along(opt_fac),  s = seq_along(opt_suelo))
nrow(g)    # 25 x 5 x 4 x 3 = 1500

mods <- vector("list", nrow(g))
for (i in seq_len(nrow(g))) {
  terminos <- c(opt_dist[[g$d[i]]], opt_cont[[g$c[i]]],
                opt_fac[[g$f[i]]],  opt_suelo[[g$s[i]]])
  f <- if (length(terminos) == 0) as.formula(paste(resp, "~ 1"))
  else reformulate(terminos, response = resp)
  mods[[i]] <- lm(f, data = a2)
}

# ---- 4b. Filtro de plausibilidad física ---------------------
# Se descarta el modelo si algún coeficiente de estas variables es > 0
vars_neg <- c("ksat_ln", "cr2sub_clean_mean_gwl_ln")

plausible <- function(m, vars = vars_neg) {
  b <- coef(m)
  b <- b[names(b) %in% vars]
  all(b <= 0)          # si el modelo no incluye ninguna de las dos, all(logical(0)) = TRUE
}

keep <- sapply(mods, plausible)
table(keep)            # cuántos se descartan y cuántos quedan
mods <- mods[keep]     # el paso 5 en adelante queda igual

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
mods <- mods[ord]                       # mismo orden que la tabla

# ---- 6. Resultados ----------------------------------------
head(dd2, 20)
summary(mods[[1]])                      # mejor modelo

mod    <- lm(memory_iso_raw~ dist_cualq_Q0p5_ln  + AI_tr_ln+ UP_AREA_raw_ln,data=a);summary(mod);AIC(mod)

win.graph();par(mfrow=c(2,3));termplot(mod, partial.resid = TRUE, se = TRUE)

boot <- boot.relimp(mod, b = 1000, type = "lmg", rank = TRUE, diff = TRUE, rela = TRUE)
booteval.relimp(boot)
win.graph();plot(booteval.relimp(boot))

























