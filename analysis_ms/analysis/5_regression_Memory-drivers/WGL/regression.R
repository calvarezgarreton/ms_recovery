setwd("C:/DOCUMENTS/EARTH_SCIENCES/CLIMA/2027_Alvarez-Garreton/ANALYSIS/codigos_ivan/ms_recovery-main/analysis/5_regression_Memory-drivers/WGL")
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
fml <- memory_iso_raw ~ UP_AREA_raw_ln + AI_basinmean_ln +
  dist_flujo_A500_ln +dist_recta_A500_ln+ dist_mismo_A500_ln+ dist_cualq_A500_ln+dist_flujo_A1000_ln +dist_recta_A1000_ln +dist_mismo_A1000_ln +
  dist_cualq_A1000_ln+dist_flujo_A1500_ln+ dist_recta_A1500_ln+ dist_mismo_A1500_ln+ dist_cualq_A1500_ln+dist_flujo_Q0p5_ln +dist_recta_Q0p5_ln+
  dist_mismo_Q0p5_ln+dist_cualq_Q0p5_ln+dist_flujo_Q1p5_ln +dist_recta_Q1p5_ln +dist_mismo_Q1p5_ln+ dist_cualq_Q1p5_ln+
  dist_recta_Q2p5_ln+ dist_mismo_Q2p5_ln +dist_cualq_Q2p5_ln+dist_flujo_Q2p5_ln+ perm2_f + imp2_f+ imp3_f+gwl_mean_balanced_ln+ksat_basinmean_ln+awc_basinmean

# ---- 3. Filas completas (las variables salen de la fórmula)
a2 <- na.omit(a[, all.vars(fml)])
nrow(a2)

# ---- 4. Los 300 modelos armados a mano (reemplaza a dredge)
resp      <- all.vars(fml)[1]
dist_vars <- grep("^dist_", all.vars(fml), value = TRUE)

opt_dist <- c(list(character(0)), as.list(dist_vars))   # ninguna o UNA distancia (25)

vars_cont <- c("UP_AREA_raw_ln",
               "AI_basinmean_ln",
               "gwl_mean_balanced_ln")

# todas las combinaciones posibles (incluida ninguna): 2^3 = 8
opt_cont <- unlist(lapply(0:length(vars_cont),
                          function(k) combn(vars_cont, k, simplify = FALSE)),
                   recursive = FALSE)
length(opt_cont)   # 8

opt_fac  <- list(character(0), "perm2_f", "imp2_f", "imp3_f")        # ninguna o UNA categórica (4)

opt_suelo <- list(character(0), "ksat_basinmean_ln", "awc_basinmean")  # ninguna o UNA (3)

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
vars_neg <- c("ksat_basinmean_ln", "gwl_mean_balanced_ln")   # deben ser <= 0
vars_pos <- c("awc_basinmean")                                # deben ser >= 0

plausible <- function(m, neg = vars_neg, pos = vars_pos) {
  b <- coef(m)
  all(b[names(b) %in% neg] <= 0) && all(b[names(b) %in% pos] >= 0)
}

keep <- sapply(mods, plausible)
table(keep)
mods <- mods[keep]

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

mod    <- lm(memory_iso_raw~ dist_cualq_Q0p5_ln  *AI_basinmean_ln+ UP_AREA_raw_ln,data=a);summary(mod);AIC(mod)
mod_add <- lm(memory_iso_raw ~ dist_cualq_Q0p5_ln + AI_basinmean_ln + UP_AREA_raw_ln, data = a);summary(mod_add);AIC(mod_add)

win.graph();par(mfrow=c(2,3));termplot(mod, partial.resid = TRUE, se = TRUE)

boot <- boot.relimp(mod, b = 1000, type = "lmg", rank = TRUE, diff = TRUE, rela = TRUE)
booteval.relimp(boot)
win.graph();plot(booteval.relimp(boot))










library(ggplot2)
library(patchwork)

mod <- lm(memory_iso_raw ~ dist_cualq_Q0p5_ln * AI_basinmean_ln + UP_AREA_raw_ln, data = a)
d   <- model.frame(mod)          # solo las filas usadas en el modelo

# ---- 1. Valores mín / promedio / máx ----------------------
rng <- function(v) c(Mínimo = min(v), Promedio = mean(v), Máximo = max(v))
r_dist <- rng(d$dist_cualq_Q0p5_ln)
r_ai   <- rng(d$AI_basinmean_ln)
m_area <- mean(d$UP_AREA_raw_ln)

# ---- 2. Pendientes simples (con IC 95%) -------------------
b <- coef(mod); V <- vcov(mod); tcrit <- qt(0.975, df.residual(mod))
nm_int <- "dist_cualq_Q0p5_ln:AI_basinmean_ln"

simple_slope <- function(var, mod_var, vals) {
  # efecto de 'var' en cada valor de la otra variable (mod_var)
  est <- b[var] + b[nm_int] * vals
  se  <- sqrt(V[var, var] + vals^2 * V[nm_int, nm_int] + 2 * vals * V[var, nm_int])
  data.frame(efecto_de = var, en_nivel_de = mod_var,
             nivel = names(vals), valor = round(vals, 3),
             pendiente = est, se = se,
             IC_inf = est - tcrit * se, IC_sup = est + tcrit * se,
             p = 2 * pt(-abs(est / se), df.residual(mod)),
             row.names = NULL)
}

slopes <- rbind(
  simple_slope("dist_cualq_Q0p5_ln", "AI_basinmean_ln",     r_ai),
  simple_slope("AI_basinmean_ln",    "dist_cualq_Q0p5_ln",  r_dist)
)
print(slopes, digits = 3)

# efecto "normal" de UP_AREA (sin interacción)
print(cbind(coef = b["UP_AREA_raw_ln"],
            confint(mod)["UP_AREA_raw_ln", , drop = FALSE]))

# ---- 3. Gráficos de la interacción ------------------------
plot_int <- function(xvar, zvar, zvals, xlab, zlab) {
  xs <- seq(min(d[[xvar]]), max(d[[xvar]]), length.out = 100)
  nd <- expand.grid(xval = xs, nivel = names(zvals))
  nd[[xvar]] <- nd$xval
  nd[[zvar]] <- zvals[as.character(nd$nivel)]
  nd$UP_AREA_raw_ln <- m_area
  p <- cbind(nd, predict(mod, nd, interval = "confidence"))
  p$nivel <- factor(p$nivel, levels = names(zvals),
                    labels = paste0(names(zvals), " (", round(zvals, 2), ")"))
  ggplot(p, aes(xval, fit, colour = nivel, fill = nivel)) +
    geom_ribbon(aes(ymin = lwr, ymax = upr), alpha = 0.12, colour = NA) +
    geom_line(linewidth = 1) +
    labs(x = xlab, y = "memory_iso_raw", colour = zlab, fill = zlab) +
    theme_bw()
}

p1 <- plot_int("dist_cualq_Q0p5_ln", "AI_basinmean_ln", r_ai,
               "dist_cualq_Q0p5_ln", "AI_basinmean_ln")
p2 <- plot_int("AI_basinmean_ln", "dist_cualq_Q0p5_ln", r_dist,
               "AI_basinmean_ln", "dist_cualq_Q0p5_ln")

# ---- 4. Efecto de UP_AREA (sin interacción) ---------------
xs <- seq(min(d$UP_AREA_raw_ln), max(d$UP_AREA_raw_ln), length.out = 100)
nd <- data.frame(UP_AREA_raw_ln = xs,
                 dist_cualq_Q0p5_ln = mean(d$dist_cualq_Q0p5_ln),
                 AI_basinmean_ln    = mean(d$AI_basinmean_ln))
p3d <- cbind(nd, predict(mod, nd, interval = "confidence"))

p3 <- ggplot(p3d, aes(UP_AREA_raw_ln, fit)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr), alpha = 0.15) +
  geom_line(linewidth = 1) +
  labs(x = "UP_AREA_raw_ln", y = "memory_iso_raw",
       subtitle = "Otras variables en su promedio") +
  theme_bw()

# ---- 5. Mostrar gráficos en ventana propia ----------------
win.graph(width = 12, height = 9)     # ventana grande para los 3 paneles
print((p1 | p2) / p3 + plot_annotation(tag_levels = "A"))


























# ---- 2. Fórmula global (lo único que se edita) ------------
fml <- memory_iso_raw ~ UP_AREA_raw_ln + AI_basinmean_ln +
  dist_flujo_A500_ln +dist_recta_A500_ln+ dist_mismo_A500_ln+ dist_cualq_A500_ln+dist_flujo_A1000_ln +dist_recta_A1000_ln +dist_mismo_A1000_ln +
  dist_cualq_A1000_ln+dist_flujo_A1500_ln+ dist_recta_A1500_ln+ dist_mismo_A1500_ln+ dist_cualq_A1500_ln+dist_flujo_Q0p5_ln +dist_recta_Q0p5_ln+
  dist_mismo_Q0p5_ln+dist_cualq_Q0p5_ln+dist_flujo_Q1p5_ln +dist_recta_Q1p5_ln +dist_mismo_Q1p5_ln+ dist_cualq_Q1p5_ln+
  dist_recta_Q2p5_ln+ dist_mismo_Q2p5_ln +dist_cualq_Q2p5_ln+dist_flujo_Q2p5_ln

# ---- 3. Filas completas (las variables salen de la fórmula)
a2 <- na.omit(a[, all.vars(fml)])
nrow(a2)

# ---- 4. Los 300 modelos armados a mano (reemplaza a dredge)
resp      <- all.vars(fml)[1]
dist_vars <- grep("^dist_", all.vars(fml), value = TRUE)

opt_dist <- c(list(character(0)), as.list(dist_vars))   # ninguna o UNA distancia (25)

vars_cont <- c("UP_AREA_raw_ln",
               "AI_basinmean_ln")

# todas las combinaciones posibles (incluida ninguna): 2^3 = 8
opt_cont <- unlist(lapply(0:length(vars_cont),
                          function(k) combn(vars_cont, k, simplify = FALSE)),
                   recursive = FALSE)
length(opt_cont)   # 8

g <- expand.grid(d = seq_along(opt_dist), c = seq_along(opt_cont))
        #         f = seq_along(opt_fac),  s = seq_along(opt_suelo))
nrow(g)    # 25 x 5 x 4 x 3 = 1500

mods <- vector("list", nrow(g))
for (i in seq_len(nrow(g))) {
  terminos <- c(opt_dist[[g$d[i]]], opt_cont[[g$c[i]]])
     #           opt_fac[[g$f[i]]],  opt_suelo[[g$s[i]]])
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
mods <- mods[ord]                       # mismo orden que la tabla

# ---- 6. Resultados ----------------------------------------
head(dd2, 20)
summary(mods[[1]])                      # mejor modelo

mod    <- lm(memory_iso_raw~ dist_cualq_Q0p5_ln  + AI_basinmean_ln+ UP_AREA_raw_ln,data=a);summary(mod);AIC(mod)

win.graph();par(mfrow=c(2,3));termplot(mod, partial.resid = TRUE, se = TRUE)

boot <- boot.relimp(mod, b = 1000, type = "lmg", rank = TRUE, diff = TRUE, rela = TRUE)
booteval.relimp(boot)
win.graph();plot(booteval.relimp(boot))










