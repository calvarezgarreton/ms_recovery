setwd("C:/DOCUMENTS/EARTH_SCIENCES/CLIMA/2027_Alvarez-Garreton/R_nue1/R_prediction/Q")
library(fBasics);library(stats);library(scatterplot3d);library(lattice);library(rsm);library(splines);library(pROC)
library(mgcv);library(FactoMineR)
library(car);library(gam);library(dplyr);library(MuMIn);library(leaps); library(caret);library(lmtest)
library(lsr);library(relaimpo);library(splines);library(reshape);library(tidyverse);library(dslabs)
library(ROCR);library(MASS);library(bestglm);library(reshape2);library(corrplot);library(ggplot2)
library(plyr);library(readr);library(klaR);library(gratia);library(patchwork); library(dplyr)
library(cowplot); library(pROC);library(combinat)

a<-read.csv("Q.csv");head(a);dim(a) #;View(a)
b <- dplyr::filter(a, r2 >= 0.45)
dim(b)


# ==========================================================
# Normalización de variables asimétricas con log + constante
# ==========================================================
library(e1071)   # skewness()

# ---- 1. Funciones auxiliares ------------------------------

# Asimetría ignorando NA
sk <- function(x) skewness(x, na.rm = TRUE)

# Busca la constante que lleva la asimetría a 0
#  - asimetría > 0: log(x + c)    -> devuelve c
#  - asimetría < 0: -log(c - x)   -> devuelve c
find_const <- function(x) {
  x   <- x[is.finite(x)]
  s0  <- sk(x)
  rng <- diff(range(x))
  
  # d = desplazamiento > 0 para que el argumento del log sea siempre positivo
  g <- if (s0 > 0) {
    function(d) sk(log(x - min(x) + d))
  } else {
    function(d) sk(-log(max(x) - x + d))
  }
  
  d_grid <- rng * 10^seq(-6, 6, length.out = 300)
  s_grid <- sapply(d_grid, g)
  i <- which(diff(sign(s_grid)) != 0)[1]       # primer cambio de signo
  
  if (is.na(i)) {                              # sin cruce por 0: mejor aproximación
    d <- d_grid[which.min(abs(s_grid))]
  } else {
    d <- uniroot(g, c(d_grid[i], d_grid[i + 1]), tol = 1e-10)$root
  }
  
  list(const = if (s0 > 0) d - min(x) else max(x) + d)
}

# Constante -> texto válido para nombre de variable (12.5 -> "12p5")
nombre_const <- function(cst) {
  s <- format(abs(cst), scientific = FALSE, trim = TRUE)
  gsub("\\.", "p", s)
}

# ---- 2. Transformación de las variables -------------------

vars <- names(b)[sapply(b, is.numeric)]
res  <- data.frame()

for (v in vars) {
  x  <- b[[v]]
  s0 <- sk(x)
  if (is.na(s0) || abs(s0) <= 0.5) next        # solo |asimetría| > 0.5
  
  cst <- signif(find_const(x)$const, 5)        # constante redondeada a 5 cifras
  nuevo <- paste0(v, "_ln")                    # nombre simple, sin constante
  
  if (s0 > 0) {
    # log(x + c): el argumento debe ser > 0
    if (min(x, na.rm = TRUE) + cst <= 0) cst <- find_const(x)$const
    b[[nuevo]] <- log(x + cst)
    formula <- paste0("log(x + ", cst, ")")
  } else {
    # -log(c - x): c debe ser > max(x); conserva el orden de la variable
    if (cst <= max(x, na.rm = TRUE)) cst <- find_const(x)$const
    b[[nuevo]] <- -log(cst - x)
    formula <- paste0("-log(", cst, " - x)")
  }
  
  s1  <- sk(b[[nuevo]])
  res <- rbind(res, data.frame(
    variable  = v,
    nueva     = nuevo,
    tipo      = if (s0 > 0) "log(x+c)" else "-log(c-x)",
    const     = cst,
    formula   = formula,
    skew_orig = round(s0, 3),
    skew_new  = round(s1, 4),
    ok        = abs(s1) <= 0.05
  ))
}

# ---- 3. Resultados ----------------------------------------

res                                            # tabla de transformaciones
write.csv(res, "transformaciones_0.45.csv", row.names = FALSE)
write.csv(b, "df_nuevo_0.45.csv", row.names = FALSE)












