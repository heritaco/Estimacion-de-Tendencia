## Hodrick–Prescott (Hodrick & Prescott, 1997)

$$
\min_{\{\tau_t\}}\;\sum_{t=1}^{T}\big(y_t-\tau_t\big)^2\;+\;\lambda\sum_{t=3}^{T}\big(\Delta^2\tau_t\big)^2,\qquad \Delta^2\tau_t=\tau_t-2\tau_{t-1}+\tau_{t-2}
$$

**Objetivo**

Extraer tendencia penalizando curvatura.

**Datos**

Anual $\lambda\!\approx\!6.25$. Trimestral $\lambda\!\approx\!1600$. Mensual $\lambda\!\approx\!129{,}600$.

## Hamilton (Hamilton, 2018)

$$
\widehat{\tau}_t=\widehat{\mathbb E}\!\left(y_{t+h}\mid y_t,\ldots,y_{t-p}\right),\qquad \widehat{c}_t=y_t-\widehat{\tau}_t
$$

**Objetivo**

Tendencia por proyección a $h$ pasos sin parámetro de suavidad.

**Datos**

Elegir $h,p$ según horizonte y frecuencia.

## Componentes no observados / Local Level (Harvey, 1989)

$$
\begin{aligned}
y_t&=\tau_t+c_t+\varepsilon_t,\\
\tau_t&=\tau_{t-1}+\beta_{t-1}+\eta_t,\quad
\beta_t=\beta_{t-1}+\zeta_t
\end{aligned}
$$

**Objetivo**

Tendencia estocástica vía filtro y suavizado de Kalman.

**Datos**

Varianzas por ML; extensible a estacionalidad y regresores.

## Beveridge–Nelson (Beveridge & Nelson, 1981)

$$
\tau_t=\lim_{h\to\infty}\mathbb{E}(y_{t+h}\mid\mathcal F_t),\qquad c_t=y_t-\tau_t
$$

**Objetivo**

Separar componente permanente en ARIMA.

**Datos**

Requiere ARIMA estimado con raíz unitaria.

## Baxter–King (Baxter & King, 1999)

$$
\tilde y_t=\sum_{k=-K}^{K} b_k\,y_{t-k},\quad H(\omega)\approx\mathbf 1_{\omega\in[\omega_L,\omega_H]}
$$

**Objetivo**

Pasa-banda simétrico para ciclo 6–32 trimestres.

**Datos**

MA finito; ventana $2K{+}1$.

## Christiano–Fitzgerald (Christiano & Fitzgerald, 2003)

$$
\tilde y_t=\sum_{k=-K}^{K} b_k(t)\,y_{t-k}
$$

**Objetivo**

Pasa-banda cuasi-óptimo con mejor borde.

**Datos**

Coeficientes dependientes de $t$.

## Butterworth discreto (Butterworth, 1930)

$$
|H(\omega)|^2=\frac{1}{1+(\omega/\omega_c)^{2m}}
$$

**Objetivo**

Pasa-bajos IIR con corte $\omega_c$.

**Datos**

Elegir orden $m$ y $\omega_c$.

## Savitzky–Golay / Henderson (Savitzky & Golay, 1964; Henderson, 1916)

$$
\widehat{\tau}_t=\sum_{k=-K}^{K}w_k\,y_{t-k}\quad\text{(ajuste polinómico local grado }d\text{)}
$$

**Objetivo**

Suavizar preservando forma.

**Datos**

Ventana $2K{+}1$, grado $d$.

## STL (Cleveland et al., 1990)

$$
y_t=\tau_t+s_t+r_t
$$

**Objetivo**

Descomposición LOESS robusta con estacionalidad variable.

**Datos**

Spans separados para $s_t$ y $\tau_t$.

## X-13-ARIMA-SEATS (U.S. Census, 2013; Gómez & Maravall, 1996/2001)

**Objetivo**

Ajuste estacional oficial y tendencia Henderson.

**Datos**

regARIMA con calendario y outliers.

## Trend filtering $L^1$ (Kim et al., 2009; Tibshirani, 2014)

$$
\min_{\tau}\;\tfrac12\|y-\tau\|_2^2+\lambda\|\Delta^2\tau\|_1
$$

**Objetivo**

Tendencias por tramos con pocos quiebres.

**Datos**

$\lambda$ por validación/SURE.

## P-splines / Splines cúbicos (Eilers & Marx, 1996; Green & Silverman, 1994)

$$
\min_f\;\sum_t\big(y_t-f(t)\big)^2+\lambda\!\int \big(f''(t)\big)^2 dt
$$

**Objetivo**

Suavizado penalizado equivalente continuo del HP.

**Datos**

Nudos + penalización $\lambda$.

## Wavelets / MODWT (Mallat, 1989; Daubechies, 1992; Percival & Walden, 2000)

$$
y_t=\sum_{j=1}^{J} d_{j,t}+a_{J,t},\quad \text{tendencia}\equiv a_{J,t}
$$

**Objetivo**

Separación multiescala.

**Datos**

Base y nivel $J$.

## SSA (Golyandina et al., 2001)

$$
\mathbf{H}=\text{Hankel}(y)\approx \sum_{r=1}^{R}\sigma_r u_r v_r^\top
$$

**Objetivo**

Tendencia como bajo rango.

**Datos**

Ventana $L$, rango $R$.

## EMD / CEEMDAN (Huang et al., 1998; Torres et al., 2011)

$$
y_t=\sum_{k=1}^{K}\text{IMF}_{k,t}+r_t,\quad \text{tendencia}\equiv r_t
$$

**Objetivo**

Descomposición adaptativa.

**Datos**

Control de modos espurios.

## Gaussian Processes (Rasmussen & Williams, 2006)

$$
y\sim \mathcal{GP}\!\left(0,\;k_{\text{SE}}+k_{\text{per}}+k_{\text{WN}}\right)
$$

**Objetivo**

Tendencia como media posterior.

**Datos**

Hiperparámetros por ML marginal.

## Suavizado exponencial (Holt, 1957; Winters, 1960)

$$
\text{SES:}\;\tau_t=\alpha y_t+(1-\alpha)\tau_{t-1}
$$

**Objetivo**

Tendencia operativa para pronósticos.

**Datos**

$\alpha$ y extensiones $(\beta,\gamma)$ según estacionalidad.
