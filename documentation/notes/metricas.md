Resumen práctico: **más alto** log-likelihood es mejor; **más bajo** AIC/BIC/HQIC es mejor. Tus números implican $T=314$ y $k=5$ parámetros (const, AR1, AR2, MA1, $\sigma^2$); se verifica porque
$-2\ell+2k=-2035.763\Rightarrow k\approx5$.

# Definiciones, fórmula, intuición y uso

## Log Likelihood $\ell(\hat\theta)$

**Qué es.** Verosimilitud del modelo en los datos con parámetros estimados.
**Fórmula (general).**

$$
\ell(\theta)=\sum_{t=1}^{T}\log f_\theta(y_t\mid\mathcal F_{t-1}).
$$

Para ARIMA gaussiano vía filtro de innovaciones:

$$
\ell(\theta)=-\tfrac{T}{2}\log(2\pi)-\tfrac{1}{2}\sum_{t=1}^{T}\Big(\log v_t+\tfrac{\varepsilon_t^2}{v_t}\Big),
$$

donde $\varepsilon_t$ y $v_t$ son error y varianza de predicción 1-paso.
**Intuición.** Cuánto “probable” hace el modelo a los datos.
**Cuándo usar.** Nunca solo para comparar entre modelos con distinto $k$; úsalo como insumo de AIC/BIC/HQIC.

## AIC (Akaike Information Criterion)

**Fórmula.**

$$
\mathrm{AIC}=2k-2\ell(\hat\theta).
$$

**Intuición.** Trade-off ajuste-complejidad con penalización **constante** por parámetro; aproxima el riesgo de predicción fuera de muestra (K-L risk).
**Uso recomendado.** Selección enfocada en **pronóstico** cuando el “modelo verdadero” puede ser complejo o no está en la familia candidata. Preferir **AICc** en muestras pequeñas:

$$
\mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{T-k-1}.
$$

**Regla práctica.** Comparar $\Delta_i=\mathrm{AIC}_i-\mathrm{AIC}_{\min}$; $\Delta\le2$ soporte sustancial, $4\!-\!7$ moderado, $\ge 10$ casi nulo. Pesos de Akaike:

$$
w_i=\frac{\exp(-\Delta_i/2)}{\sum_j \exp(-\Delta_j/2)}.
$$

## BIC (Bayesian Information Criterion, Schwarz)

**Fórmula.**

$$
\mathrm{BIC}=k\log T-2\ell(\hat\theta).
$$

**Intuición.** Penalización **creciente con $T$**; aproximación de la evidencia bayesiana (margen) con prior “unit information”.
**Uso recomendado.** **Identificación estructural** y parsimonia; es **consistente** si el verdadero modelo finito está entre los candidatos.
**Regla práctica.** Diferencias $\Delta\mathrm{BIC}$ se interpretan como log-Bayes factors aproximados: evidencia positiva $2\!-\!6$, fuerte $6\!-\!10$, muy fuerte $>10$.

## HQIC (Hannan–Quinn)

**Fórmula.**

$$
\mathrm{HQIC}=2k\log\log T-2\ell(\hat\theta).
$$

**Intuición.** Penalización intermedia entre AIC y BIC; **consistente** como BIC pero menos severo en muestras moderadas.
**Uso recomendado.** Cuando BIC parece “sub-ajustar” y AIC “sobre-ajustar”; punto medio prudente.

# Verificación con tus cifras

$$
\begin{aligned}
\ell&=1022.881, \quad T=314,\quad k=5,\\
\mathrm{AIC}&=2(5)-2(1022.881)=-2035.763,\\
\mathrm{BIC}&=5\ln(314)-2(1022.881)=-2017.016,\\
\mathrm{HQIC}&=2(5)\ln\ln(314)-2(1022.881)=-2028.272.
\end{aligned}
$$

# ¿Cuál conviene?

* **Pronóstico puro / modelos candidatos todos “mal-especificados”:** AIC o AICc.
* **Parquedad y posible “modelo verdadero” finito:** BIC.
* **Compromiso ajuste-parsimonia cuando AIC y BIC discrepan:** HQIC.
* **Series muy largas:** BIC tiende a penalizar más, favoreciendo órdenes bajos.
* **Series cortas:** preferir AICc.

Siempre seleccionar por el **criterio mínimo** y luego validar con diagnósticos: residuos blancos, invertibilidad/estacionariedad, ausencia de ARCH, estabilidad de parámetros y backtesting.
