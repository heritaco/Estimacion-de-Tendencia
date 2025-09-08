**El ciclo HP debe tener media cero y revertir a cero, se entrena con n**

Porque el **ciclo HP** es, por construcción, una serie **de media cero y sin deriva determinista**.
En el HP: $y_t=\tau_t+c_t$ y la penalización $\sum(\Delta^2\tau_t)^2$ tiene núcleo $\{\text{constante}+ \text{tendencia lineal}\}$. Las CPO implican que $c_t$ es **ortogonal** a ese núcleo, i.e. $\sum c_t\approx0$ y $\sum t\,c_t\approx0$. Por eso el componente cíclico debe **revertir a 0**.

Si incluyes $\text{trend='c' o 't'}$ en el ARMA, el modelo “roba” parte de la tendencia y rompe la separación tendencia–ciclo, además de sesgar el pronóstico del ciclo lejos de cero.

Evidencia empírica en tu salida: la constante no es significativa. Coherente con $\text{trend='n'}$.

Qué hacer si tu ciclo no parece centrado:

* Demean: $c_t \leftarrow c_t - \bar c$.
* Revisa $\lambda$ y bordes; el sesgo de extremos puede desplazar la media.
* Sólo si $\bar c$ es estadísticamente diferente de 0, usa `trend='c'` como parche, pero documenta que ya no es un “ciclo HP puro”.
Significado de `trend` en `statsmodels` (ARIMA/SARIMAX):

- `trend='n'` → **sin término determinista**. Media cero en $d=0$. En un ciclo HP debe usarse esto.  
- `trend='c'` → **constante** $\mu$. En ARMA ($d=0$) es una media no nula. En ARIMA ($d\ge1$) la constante induce **deriva**: una tendencia polinómica de grado $d$ en niveles.  
- `trend='t'` → **tendencia lineal sin intercepto**: coeficiente $\delta\cdot t$.  
- `trend='ct'` → **intercepto + tendencia lineal**: $\mu+\delta t$.  
- “**polinomial**” → en ARIMA/SARIMAX no se pasa como cadena; se construye con **regresores exógenos** $1,t,t^2,\dots$ (o con `DeterministicProcess`). En OLS existe además `ctt` (constante, lineal y cuadrática), pero ARIMA/SARIMAX estándar acepta solo `n`,`c`,`t`,`ct`.

**Efectos formales**

Con ARIMA $(p,d,q)$ :

$$
\Phi(L)(1-L)^d y_t=\mu\cdot \mathbf 1\{trend\ \text{incluye }c\}+\Theta(L)\varepsilon_t.
$$

- Si $d=0$ y `c`: $E[y_t]=\dfrac{\mu}{1-\sum_{i=1}^p\phi_i}$.  
- Si $d=1$ y `c`: $y_t$ tiene **pendiente** $\kappa=\dfrac{\mu}{1-\sum\phi_i}$ (drift).  
- Si $d=2$ y `c`: $y_t$ tiene tendencia **cuadrática**; en general, el grado del polinomio determinista es $d$.

Con `t` o `ct`, se añaden términos $\delta t$ (y eventualmente $\mu$) al lado derecho. En $d>0$ esto se acumula en niveles como polinomios de grado $d+1$ o $d$.

### Cuándo usar cada uno
- **Ciclo HP o series ya centradas**: `n`.  
- **ARMA en rendimientos o log-diferencias**: normalmente `n` o, si hay sesgo, `c`.  
- **Niveles con paseo aleatorio ($d=1$)**: `c` para permitir drift. Evita simultanear `ct` con diferenciar $d=1$ si ese crecimiento ya es estocástico.  
- **Tendencias deterministas claras** en niveles estacionarios ($d=0$): `t` o `ct`.  
- **Polinomios de mayor grado**: usa exógenas; no se indican con el string `trend`.

### Polinomio en práctica (ARIMA/SARIMAX)
```python
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.arima.model import ARIMA

idx = y.index  # DatetimeIndex regular
dp = DeterministicProcess(index=idx, constant=True, order=2)  # 1, t, t^2
X = dp.in_sample()
res = ARIMA(y, order=(p,d,q), exog=X).fit()

Xf = dp.out_of_sample(steps=h)   # para pronóstico
fc = res.get_forecast(steps=h, exog=Xf)
```

### Advertencias técnicas
- Un término determinista puede **competir** con la parte integrada: decide si la tendencia es estocástica (diferenciación) o determinista (trend), evitando redundancia.  
- En ciclos, añadir `c` o `t` **contamina** la separación tendencia–ciclo.  
- La inclusión de polinomios altos aumenta colinealidad y varianza de estimación. Usa criterios de información y pruebas de diagnóstico.
