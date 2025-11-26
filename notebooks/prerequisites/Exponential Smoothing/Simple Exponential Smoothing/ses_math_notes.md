# Suavizamiento Exponencial Simple (SES) con notación matricial
Notebook didáctico. Solo NumPy y matplotlib. Se muestran PNG y se formaliza el filtro en álgebra lineal.


---

## 1) Desagregación de la serie
Supondremos que la observación se descompone como

$$ y_t = s_t + \varepsilon_t, \quad t=0,1,\dots,n-1, $$
donde $s_t$ es el **nivel** (tendencia local) y $\varepsilon_t$ es el residuo de uno a paso.


---

## 2) Modelo SES y expansión cerrada
Recursión de nivel:

$$ L_0 = y_0, \qquad L_t = \alpha y_t + (1-\alpha) L_{t-1},\quad 0<\alpha\le 1. $$
Fitted one-step: $\hat y_t=L_{t-1}$ para $t\ge 1$. Expansión explícita por sustitución:
$$ L_t = (1-\alpha)^t y_0 + \alpha\sum_{j=1}^{t} (1-\alpha)^{t-j} y_j. $$
Esto muestra que SES es un **promedio móvil exponencial** con pesos decrecientes geométricamente.


---

## 3) Notación matricial del filtro exponencial
Sea $Y=[y_1,\dots,y_n]'$ y $S=[L_1,\dots,L_n]'$. Existe una matriz estrictamente triangular inferior $W(\alpha)\in\mathbb{R}^{n\times n}$ tal que $S=W(\alpha)\,Y$, con
$$ W_{t,t'}(\alpha)= \begin{cases}
(1-\alpha)^{t-1}, & t'=1, \\
\alpha(1-\alpha)^{t-t'}, & 2\le t'\le t, \\
0, & t'<t.
\end{cases} $$
Ejemplo de estructura ($n=6$):
$$
W(\alpha)=\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0\\
(1-\alpha) & \alpha & 0 & 0 & 0 & 0\\
(1-\alpha)^2 & \alpha(1-\alpha) & \alpha & 0 & 0 & 0\\
(1-\alpha)^3 & \alpha(1-\alpha)^2 & \alpha(1-\alpha) & \alpha & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \ddots & 0\\
(1-\alpha)^5 & \alpha(1-\alpha)^4 & \alpha(1-\alpha)^3 & \alpha(1-\alpha)^2 & \alpha(1-\alpha) & \alpha
\end{bmatrix}.$$
Así, el filtro SES es lineal y **causal**: $L_t$ depende solo de $\{y_j\}_{j\le t}$.


---

## 4) Elección de $\alpha$ y vida media efectiva
Elegimos $\alpha$ minimizando el MSE in-sample de los one-step errors $e_t=y_t-\hat y_t$ con $t\ge1$:
$$ \operatorname{MSE}(\alpha)=\frac{1}{n-1}\sum_{t=1}^{n-1} e_t(\alpha)^2. $$
La **vida media** de los pesos exponenciales satisface $(1-\alpha)^{h_{1/2}}=\tfrac12$, por lo que
$$ h_{1/2} = \frac{\ln(1/2)}{\ln(1-\alpha)}. $$
Pequeño $\alpha$ $\Rightarrow$ larga memoria; grande $\alpha$ $\Rightarrow$ reacción rápida.


---

## 5) Carga de datos y ajuste
Intentamos leer `cd/../data/air_passengers.csv` con cabeceras `Month,#Passengers`. Si no existe, usamos una muestra corta.


---

## 6) Implementación del SES y selección de $\alpha$
Aplicamos la recursión y hacemos una búsqueda en malla para $\alpha\in(0,1]\.$


---

## 7) Pronósticos
Para SES, el pronóstico $h$-pasos adelante después del último punto es constante:
$$ \hat y_{n+h\,|\,n-1} = L_{n-1}, \quad h\ge 1. $$


---

## 8) Visualización (PNG con matplotlib)
La curva discontinua es el ajuste de un paso $\hat y_t=L_{t-1}$.

---

## 9) Construcción explícita de $W(\alpha)$ y ejemplo pequeño
Para $T=5$ y $\alpha=0.4$, la relación $S=WY$ produce las siguientes combinaciones lineales:
$$\begin{aligned}
 s_1 &= & &1.00y_1\\
 s_2 &= & &0.60y_1 &+&0.40y_2\\
 s_3 &= & &0.36y_1 &+&0.24y_2 &+&0.40y_3\\
 s_4 &= & &0.22y_1 &+&0.14y_2 &+&0.24y_3 &+&0.40y_4\\
 s_5 &= & &0.13y_1 &+&0.09y_2 &+&0.14y_3 &+&0.24y_4 &+&0.40y_5\end{aligned}$$
Observe la naturaleza de **promedio ponderado** y que los pesos decrecen geométricamente sobre el pasado.


---

## 10) Apéndice: contraste con el filtro de Hodrick–Prescott
HP busca $S$ como la solución de
$$ \min_S\ \|Y-S\|^2 + \lambda\,\|AS\|^2, $$
con $A$ la matriz de segundas diferencias (dimensión $(T-2)\times T$). Las CPO implican
$$ S^{HP} = (I + \lambda A' A)^{-1} Y, \qquad C^{HP}=Y-S^{HP}. $$
**Diferencia clave:** SES es un filtro recursivo causal con parámetro $\alpha$; HP es un estimador global que resuelve un sistema lineal con parámetro de suavidad $\lambda$.


---

### Notas finales
- Inicialización usada: $L_0=y_0$. Otras opciones existen.
- La malla de $\alpha$ puede refinarse o sustituirse por optimización continua.
- SES no modela tendencia ni estacionalidad explícitas; para ello use Holt o Holt–Winters.