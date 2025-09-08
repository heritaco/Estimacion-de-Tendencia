# *Con $m{=}\mu{=}0$ y $d{=}1$ es un caso de Exponential Smoothing*

Porque con $m{=}\mu{=}0$ y $d{=}1$ el problema

$$
\min_{\{\tau_t\}}\;\sum_{t=1}^N (Z_t-\tau_t)^2+\lambda\sum_{t=2}^N(\tau_t-\tau_{t-1})^2
$$

genera ecuaciones normales tridiagonales

$$
\begin{aligned}
(1{+}\lambda)\tau_1-\lambda\tau_2&=Z_1,\\
-\lambda\tau_{t-1}+(1{+}2\lambda)\tau_t-\lambda\tau_{t+1}&=Z_t\quad(2\le t\le N{-}1),\\
-\lambda\tau_{N-1}+(1{+}\lambda)\tau_N&=Z_N,
\end{aligned}
$$

cuyo **kernel de Green** es geométrico. En el límite infinito,

$$
\widehat{\tau}_t=\sum_{k=-\infty}^{\infty}w_k\,Z_{t-k},\qquad 
w_k=\frac{1-\rho}{1+\rho}\,\rho^{|k|},
$$

con

$$
\rho=\frac{(1+2\lambda)-\sqrt{1+4\lambda}}{2\lambda}\in(0,1).
$$

Ese núcleo exponencial simétrico es exactamente el **exponential smoothing** de King–Rebelo al que el paper se refiere para $d{=}1$ y $m{=}0$. &#x20;

Equivalente recursivo: factoriza el filtro como alisado exponencial hacia delante y hacia atrás con parámetro $\alpha=1-\rho$:

$$
\begin{aligned}
u_t&=\alpha Z_t+(1-\alpha)u_{t-1},\\
\widehat{\tau}_t&=\alpha u_t+(1-\alpha)\widehat{\tau}_{t+1},
\end{aligned}
$$

que reproduce los pesos $w_k$ anteriores tras normalización. El artículo enmarca este caso dentro de su forma matricial general $\hat{\tau}=(I+\lambda K_1'K_1)^{-1}Z$.&#x20;