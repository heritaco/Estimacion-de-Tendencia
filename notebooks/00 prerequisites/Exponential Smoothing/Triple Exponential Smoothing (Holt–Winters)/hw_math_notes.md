# Holt–Winters — notas matemáticas

## Formas del modelo

**Aditiva**

$$ L_t = \alpha (y_t - S_{t-m}) + (1-\alpha)(L_{t-1}+B_{t-1}), \quad B_t = \beta (L_t - L_{t-1}) + (1-\beta) B_{t-1}, $$

$$ S_t = \gamma (y_t - L_t) + (1-\gamma) S_{t-m}. $$

One-step: $$ \hat y_t = L_{t-1} + B_{t-1} + S_{t-m}. $$

h-step: $$ \hat y_{t+h} = L_t + h B_t + S_{t+h-m\,\bmod\, m}. $$


**Multiplicativa**

$$ L_t = \alpha \frac{y_t}{S_{t-m}} + (1-\alpha)(L_{t-1}+B_{t-1}), \quad B_t = \beta (L_t - L_{t-1}) + (1-\beta) B_{t-1}, $$

$$ S_t = \gamma \frac{y_t}{L_t} + (1-\gamma) S_{t-m}. $$

One-step: $$ \hat y_t = (L_{t-1}+B_{t-1}) S_{t-m}. $$

h-step: $$ \hat y_{t+h} = (L_t + h B_t)\, S_{t+h-m\,\bmod\, m}. $$


## Inicialización (regla práctica)

Promedios por temporada y promediar sobre temporadas:

$$ S_j^{(0)} = \begin{cases} \frac{1}{k}\sum_{i=0}^{k-1}(y_{im+j}-\bar y_i), &\text{aditivo}\\[4pt]

\frac{1}{k}\sum_{i=0}^{k-1}\frac{y_{im+j}}{\bar y_i}, &\text{multiplicativo}\end{cases} $$

con $\bar y_i=\frac{1}{m}\sum_{j=0}^{m-1} y_{im+j}$ y $k=\lfloor n/m\rfloor$.

Luego $L_0\approx \frac{1}{m}\sum_{j=0}^{m-1}(y_j - S_j^{(0)})$ (aditivo) o $\frac{1}{m}\sum (y_j/S_j^{(0)})$ (multiplicativo); 

$B_0\approx \frac{1}{m^2}\sum_{j=0}^{m-1}\big[(y_{m+j}^{(*)})-(y_{j}^{(*)})\big]$ con $(*)$ estacionalmente ajustado.


## Estado (boceto)

Defina $X_t=[L_t, B_t, S_{t-m+1},\dots,S_t]'$. El avance estacional es un **shift** circular del bloque $S$.

El sistema es afín $X_t = A(\alpha,\beta,\gamma) X_{t-1} + K(\alpha,\beta,\gamma) y_t$ con $A$ dependiente del tipo aditivo/multiplicativo.
