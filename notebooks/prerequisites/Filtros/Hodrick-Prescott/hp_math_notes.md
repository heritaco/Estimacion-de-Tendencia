# Filtro de Hodrick–Prescott — notas matemáticas

## Desagregación

$$ y_t = s_t + c_t, \qquad t=1,\dots,T. $$

## Problema de optimización

$$ S^{HP} = \arg\min_S\ \sum_{t=1}^T (y_t-s_t)^2 + \lambda\sum_{t=2}^{T-1}(s_{t+1}-2s_t+s_{t-1})^2. $$

## Forma matricial y solución cerrada

Defina $Y=[y_1,\dots,y_T]'$, $S=[s_1,\dots,s_T]'$ y $A\in\mathbb{R}^{(T-2)\times T}$ con segundas diferencias. Entonces

$$ S^{HP} = (I + \lambda A'A)^{-1}Y, \qquad C^{HP} = Y - S^{HP}. $$

## KKT / CPO

Tomando derivadas: $-2Y + 2(I+\lambda A'A)S=0 \Rightarrow (I+\lambda A'A)S=Y$.

## Elección de $\lambda$

Anual $\approx 100$, trimestral $\approx 1600$, mensual $\approx 14400$. En muestras pequeñas reduzca $\lambda$.

## Observaciones

- El operador $A'A$ es tridiagonal ampliada (pentadiagonal efectiva) y SPD; se puede resolver en tiempo lineal con algoritmos de banda.
