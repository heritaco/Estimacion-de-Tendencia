# Holt Linear (Double Exponential Smoothing) — notas matemáticas

## Descomposición

$$ y_t = L_t + \varepsilon_t, \quad t=0,1,\dots,n-1. $$

## Modelo

$$ L_0=y_0,\qquad B_0=y_1-y_0. $$

$$ L_t = \alpha y_t + (1-\alpha)(L_{t-1}+B_{t-1}). $$

$$ B_t = \beta(L_t - L_{t-1}) + (1-\beta) B_{t-1}. $$

One-step: $$ \hat y_t = L_{t-1} + B_{t-1},\ t\ge1. $$

h-step: $$ \hat y_{n+h\,|\,n} = L_n + h B_n. $$


## Notación matricial

Defina $X_t=[L_t,B_t]'$ y matrices

$$ A=\begin{bmatrix}1-\alpha & 1-\alpha\\ -\beta\alpha & 1-\beta\alpha\end{bmatrix},\quad K=\begin{bmatrix}\alpha\\ \beta\alpha\end{bmatrix}. $$

Entonces $$ X_t = A X_{t-1} + K y_t,\quad \hat y_t = [1\ \ 1] X_{t-1}. $$


## Elección de $(\alpha,\beta)$

Minimizar $$ \operatorname{MSE}(\alpha,\beta)=\frac{1}{n-1}\sum_{t=1}^{n-1}(y_t-\hat y_t)^2. $$


## Inicialización

$$ L_0=y_0,\quad B_0=y_1-y_0\ (n\ge2). $$ Otras variantes son posibles.
