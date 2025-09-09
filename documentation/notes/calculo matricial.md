Aquí tienes un arsenal de “trucos” estándar de cálculo matricial. Todos suponen vectores columna $x,y\in\mathbb{R}^n$ y matrices $A\in\mathbb{R}^{n\times n}$. Se toman gradientes respecto a $x$, y la convención es que el gradiente es vector columna:

---

### Escalares lineales

$$
\frac{\partial}{\partial x}\,(a^\top x)=a,
\qquad
\frac{\partial}{\partial x}\,(x^\top a)=a.
$$

### Cuadráticos

$$
\frac{\partial}{\partial x}\,(x^\top A x)=(A+A^\top)x.
$$

Si $A$ es simétrica:

$$
\frac{\partial}{\partial x}\,(x^\top A x)=2Ax.
$$

### Productos con matriz fija

$$
\frac{\partial}{\partial x}\,(Ax) = A^\top,
$$

donde el resultado es la jacobiana de $Ax$ respecto a $x$.

### Normas

$$
\frac{\partial}{\partial x}\,\|x\|^2 = \frac{\partial}{\partial x}(x^\top x)=2x.
$$

Más general:

$$
\frac{\partial}{\partial x}\,\|Ax-b\|^2=2A^\top(Ax-b).
$$

### Determinantes y trazas

* $\dfrac{\partial}{\partial A}\,\mathrm{tr}(A^\top X)=X$.
* $\dfrac{\partial}{\partial A}\,\mathrm{tr}(AX)=X^\top$.
* $\dfrac{\partial}{\partial A}\,\log\det A = (A^{-1})^\top$ para $A$ invertible.

### Derivadas mixtas

Si $f(x)=y^\top x$ con $y=y(x)$, entonces:

$$
\nabla_x f = J_y(x)^\top x + y(x),
$$

donde $J_y$ es la jacobiana de $y$.

### Regla útil: **$\mathrm{d}\,\mathrm{tr}(X^\top A X)=\mathrm{tr}((A+A^\top)X^\top \mathrm{d}X)$.**

---

¿Quieres que te prepare una tabla-resumen con estas fórmulas (como “chuleta”) lista para usar en problemas de derivación matricial?
