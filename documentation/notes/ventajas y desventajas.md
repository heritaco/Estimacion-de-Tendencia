Fuente de datos: <https://fred.stlouisfed.org/series/GDPC1/>

{{ termina_ejemplo }}





## El filtro HP tiene malas propiedades estadísticas
\cite{Hamilton:2017}: Why You Should Never Use the HP Filter?

1. El filtro Hodrick-Prescott introduce relaciones dinámicas espurias que no tienen sustento en el proceso generador de datos subyacente.
2. Los valores filtrados al final de la muestra son muy distintos de los del medio, y también están caracterizados por una dinámica espuria.
3. Una formalización estadística del problema típicamente produce valores de $\lambda$ que distan mucho de los usados comúnmente en la práctica.
4. Para Hamilton, hay una alternativa mejor: una regresión AR(4) alcanza todos los objetivos buscados por usuarios del filtro HP pero con ninguno de sus desventajas.



{{ empieza_ejemplo }} Filtrando el PIB de Costa Rica con HP {{ fin_titulo_ejemplo }}
Los datos filtrados son muy sensibles a nueva información.

```{figure} figures/pib-hp-tails.png

Ciclo del PIB de Costa Rica, conforme se van agregando nuevas observaciones
```
{{ termina_ejemplo }}



## HP puede inducir conclusiones equivocadas acerca del comovimiento de series
\textcite{CogleyNason:1995} analizaron las propiedades espectrales del filtro HP

Cuando se mide el componente cíclico de una serie de tiempo, ¿es buena idea usar el filtro HP?

Depende de la serie original

- **Sí**, si es estacionaria alrededor de tendencia
- **No**, si es estacionaria en diferencia

Este resultado tiene implicaciones importantes para modelos DSGE: Cuando se aplica el filtro HP a una serie integrada, el filtro introduce periodicidad y comovimiento en las frecuencias del ciclo económico, **aún si no estaban presentes en los datos originales**.