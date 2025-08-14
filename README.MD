en proceso!


:: receta mínima (reproducible)
conda env export --from-history -n guerrero-gpu > environment.yml

:: lockfile exacto (clon 1:1)
conda env export -n guerrero-gpu > environment-lock.yml
