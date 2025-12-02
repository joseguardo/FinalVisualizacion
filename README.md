# Examen Final Visualización 
Analiza el problema planteado en el notebook DAVD_Examen_final_2025_2026.ipynb. Hace referencia al csv meidcal_insurance.csv que tendrás que leer. Procede a entender el planteamiento del problema. Después procede a volver a leer el README.md. Este contiene las visualizaciones que vamos a desarrollar para poder sacar el insight. No quiero que generes ninguna documentación dado que eso lo quiero hacer yo. Tu tarea es encargarte del código, de las visualizaciones y del modelo que haya que entrenar en base a mis indicaciones. Genera la solución en un Notebook aparte que se llama solucion_visualizacion.ipynb que tendrás que generar. 
No quiero que utilices librerías que no estén presentes en Librerias.md a menos que sea estrictamente necesario. 
Procede a hacer un código robusto y minimalista, no te excedas en documenationes. 





# Planificación del Examen

Objetivo principal: Cómo determinar un perfil de alto riesgo que esté pagando un coste menor de póliza. 


## Procedimiento
1. Quiero ver el perfil sociodemográfico de la muestra

2. Quiero ver la distribución de enfermedades dentro de la muestra. 

3. Quiero ver el promedio de lifstyles por cada plan de plan_type

4. Quiero ver primero que gasto e ingreso medio real tiene la aseguradora  cada tipo de cliente en función del plan al que está sujeto: 
Esto implica contrastar total_claims_paid contra la suma de prima anual pagada por el paciente y prima mensual pagada por el cliente agrupados por el tipo de plan (plan_type). 

4. Quiero ver el perfil socio demográfico por cada tipo de seguro. 

5. Quiero ver el perfil sociodemográfico promedio por cada network_tier

6. Quiero ver el número de reclamaciones realizadas por cada network_tier

## Dashboard
1. Después, quiero ver con respecto a cada plan_type cuáles son las enfermedades más presentes dentro de cada plan. Esto me debería dar un insight de cuáles son los perfiles de riesgo más elevados que tenemos. 

2. Después quiero ver el perfil de Lifestyle en función del tipo de enfermedad. 

3. Quiero ver si existe alguna relación entre clientes que hayan realizado cambios a la póliza en 2 años y el total pagado por la asegurados. 

4. Añade tú uno que sean los coeficientes. 


## Modelo

Quiero que entrenes un modelo que sea capaz de predecir el factor de riesgo en función de todas las variables numericas dentro de Demographics y LifeStyle & Habits

















