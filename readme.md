# Análisis Exploratorio de Datos: Ideación Suicida en Jóvenes (Datos Simulados)

**Autor:** Reiner Fuentes Ferrada  
**Fecha:** Octubre 2025  
**Dataset:** `suicidalidad_jovenes_anonimizado.csv` (N = 1,029 observaciones simuladas)

> ⚠️ **ADVERTENCIA IMPORTANTE**:  
> **Este dataset NO contiene datos reales de personas**. Ha sido generado mediante técnicas de anonimización y permutación aleatoria a partir de una fuente original (con autorización y bajo protocolo ético).  
> **Ningún hallazgo aquí debe interpretarse como evidencia clínica, epidemiológica o psicológica real**.  
> El propósito de este proyecto es **exclusivamente metodológico, educativo y de demostración técnica**.

---

## Contexto del Proyecto

Este repositorio presenta un **ejercicio académico de Análisis Exploratorio de Datos (EDA)** aplicado a un conjunto de datos **simulados** sobre factores asociados a ideación suicida en población joven. El objetivo es **demostrar buenas prácticas en manejo, análisis y visualización de datos sensibles**, respetando principios de privacidad y ética en ciencia de datos.

### Características del Dataset Simulado

- **N total:** 1,029 observaciones
- **Edad:** Rango 18–36 años (distribución permutada)
- **Variables:** 42 columnas, incluyendo:
  - Sociodemográficas simuladas (edad, género, orientación sexual)
  - Sintomatología clínica (depresión, ansiedad, ideación suicida)
  - Escalas psicométricas sintéticas (AAQ-II, SWB-7)
  → **No existe relación real entre las variables en cada fila**.

---

## Objetivos Metodológicos (no clínicos)

Este análisis busca ilustrar:

1. **Preparación ética de datos sensibles**: eliminación de identificadores, permutación controlada.
2. **Estructuración de un EDA completo**: desde limpieza hasta visualización avanzada.
3. **Aplicación de técnicas de clustering y correlación** en contextos clínicos simulados.
4. **Comunicación responsable** de hallazgos en salud mental.

> ❗ **No se plantean ni prueban hipótesis científicas reales**, ya que los datos carecen de coherencia clínica interna.

---

## Metodología (Enfoque Técnico)

### Análisis Realizados (Demostrativos)
- Estadística descriptiva univariada (frecuencias, distribuciones)
- Visualización de patrones *artificiales* (correlaciones espurias)
- Comparación de métodos de clustering (Gaussian Mixture Model, K-Means, Jerárquico, Spectral)
- Generación de perfiles latentes *simulados*

> 🔍 **Nota**: Los "perfiles" y "correlaciones" observados **son artefactos del formato de los datos originales**, no reflejan fenómenos psicológicos reales.

---

## Hallazgos: Interpretación Responsable

Los resultados presentados (ej.: "mayor inflexibilidad en ideación suicida") **son producto del azar**, no evidencia empírica.  
Se incluyen únicamente para:

- Demostrar un **flujo de trabajo analítico completo**
- Ilustrar cómo **no deben interpretarse datos desvinculados**
- Enfatizar la **importancia de la integridad de los datos** en investigación clínica

---

## Estructura del Proyecto

- `README.md` — Documentación principal
- `suicidalidad_jovenes_anonimizado.csv` — Dataset simulado y anonimizado
- `codigo_eda_completo.py` — Script de análisis completo
- `figuras/` — Carpeta con todos los outputs:
  - `diccionario_variables.csv`
  - `01_distribucion_variables_categorias.png`
  - `02_composicion_dataset_pie.png`
  - `02_resumen_clasificacion_variables.csv`
  - `03_porcentaje_missing_por_variable.png`
  - `03_analisis_valores_perdidos.csv`
  - `03_analisis_outliers.csv`
  - `03_reporte_calidad_datos.csv`
  - `04_boxplots_variables_continuas.png`
  - `04_heatmap_completitud.png`
  - `05_distribucion_edad.png`
  - `06_variables_sociodemograficas.png`
  - `07_prevalencia_condiciones_clinicas.png`
  - `08_distribuciones_severidad.png`
  - `09_variables_riesgo_suicida.png`
  - `10_comparacion_grupos_ideacion.png`
  - `11_escalas_psicometricas.png`
  - `12_matriz_correlaciones_profesional.png`
  - `12_matriz_correlaciones.csv`
  - `12_correlaciones_ordenadas.csv`
  - `13_comparacion_metodos_clustering.png`
  - `13_comparacion_clustering.csv`
  - `14_perfiles_caracterizacion.png`
  - `15_mapa_3d_todos_ideadores.html`
  - `16_comparacion_perfiles_radar.html`
  - `17_comparacion_perfiles_radar_matplotlib.png`
  - `17_resumen_perfiles_radar.csv`


> Este repositorio **es seguro para compartir públicamente**, ya que **no contiene datos reales**.

---

## Tecnologías Utilizadas

**Lenguaje:** Python 3.8+  
**Librerías:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `plotly`  
**Técnicas demostradas:** EDA, clustering, visualización interactiva, gestión de datos sensibles

---

## Consideraciones Éticas y de Uso

- **Este dataset NO debe usarse** para:
  - Publicaciones científicas
  - Toma de decisiones clínicas
  - Diseño de políticas públicas
  - Cualquier inferencia sobre poblaciones reales
- **Sí puede usarse** para:
  - Enseñanza de EDA y ciencia de datos
  - Pruebas de código y visualización
  - Discusión sobre ética en manejo de datos sensibles

> 💡 **Recomendación**: Si trabajas con datos reales sobre salud mental, siempre consulta con un comité de ética y aplica protocolos de anonimización robustos (ej.: k-anonimidad, datos sintéticos con SDV).

---

## Recursos de Apoyo en Salud Mental

Si tú o alguien que conoces está atravesando una crisis emocional o tiene pensamientos suicidas, **busca ayuda inmediata**:

- **Chile**:  
  - **Teléfono de la Esperanza**: 562 2757 7777  
  - **Fono Salud Mental**: 600 360 7777  
  
- **Internacional**:  
  - [International Association for Suicide Prevention (IASP)](https://www.iasp.info/resources/Crisis_Centres/)

---

## Contacto

**Reiner Fuentes Ferrada**  
reinerfuentes7@gmail.com  

> Este trabajo forma parte de una reflexión académica sobre la intersección entre **ciencia de datos, ética y salud mental**, con enfoque en buenas prácticas para el manejo responsable de información sensible.

---

**Última actualización:** Octubre 2025  
**Licencia:** Uso educativo y no comercial. Atribución requerida.
