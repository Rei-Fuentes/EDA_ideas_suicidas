# Análisis Exploratorio de Datos: Ideación Suicida en Jóvenes

**Autor:** Reiner Fuentes Ferrada  
**Fecha:** Octubre 2025  
**Dataset:** `suicidalidad_jovenes.csv` (N = 1,029 jóvenes)

---

## Contexto del Estudio

Este proyecto presenta un análisis exploratorio exhaustivo de datos sobre ideación suicida en población joven universitaria. El estudio examina las relaciones entre variables psicológicas, sintomatología clínica y factores de riesgo suicida con el objetivo de identificar patrones que informen estrategias de prevención e intervención.

### Muestra

- **N total:** 1,029 estudiantes universitarios
- **Edad promedio:** 19-20 años
- **Distribución por sexo:** ~29% masculino, ~71% femenino
- **Prevalencia de ideación suicida (último mes):** ~56%

---

## Hipótesis de Investigación

### **H1: Inflexibilidad Psicológica e Ideación Suicida**
La inflexibilidad psicológica (medida con AAQ-II) estará positivamente relacionada con la ideación suicida.

### **H2: Bienestar Psicológico e Ideación Suicida**
El bienestar psicológico (medido con SWB-7) estará inversamente relacionado con la ideación suicida.

### **H3: Sintomatología Clínica e Ideación Suicida**
Los síntomas de ansiedad y depresión se asociarán con mayor nivel de ideación suicida.

---

## Metodología

### Variables Analizadas (42 variables)

**Sociodemográficas:** Edad, sexo, identidad de género, orientación sexual, nivel educativo parental

**Clínicas:**
- Episodio depresivo mayor (dicotómico y severidad)
- Ansiedad generalizada (dicotómico y severidad)
- Duración temporal de sintomatología

**Riesgo Suicida:**
- Ideación suicida pasiva y activa (lifetime)
- Ideación suicida último mes (variable dependiente principal)
- Severidad de ideación (0-4)
- Conductas autolesivas no suicidas

**Escalas Psicométricas:**
- **AAQ-II** (7 ítems): Inflexibilidad psicológica / evitación experiencial
- **SWB-7** (7 ítems): Bienestar psicológico subjetivo

---

## Análisis Realizados

### **Fase 1: Preparación y Documentación**
- Creación de diccionario completo de variables
- Clasificación por categorías (sociodemográficas, clínicas, escalas)
- Identificación de variables clave para hipótesis

### **Fase 2: Inspección Inicial**
- Análisis de dimensionalidad (1,029 × 42)
- Clasificación de tipos de datos
- Distribución de variables por categoría

### **Fase 3: Limpieza y Validación**
- Análisis de valores perdidos (completitud >97%)
- Detección de outliers (método IQR)
- Verificación de duplicados
- Evaluación de calidad de datos

### **Fase 4: Análisis Descriptivo Univariado**
- Perfil sociodemográfico de la muestra
- Prevalencias de condiciones clínicas
  - Episodio depresivo: ~55%
  - Ansiedad generalizada: ~29%
  - Ideación suicida (último mes): ~56%
- Distribuciones de severidad (depresión, ansiedad, ideación)
- Comparación de grupos CON vs SIN ideación suicida
  - Diferencias significativas en depresión, ansiedad, inflexibilidad y bienestar (p < .001)
- Análisis de escalas psicométricas (AAQ-II y SWB-7)

### **Fase 5: Análisis de Correlaciones y Perfiles Latentes**

**Matriz de Correlaciones:**
- Análisis de relaciones bivariadas entre variables clave
- Identificación de correlaciones fuertes entre:
  - Depresión ↔ Ansiedad
  - Inflexibilidad ↔ Ideación
  - Bienestar ↔ Depresión (negativa)

**Análisis de Clustering:**
- Comparación de 5 métodos de clustering profesionales:
  - Gaussian Mixture Model (GMM)
  - K-Means
  - Clustering Jerárquico (Ward)
  - Spectral Clustering
- Evaluación con 3 métricas: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- Selección basada en análisis multi-criterio
- Identificación de perfiles latentes en estudiantes con ideación
- Caracterización psicológica de cada perfil

### **Fase 6: Visualizaciones Avanzadas**
- **Mapa 3D interactivo:** Visualización de todos los estudiantes con ideación en espacio depresión-bienestar-inflexibilidad
- **Gráfico Radar:** Comparación multidimensional de perfiles latentes
- Análisis de dispersión y heterogeneidad de la muestra

---

## Hallazgos Principales

### Soporte a las Hipótesis

✓ **H1 Confirmada:** Los estudiantes con ideación suicida presentan significativamente mayor inflexibilidad psicológica (Δ = +5.9 puntos, p < .001)

✓ **H2 Confirmada:** Los estudiantes con ideación suicida presentan significativamente menor bienestar psicológico (Δ = -3.4 puntos, p < .001)

✓ **H3 Confirmada:** Los estudiantes con ideación suicida presentan mayor severidad de depresión (Δ = +3.7 puntos, p < .001) y ansiedad (Δ = +2.2 puntos, p < .001)

### Perfiles Latentes

Se identificaron perfiles diferenciados de riesgo en estudiantes con ideación suicida, caracterizados por distintas configuraciones de:
- Sintomatología depresiva y ansiosa
- Niveles de inflexibilidad psicológica
- Recursos de bienestar psicológico
- Severidad de ideación

### Implicaciones Clínicas

1. **Evaluación multidimensional:** La ideación suicida no es función de una sola variable, sino de la interacción compleja entre factores clínicos y psicológicos

2. **Intervenciones diferenciadas:** Los perfiles latentes sugieren la necesidad de abordajes terapéuticos adaptados a configuraciones específicas de vulnerabilidad

3. **Targets terapéuticos:** La inflexibilidad psicológica y el bienestar emergen como blancos relevantes de intervención, además del tratamiento sintomático de depresión/ansiedad

---

## Estructura del Proyecto

```
.
├── README.md                                    # Este archivo
├── suicidalidad_jovenes.csv                     # Dataset original (no incluido)
├── codigo_eda_completo.py                       # Código completo del análisis
│
└── figuras/                                     # Outputs generados
    │
    ├── diccionario_variables.csv                # Fase 1
    │
    ├── 01_distribucion_variables_categorias.png # Fase 2
    ├── 02_composicion_dataset_pie.png
    ├── 02_resumen_clasificacion_variables.csv
    │
    ├── 03_porcentaje_missing_por_variable.png   # Fase 3
    ├── 03_analisis_valores_perdidos.csv
    ├── 03_analisis_outliers.csv
    ├── 03_reporte_calidad_datos.csv
    ├── 04_boxplots_variables_continuas.png
    ├── 04_heatmap_completitud.png
    │
    ├── 05_distribucion_edad.png                 # Fase 4
    ├── 06_variables_sociodemograficas.png
    ├── 07_prevalencia_condiciones_clinicas.png
    ├── 08_distribuciones_severidad.png
    ├── 09_variables_riesgo_suicida.png
    ├── 10_comparacion_grupos_ideacion.png
    ├── 11_escalas_psicometricas.png
    │
    ├── 12_matriz_correlaciones_profesional.png  # Fase 5
    ├── 12_matriz_correlaciones.csv
    ├── 12_correlaciones_ordenadas.csv
    ├── 13_comparacion_metodos_clustering.png
    ├── 13_comparacion_clustering.csv
    ├── 14_perfiles_caracterizacion.png
    │
    ├── 15_mapa_3d_todos_ideadores.html          # Fase 6
    ├── 16_comparacion_perfiles_radar.html
    ├── 17_comparacion_perfiles_radar_matplotlib.png
    └── 17_resumen_perfiles_radar.csv
```

---

## Tecnologías Utilizadas

**Lenguaje:** Python 3.8+

**Librerías principales:**
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `matplotlib` - Visualizaciones estáticas
- `seaborn` - Gráficos estadísticos
- `scipy` - Pruebas estadísticas
- `scikit-learn` - Clustering y métricas
- `plotly` - Visualizaciones interactivas 3D/radar

**Técnicas estadísticas:**
- Estadística descriptiva univariada
- Correlaciones de Pearson
- Pruebas t de Student
- ANOVA y Kruskal-Wallis
- Clustering: GMM, K-Means, Jerárquico, Spectral
- Métricas de validación: Silhouette, Calinski-Harabasz, Davies-Bouldin

---

## Diseño Visual

Todas las visualizaciones siguen un diseño profesional consistente:
- **Fondo negro** para presentaciones profesionales
- **Paleta de colores Tailwind personalizada:**
  - Tea Green (#c5ebc3)
  - Ash Gray (#b7c8b5)
  - Rose Quartz (#a790a5)
  - Chinese Violet (#875c74)
  - Eggplant (#54414e)
  - Charcoal (#2f4858)
- **Alta resolución** (300 DPI) para publicaciones
- **Tipografía clara** y legible

---

## Resultados en Números

- **17-19 figuras** de alta calidad generadas
- **8 tablas CSV** con análisis detallados
- **2 visualizaciones interactivas** (HTML)
- **42 variables** analizadas exhaustivamente
- **1,029 participantes** caracterizados
- **5 métodos de clustering** comparados
- **3 hipótesis** evaluadas y confirmadas

---

## Reproducibilidad

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly kaleido
```

### Ejecución

```python
python codigo_eda_completo.py
```

El código:
- ✅ Crea automáticamente la estructura de carpetas
- ✅ Genera todas las figuras y tablas
- ✅ Maneja errores y librerías faltantes
- ✅ Proporciona feedback detallado del progreso

---

## Notas Metodológicas

- **Exclusiones:** Variables de crisis de pánico y PTSD fueron excluidas por alejarse del objetivo central del estudio
- **Missing data:** Tratamiento con eliminación por lista (listwise deletion) dado el alto porcentaje de completitud (>97%)
- **Outliers:** Mantenidos en el análisis por ser clínicamente plausibles y representar casos reales de alta sintomatología
- **Clustering:** Selección del mejor método basada en score compuesto de múltiples métricas de validación

---

## Referencias Clave

- **AAQ-II:** Acceptance and Action Questionnaire-II (Bond et al., 2011)
- **SWB-7:** Scale of Positive and Negative Experience (Diener et al., 2010)
- **Clustering validation:** Rousseeuw (1987), Caliński & Harabasz (1974), Davies & Bouldin (1979)

---

## ⚠️ Consideraciones Éticas

Este estudio involucra datos sensibles sobre ideación suicida. Todos los análisis se realizaron con:
- Respeto a la confidencialidad de los participantes
- Enfoque preventivo y de promoción de salud mental
- Objetivo de informar intervenciones basadas en evidencia

**Si tú o alguien que conoces experimenta ideación suicida, busca ayuda profesional inmediatamente.**

---

## 📧 Contacto

**Reiner Fuentes Ferrada**  
**reinerfuentes7@gmail.coma**  


---

**Última actualización:** Octubre 2025
