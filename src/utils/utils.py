"""
utils.py - Funciones Auxiliares para Análisis Exploratorio de Datos
Autor: Reiner Fuentes Ferrada
Fecha: Octubre 2025

Módulo con funciones reutilizables para análisis de datos de salud mental y suicidalidad.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Paleta de colores Tailwind personalizada
COLORES = ["#c5ebc3", "#b7c8b5", "#a790a5", "#875c74", "#54414e", "#2f4858"]

def configurar_estilo_matplotlib(fondo_negro=True):
    """
    Configura el estilo global de matplotlib para visualizaciones.
    
    Parameters:
    -----------
    fondo_negro : bool, default=True
        Si True, configura fondo negro para presentaciones profesionales.
    """
    if fondo_negro:
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['savefig.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
    else:
        plt.style.use('default')
    
    # Configuración general
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


# =============================================================================
# MANEJO DE ARCHIVOS Y DIRECTORIOS
# =============================================================================

def crear_estructura_proyecto(carpetas=None):
    """
    Crea la estructura de carpetas del proyecto.
    
    Parameters:
    -----------
    carpetas : list, optional
        Lista de carpetas a crear. Por defecto: ['data', 'figuras', 'src', 'src/utils']
    
    Returns:
    --------
    dict : Diccionario con rutas creadas
    """
    if carpetas is None:
        carpetas = ['data', 'figuras', 'src', 'src/utils']
    
    rutas = {}
    for carpeta in carpetas:
        Path(carpeta).mkdir(parents=True, exist_ok=True)
        rutas[carpeta] = Path(carpeta)
    
    return rutas


def cargar_dataset(ruta_archivo, encoding='utf-8', sep=','):
    """
    Carga un dataset con manejo de diferentes encodings y separadores.
    
    Parameters:
    -----------
    ruta_archivo : str
        Ruta al archivo CSV
    encoding : str, default='utf-8'
        Encoding preferido
    sep : str, default=','
        Separador preferido
    
    Returns:
    --------
    pd.DataFrame : Dataset cargado
    """
    # Intentar con configuración preferida primero
    try:
        df = pd.read_csv(ruta_archivo, encoding=encoding, sep=sep)
        if df.shape[1] > 1:
            return df
    except:
        pass
    
    # Intentar diferentes combinaciones
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    separadores = [',', ';', '\t', '|']
    
    for enc in encodings:
        for sep_test in separadores:
            try:
                df = pd.read_csv(ruta_archivo, encoding=enc, sep=sep_test)
                if df.shape[1] > 1:
                    print(f"✓ Archivo cargado con encoding={enc}, sep='{sep_test}'")
                    return df
            except:
                continue
    
    raise ValueError("No se pudo cargar el archivo con ninguna combinación de encoding/separador")


# =============================================================================
# ANÁLISIS DE CALIDAD DE DATOS
# =============================================================================

def analizar_valores_perdidos(df, umbral_critico=0.05):
    """
    Analiza valores perdidos en el dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    umbral_critico : float, default=0.05
        Umbral para considerar una variable como problemática (5%)
    
    Returns:
    --------
    pd.DataFrame : Tabla con análisis de valores perdidos
    """
    missing_data = pd.DataFrame({
        'Variable': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Non_Missing_Count': df.notnull().sum()
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    # Clasificar severidad
    missing_data['Severidad'] = pd.cut(
        missing_data['Missing_Percentage'],
        bins=[0, umbral_critico*100, 10, 30, 100],
        labels=['Bajo', 'Moderado', 'Alto', 'Crítico']
    )
    
    return missing_data


def detectar_outliers_iqr(serie, multiplicador=1.5):
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).
    
    Parameters:
    -----------
    serie : pd.Series
        Serie numérica a analizar
    multiplicador : float, default=1.5
        Multiplicador del IQR para definir límites
    
    Returns:
    --------
    tuple : (outliers, lower_bound, upper_bound, indices)
    """
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplicador * IQR
    upper_bound = Q3 + multiplicador * IQR
    
    mask_outliers = (serie < lower_bound) | (serie > upper_bound)
    outliers = serie[mask_outliers]
    indices = serie.index[mask_outliers].tolist()
    
    return outliers, lower_bound, upper_bound, indices


def calcular_completitud(df):
    """
    Calcula métricas de completitud del dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    
    Returns:
    --------
    dict : Diccionario con métricas de completitud
    """
    total_celdas = df.shape[0] * df.shape[1]
    celdas_con_datos = df.notnull().sum().sum()
    celdas_missing = df.isnull().sum().sum()
    completitud = (celdas_con_datos / total_celdas) * 100
    
    return {
        'total_celdas': total_celdas,
        'celdas_con_datos': celdas_con_datos,
        'celdas_missing': celdas_missing,
        'completitud_pct': completitud,
        'variables_con_missing': (df.isnull().sum() > 0).sum(),
        'filas_completas': df.notnull().all(axis=1).sum()
    }


# =============================================================================
# ESTADÍSTICA DESCRIPTIVA
# =============================================================================

def estadisticas_grupo(df, variable, grupo):
    """
    Calcula estadísticas descriptivas por grupos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    variable : str
        Variable a analizar
    grupo : str
        Variable de agrupación
    
    Returns:
    --------
    pd.DataFrame : Estadísticas por grupo
    """
    stats_df = df.groupby(grupo)[variable].agg([
        ('n', 'count'),
        ('media', 'mean'),
        ('de', 'std'),
        ('mediana', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75))
    ]).round(2)
    
    return stats_df


def comparar_grupos(df, variable, grupo, prueba='ttest'):
    """
    Compara grupos usando pruebas estadísticas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    variable : str
        Variable a comparar
    grupo : str
        Variable de agrupación (debe tener 2 valores únicos para t-test)
    prueba : str, default='ttest'
        Tipo de prueba: 'ttest', 'mannwhitney', 'anova', 'kruskal'
    
    Returns:
    --------
    dict : Resultados de la prueba estadística
    """
    grupos_valores = df[grupo].unique()
    
    if prueba == 'ttest' and len(grupos_valores) == 2:
        grupo1 = df[df[grupo] == grupos_valores[0]][variable].dropna()
        grupo2 = df[df[grupo] == grupos_valores[1]][variable].dropna()
        statistic, pvalue = stats.ttest_ind(grupo1, grupo2)
        prueba_nombre = "t de Student"
        
    elif prueba == 'mannwhitney' and len(grupos_valores) == 2:
        grupo1 = df[df[grupo] == grupos_valores[0]][variable].dropna()
        grupo2 = df[df[grupo] == grupos_valores[1]][variable].dropna()
        statistic, pvalue = stats.mannwhitneyu(grupo1, grupo2)
        prueba_nombre = "Mann-Whitney U"
        
    elif prueba == 'anova':
        grupos_lista = [df[df[grupo] == val][variable].dropna() for val in grupos_valores]
        statistic, pvalue = stats.f_oneway(*grupos_lista)
        prueba_nombre = "ANOVA"
        
    elif prueba == 'kruskal':
        grupos_lista = [df[df[grupo] == val][variable].dropna() for val in grupos_valores]
        statistic, pvalue = stats.kruskal(*grupos_lista)
        prueba_nombre = "Kruskal-Wallis"
    
    else:
        raise ValueError("Prueba no válida o número de grupos incompatible")
    
    # Interpretación
    if pvalue < 0.001:
        interpretacion = "Altamente significativa (p < .001)"
    elif pvalue < 0.01:
        interpretacion = "Muy significativa (p < .01)"
    elif pvalue < 0.05:
        interpretacion = "Significativa (p < .05)"
    else:
        interpretacion = "No significativa (p ≥ .05)"
    
    return {
        'prueba': prueba_nombre,
        'estadistico': statistic,
        'p_valor': pvalue,
        'interpretacion': interpretacion,
        'n_grupos': len(grupos_valores)
    }


# =============================================================================
# CORRELACIONES
# =============================================================================

def matriz_correlaciones_ordenada(df, metodo='pearson', umbral_abs=0.3):
    """
    Calcula y ordena correlaciones por magnitud.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con variables numéricas
    metodo : str, default='pearson'
        Método de correlación: 'pearson', 'spearman', 'kendall'
    umbral_abs : float, default=0.3
        Valor absoluto mínimo para incluir en resultados
    
    Returns:
    --------
    pd.DataFrame : Correlaciones ordenadas por magnitud
    """
    # Calcular matriz de correlaciones
    corr_matrix = df.corr(method=metodo)
    
    # Extraer pares únicos
    pares = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            pares.append({
                'Variable_1': corr_matrix.columns[i],
                'Variable_2': corr_matrix.columns[j],
                'r': corr_matrix.iloc[i, j],
                'r_abs': abs(corr_matrix.iloc[i, j])
            })
    
    df_corr = pd.DataFrame(pares)
    df_corr = df_corr[df_corr['r_abs'] >= umbral_abs].sort_values('r_abs', ascending=False)
    
    # Clasificar fuerza
    def clasificar_correlacion(r):
        r_abs = abs(r)
        if r_abs >= 0.7:
            return 'Fuerte'
        elif r_abs >= 0.4:
            return 'Moderada'
        elif r_abs >= 0.2:
            return 'Débil'
        else:
            return 'Muy débil'
    
    df_corr['Fuerza'] = df_corr['r'].apply(clasificar_correlacion)
    df_corr['Direccion'] = df_corr['r'].apply(lambda x: 'Positiva' if x > 0 else 'Negativa')
    
    return df_corr


# =============================================================================
# CLUSTERING
# =============================================================================

def evaluar_clustering(X, clusters):
    """
    Evalúa la calidad de un clustering usando múltiples métricas.
    
    Parameters:
    -----------
    X : array-like
        Datos (debe estar escalado)
    clusters : array-like
        Asignación de clusters
    
    Returns:
    --------
    dict : Métricas de evaluación
    """
    n_clusters = len(np.unique(clusters))
    
    if n_clusters < 2:
        raise ValueError("Se requieren al menos 2 clusters para evaluar")
    
    metrics = {
        'n_clusters': n_clusters,
        'silhouette': silhouette_score(X, clusters),
        'calinski_harabasz': calinski_harabasz_score(X, clusters),
        'davies_bouldin': davies_bouldin_score(X, clusters)
    }
    
    # Distribución de clusters
    unique, counts = np.unique(clusters, return_counts=True)
    metrics['distribucion'] = dict(zip(unique, counts))
    
    # Tamaño del cluster más pequeño y más grande
    metrics['cluster_min_size'] = counts.min()
    metrics['cluster_max_size'] = counts.max()
    metrics['cluster_balance'] = counts.min() / counts.max()  # Cercano a 1 = balanceado
    
    return metrics


def preparar_datos_clustering(df, variables, escalar=True):
    """
    Prepara datos para clustering: selección, limpieza y escalado.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset original
    variables : list
        Lista de variables a usar
    escalar : bool, default=True
        Si True, escala los datos con StandardScaler
    
    Returns:
    --------
    tuple : (df_limpio, X_escalado, scaler)
    """
    # Seleccionar y limpiar
    df_cluster = df[variables].dropna()
    
    print(f"📊 Datos para clustering:")
    print(f"  - Variables: {len(variables)}")
    print(f"  - Casos originales: {len(df)}")
    print(f"  - Casos completos: {len(df_cluster)}")
    print(f"  - Pérdida: {(1 - len(df_cluster)/len(df))*100:.1f}%")
    
    if escalar:
        scaler = StandardScaler()
        X_escalado = scaler.fit_transform(df_cluster)
        return df_cluster, X_escalado, scaler
    else:
        return df_cluster, df_cluster.values, None


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def guardar_figura(ruta, dpi=300, facecolor='black', bbox_inches='tight'):
    """
    Guarda una figura de matplotlib con configuración estándar.
    
    Parameters:
    -----------
    ruta : str
        Ruta donde guardar la figura
    dpi : int, default=300
        Resolución en DPI
    facecolor : str, default='black'
        Color de fondo
    bbox_inches : str, default='tight'
        Ajuste de márgenes
    """
    try:
        plt.savefig(ruta, dpi=dpi, facecolor=facecolor, 
                   edgecolor='none', bbox_inches=bbox_inches)
        print(f"✓ Figura guardada: {ruta}")
        return True
    except Exception as e:
        print(f"✗ Error al guardar figura: {str(e)}")
        return False


def crear_figura_con_estilo(figsize=(12, 8), nrows=1, ncols=1, **kwargs):
    """
    Crea una figura con el estilo predefinido del proyecto.
    
    Parameters:
    -----------
    figsize : tuple, default=(12, 8)
        Tamaño de la figura
    nrows, ncols : int, default=1
        Número de subplots
    **kwargs : argumentos adicionales para plt.subplots()
    
    Returns:
    --------
    tuple : (fig, axes)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                            facecolor='black', **kwargs)
    
    # Configurar cada eje
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            configurar_eje(ax)
    else:
        configurar_eje(axes)
    
    return fig, axes


def configurar_eje(ax, fondo_negro=True):
    """
    Configura un eje con el estilo del proyecto.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Eje a configurar
    fondo_negro : bool, default=True
        Si True, aplica estilo de fondo negro
    """
    if fondo_negro:
        ax.set_facecolor('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')


# =============================================================================
# REPORTES
# =============================================================================

def crear_reporte_basico(df, nombre_archivo='reporte_basico.txt'):
    """
    Crea un reporte de texto básico del dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a reportar
    nombre_archivo : str, default='reporte_basico.txt'
        Nombre del archivo de salida
    """
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE BÁSICO DEL DATASET\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas\n\n")
        
        f.write("Tipos de datos:\n")
        for dtype, count in df.dtypes.value_counts().items():
            f.write(f"  - {dtype}: {count} variables\n")
        
        f.write("\nCompletitud:\n")
        completitud = calcular_completitud(df)
        f.write(f"  - Total de celdas: {completitud['total_celdas']:,}\n")
        f.write(f"  - Completitud: {completitud['completitud_pct']:.2f}%\n")
        f.write(f"  - Variables con missing: {completitud['variables_con_missing']}\n")
        f.write(f"  - Filas completas: {completitud['filas_completas']:,}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Reporte guardado: {nombre_archivo}")


# =============================================================================
# UTILIDADES VARIAS
# =============================================================================

def interpretar_pvalor(p, alpha=0.05):
    """
    Interpreta un p-valor según niveles de significancia estándar.
    
    Parameters:
    -----------
    p : float
        P-valor a interpretar
    alpha : float, default=0.05
        Nivel de significancia
    
    Returns:
    --------
    str : Interpretación del p-valor
    """
    if p < 0.001:
        return "*** (p < .001) - Altamente significativo"
    elif p < 0.01:
        return "** (p < .01) - Muy significativo"
    elif p < alpha:
        return f"* (p < {alpha}) - Significativo"
    else:
        return f"ns (p ≥ {alpha}) - No significativo"


def calcular_tamaño_efecto_cohen_d(grupo1, grupo2):
    """
    Calcula el tamaño del efecto d de Cohen para dos grupos.
    
    Parameters:
    -----------
    grupo1, grupo2 : array-like
        Datos de los dos grupos
    
    Returns:
    --------
    dict : d de Cohen e interpretación
    """
    mean1, mean2 = np.mean(grupo1), np.mean(grupo2)
    std1, std2 = np.std(grupo1, ddof=1), np.std(grupo2, ddof=1)
    n1, n2 = len(grupo1), len(grupo2)
    
    # Desviación estándar pooled
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    d = (mean1 - mean2) / pooled_std
    
    # Interpretación según Cohen (1988)
    if abs(d) < 0.2:
        interpretacion = "Trivial"
    elif abs(d) < 0.5:
        interpretacion = "Pequeño"
    elif abs(d) < 0.8:
        interpretacion = "Mediano"
    else:
        interpretacion = "Grande"
    
    return {
        'd': d,
        'd_abs': abs(d),
        'interpretacion': interpretacion
    }


def resumen_variable_categorica(df, variable):
    """
    Genera un resumen completo de una variable categórica.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    variable : str
        Nombre de la variable
    
    Returns:
    --------
    pd.DataFrame : Tabla resumen con frecuencias y porcentajes
    """
    resumen = pd.DataFrame({
        'Frecuencia': df[variable].value_counts(),
        'Porcentaje': df[variable].value_counts(normalize=True) * 100,
        'Porcentaje_Acumulado': df[variable].value_counts(normalize=True).cumsum() * 100
    })
    
    resumen['Porcentaje'] = resumen['Porcentaje'].round(2)
    resumen['Porcentaje_Acumulado'] = resumen['Porcentaje_Acumulado'].round(2)
    
    # Agregar fila de totales
    total = pd.DataFrame({
        'Frecuencia': [resumen['Frecuencia'].sum()],
        'Porcentaje': [100.0],
        'Porcentaje_Acumulado': [100.0]
    }, index=['TOTAL'])
    
    resumen = pd.concat([resumen, total])
    
    return resumen


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MÓDULO UTILS - FUNCIONES PARA EDA")
    print("="*80)
    print("\nEste módulo contiene funciones reutilizables para:")
    print("  ✓ Manejo de archivos y directorios")
    print("  ✓ Análisis de calidad de datos")
    print("  ✓ Estadística descriptiva")
    print("  ✓ Correlaciones")
    print("  ✓ Clustering")
    print("  ✓ Visualización")
    print("  ✓ Reportes")
    print("\nImportar con: from utils import *")
    print("O importar funciones específicas: from utils import cargar_dataset, analizar_valores_perdidos")
    print("="*80)