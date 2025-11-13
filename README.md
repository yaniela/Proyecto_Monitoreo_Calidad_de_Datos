# Pipeline de Detección de Outliers en Series de Tiempo

Pipeline modular para detectar outliers y cambios en series de tiempo usando modelos AR, MA, ARMA y dos métodos de detección diferentes.

## 📋 Tabla de Contenidos
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Formato de Datos](#formato-de-datos)
- [Uso Básico](#uso-básico)
- [Configuración](#configuración)
- [Métodos de Detección](#métodos-de-detección)
- [Parámetros Sugeridos](#parámetros-sugeridos)

## 🔧 Requisitos

- Python 3.8+
- pandas
- numpy
- statsmodels
- changefinder

## 📦 Instalación
```bash
# Clonar el repositorio
git clone 
cd 

# Instalar dependencias
pip install pandas numpy statsmodels changefinder
```

## 📊 Formato de Datos

El CSV de entrada **debe** tener las siguientes características:

### Estructura requerida:
```csv
date_time,variable1,variable2,variable3,...
2025-01-01 00:00:00,10.5,20.3,15.8,...
2025-01-01 00:05:00,10.7,20.1,15.9,...
...
```

### Requisitos:
- ✅ **Columna `date_time` obligatoria**: Formato `YYYY-MM-DD HH:MM:SS`
- ✅ **Variables numéricas**: Todas las demás columnas deben ser valores numéricos (float/int)
- ✅ **Valores faltantes**: Se pueden tener NaN, serán manejados automáticamente
- ✅ **Encoding**: UTF-8 o Latin-1 (se detecta automáticamente)

### Ejemplo:
```csv
date_time,temperatura,presion,flujo
2025-01-01 00:00:00,25.3,101.2,150.5
2025-01-01 00:05:00,25.5,101.3,151.2
2025-01-01 00:10:00,25.4,101.1,150.8
```

## 🚀 Uso Básico

### 1. Generar archivo de configuración
```bash
# Genera config.json con valores por defecto para todas las columnas
python generar_config.py datos.csv

# Con nombre personalizado
python generar_config.py datos.csv --output mi_config.json

# Sobrescribir sin preguntar
python generar_config.py datos.csv --overwrite
```

Esto generará un `config.json` con configuración por defecto para cada columna (excepto `date_time` que se excluye automáticamente).

### 2. (Opcional) Editar configuración

Abre `config.json` y ajusta los parámetros para las columnas que necesites personalizar.

### 3. Ejecutar el pipeline
```bash
# Procesar todas las columnas del config
python main.py datos.csv --config config.json

# Procesar columnas específicas
python main.py datos.csv --config config.json --columns "temperatura" "presion"

# Procesar una sola columna
python main.py datos.csv --config config.json --columns "flujo"

# Especificar directorio de salida
python main.py datos.csv --config config.json --output-dir resultados
```

### 4. Resultados

Los archivos etiquetados se guardarán en la carpeta `output/` (o la especificada) con el formato:
```
<nombre_columna>_labeled.csv
```

## ⚙️ Configuración

### Estructura del config.json
```json
{
  "nombre_columna": {
    "ts_model": "MA",
    "ts_params": {
      "q": 2,
      "alpha": 0.005,
      "quantile": 0.995,
      "factor_olvido": 0.02,
      "lag_cambio": 2,
      "suavizado": 7,
      "change_quantile": 0.99
    },
    "outlier_detector": "diff",
    "outlier_params": {
      "lambda_centrada": 12,
      "k": 0
    }
  }
}
```

### Parámetros por defecto:

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `ts_model` | `"MA"` | Modelo de serie de tiempo (MA, AR, ARMA) |
| `q` | `2` | Orden del modelo MA/AR |
| `alpha` | `0.005` | Factor de suavizado para varianza adaptativa |
| `quantile` | `0.995` | Cuantil para threshold de outliers |
| `factor_olvido` | `0.02` | Factor de olvido para ChangeFinder |
| `lag_cambio` | `2` | Orden AR para ChangeFinder |
| `suavizado` | `7` | Ventana de suavizado para ChangeFinder |
| `change_quantile` | `0.99` | Cuantil para threshold de cambios |
| `outlier_detector` | `"diff"` | Método de detección (diff o adaptive_variance) |
| `lambda_centrada` | `12` | Threshold para diferencia centrada |
| `k` | `0` | Threshold para diferencia con dato anterior (0 = auto) |

## 🔍 Métodos de Detección

### 1. Método "diff" (Por defecto)

Detecta outliers basándose en diferencias centradas. Es el método recomendado para la mayoría de casos.

**Criterios de outlier:**
- `diff_centrada >= lambda_centrada`
- `diff_prev >= k`

**Configuración:**
```json
"outlier_detector": "diff",
"outlier_params": {
  "lambda_centrada": 12,  // 0 = auto (cuantil 0.99 - cuantil 0.01)
  "k": 0                  // 0 = auto (cuantil 0.99 - cuantil 0.01)
}
```

**Output:**
- `date_time`: Timestamp
- `value`: Valor original
- `valores_sin_outliers`: Valor corregido (outliers reemplazados)
- `label`: "normal" o "outlier"

**Ajuste de cuantiles:**
Los cuantiles para calcular `lambda_centrada` y `k` se pueden modificar en `outlier_detectors.py`:
```python
class DiffDetector(OutlierDetector):
    def __init__(self, lambda_centrada=None, k=0, quantile_low=0.01, quantile_high=0.99):
        # quantile_low: cuantil inferior (default: 0.01)
        # quantile_high: cuantil superior (default: 0.99)
```

### 2. Método "adaptive_variance"

Usa varianza adaptativa sobre residuos de modelos AR/MA/ARMA y ChangeFinder para detectar cambios de régimen.

**Configuración:**
```json
"ts_model": "AR",
"ts_params": {
  "q": 2,
  "alpha": 0.005,
  "quantile": 0.995,
  "factor_olvido": 0.02,
  "lag_cambio": 2,
  "suavizado": 7,
  "change_quantile": 0.99
},
"outlier_detector": "adaptive_variance"
```

**Output:**
- `date_time`: Timestamp
- `value`: Valor original
- `residual`: Residuo del modelo
- `outlier_score`: Score de outlier
- `change_score`: Score de cambio
- `label`: "normal", "outlier" o "change"

## 📈 Parámetros Sugeridos

### Para el método "diff":

| Tipo de variable | lambda_centrada | k | Descripción |
|------------------|-----------------|---|-------------|
| Flujo estable | 8-15 | 0 | Variables con baja variabilidad |
| Temperatura | 10-20 | 0 | Variables con cambios graduales |
| Presión | 5-12 | 0 | Variables sensibles |
| Flujo variable | 15-25 | 0 | Alta variabilidad esperada |

**Recomendación:** Usar `lambda_centrada = 0` y `k = 0` para cálculo automático basado en los cuantiles de los datos.

### Para el método "adaptive_variance":

| Parámetro | Rango sugerido | Uso |
|-----------|----------------|-----|
| `q` (AR/MA) | 1-5 | Orden del modelo (empezar con 2) |
| `alpha` | 0.001-0.01 | Menor = más suave, Mayor = más reactivo |
| `quantile` | 0.99-0.999 | Menor = más sensible, Mayor = más estricto |
| `factor_olvido` | 0.01-0.05 | Adaptación de ChangeFinder |
| `lag_cambio` | 1-3 | Orden AR para detección de cambios |
| `suavizado` | 5-15 | Ventana de suavizado |
| `change_quantile` | 0.95-0.999 | Sensibilidad para cambios de régimen |

## 📂 Estructura del Proyecto
```
.
├── data/                       # Carpeta con datos (ignorada en git)
├── output/                     # Resultados del pipeline (ignorada en git)
├── config.json                 # Configuración generada
├── generar_config.py          # Script para generar configuración
├── main.py                    # Punto de entrada del pipeline
├── pipeline.py                # Lógica principal del pipeline
├── time_series_models.py      # Modelos AR, MA, ARMA
├── outlier_detectors.py       # Detectores de outliers
├── .gitignore                 # Archivos ignorados por git
└── README.md                  # Este archivo
```

## 💡 Ejemplos de Uso

### Ejemplo 1: Pipeline completo con valores por defecto
```bash
# 1. Generar config
python generar_config.py data/planta.csv

# 2. Ejecutar pipeline en todas las columnas
python main.py data/planta.csv --config config.json
```

### Ejemplo 2: Procesar columnas específicas
```bash
python main.py data/planta.csv --config config.json \
  --columns "Flujo Digestor 1" "Flujo Digestor 2" "Temperatura"
```

### Ejemplo 3: Cambiar método de detección

Edita `config.json` y cambia el detector para una columna específica:
```json
{
  "temperatura": {
    "ts_model": "AR",
    "ts_params": {
      "q": 2,
      ...
    },
    "outlier_detector": "adaptive_variance"  // Cambiar aquí
  }
}
```

### Ejemplo 4: Ajustar sensibilidad del método diff

Para hacer el detector **más estricto** (detecta menos outliers):
```json
"outlier_params": {
  "lambda_centrada": 20,  // Aumentar threshold
  "k": 0
}
```

Para hacer el detector **más sensible** (detecta más outliers):
```json
"outlier_params": {
  "lambda_centrada": 5,   // Disminuir threshold
  "k": 0
}
```

## 🐛 Solución de Problemas

### Error de encoding
Si aparece error con caracteres especiales (á, é, í, ó, ú, ñ):
- El pipeline intenta automáticamente UTF-8 y Latin-1
- Los nombres de columnas se normalizan automáticamente

### Columna no encontrada en config
Verifica que:
1. El nombre de la columna coincida exactamente (incluyendo espacios)
2. La columna exista en el CSV
3. No sea la columna `date_time` (se excluye automáticamente)