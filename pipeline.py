import pandas as pd
import json
import numpy as np
from pathlib import Path
from time_series_models import ARModel, MAModel, ARMAModel
from outlier_detectors import DiffDetector
import unicodedata


class DataPipeline:
    def __init__(self, input_path, output_dir, config_path, columns_to_process=None):
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.columns_to_process = columns_to_process
        
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Mapeo de modelos de series de tiempo
        self.ts_model_map = {
            'MA': MAModel,
            'AR': ARModel,
            'ARMA': ARMAModel
        }
    
    def normalize_column_name(self, name):
        """Normaliza nombres de columnas para comparación"""
        # Normalizar unicode (convierte caracteres acentuados a su forma base)
        normalized = unicodedata.normalize('NFKD', name)
        # Remover espacios extra
        normalized = ' '.join(normalized.split())
        return normalized
    
    def find_column_in_config(self, column_name):
        """Busca una columna en el config, manejando diferencias de encoding"""
        # Primero buscar coincidencia exacta
        if column_name in self.config:
            return column_name
        
        # Normalizar y buscar
        normalized_search = self.normalize_column_name(column_name)
        
        for config_col in self.config.keys():
            if self.normalize_column_name(config_col) == normalized_search:
                return config_col
        
        return None
    
    def get_models_for_column(self, column_name):
        """
        Obtiene el modelo de serie de tiempo y el detector de outliers
        configurados para una columna específica
        """
        # Buscar la columna en el config
        config_key = self.find_column_in_config(column_name)
        
        if config_key is None:
            raise ValueError(f"Columna '{column_name}' no encontrada en config")
        
        col_config = self.config[config_key]
        
        # Modelo de serie de tiempo
        ts_model_type = col_config.get('ts_model', 'MA')
        ts_params = col_config.get('ts_params', {})
        
        # Obtener parámetro 'q' del modelo
        q = ts_params.get('q', 2)
        
        # Para ARMA necesitamos p y q
        if ts_model_type == 'ARMA':
            p = ts_params.get('p', 1)
            ts_model = ARMAModel(p=p, q=q)
        else:
            TSModelClass = self.ts_model_map[ts_model_type]
            ts_model = TSModelClass(q=q)
        
        # Detector de outliers (siempre DiffDetector)
        detector_params = {
            'alpha': ts_params.get('alpha', 0.005),
            'quantile': ts_params.get('quantile', 0.995),
            'factor_olvido': ts_params.get('factor_olvido', 0.02),
            'lag_cambio': ts_params.get('lag_cambio', 2),
            'suavizado': ts_params.get('suavizado', 7),
            'change_quantile': ts_params.get('change_quantile', 0.99)
        }
        
        detector = DiffDetector(**detector_params)
        
        return ts_model, detector
    
    def process_column(self, column_name, data):
        """Procesa una columna individual"""
        print(f"\nProcesando columna: {column_name}")
        
        # Obtener modelos específicos
        ts_model, detector = self.get_models_for_column(column_name)
        
        # Limpiar datos (remover NaN)
        clean_data = data.dropna().astype(float).values
        
        if len(clean_data) < 10:
            print(f"  ⚠️  Advertencia: muy pocos datos válidos ({len(clean_data)})")
            return None
        
        # Ajustar modelo de serie de tiempo y obtener residuos
        print(f"  - Ajustando modelo: {ts_model.__class__.__name__} (q={ts_model.q})")
        residuals = ts_model.get_residuals(clean_data)
        
        # Detectar outliers
        print(f"  - Detectando outliers con: {detector.__class__.__name__}")
        labels, outlier_score, change_score = detector.detect(clean_data, residuals)
        
        return {
            'values': clean_data,
            'residuals': residuals,
            'labels': labels,
            'outlier_score': outlier_score,
            'change_score': change_score
        }
    
    def run(self):
        """Ejecuta el pipeline completo"""
        # Leer CSV con encoding correcto
        try:
            df = pd.read_csv(self.input_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("⚠️  Error con UTF-8, intentando con latin-1...")
            df = pd.read_csv(self.input_path, encoding='latin-1')
        
        print(f"CSV cargado: {df.shape[0]} filas, {df.shape[1]} columnas\n")
        
        # Determinar qué columnas procesar
        if self.columns_to_process is None:
            # Procesar todas las columnas del CSV que estén en el config
            columns_to_run = []
            for csv_col in df.columns:
                if self.find_column_in_config(csv_col) is not None:
                    columns_to_run.append(csv_col)
            
            print(f"📋 Procesando TODAS las columnas del config ({len(columns_to_run)} columnas)")
        else:
            # Procesar solo las columnas especificadas
            columns_to_run = self.columns_to_process
            print(f"📋 Procesando columnas especificadas: {columns_to_run}")
            
            # Verificar que las columnas especificadas estén en el config
            missing_in_config = []
            valid_columns = []
            
            for col in columns_to_run:
                if self.find_column_in_config(col) is None:
                    missing_in_config.append(col)
                else:
                    valid_columns.append(col)
            
            if missing_in_config:
                print(f"⚠️  Advertencia: Las siguientes columnas no están en config.json:")
                for col in missing_in_config:
                    print(f"    - {col}")
                
                # Mostrar columnas disponibles en config
                print(f"\n📋 Columnas disponibles en config.json:")
                for i, col in enumerate(self.config.keys(), 1):
                    print(f"    {i}. {col}")
                print()
                
            columns_to_run = valid_columns
        
        if not columns_to_run:
            print("❌ No hay columnas para procesar.")
            return
        
        print()
        
        # Procesar cada columna
        processed_count = 0
        error_count = 0
        
        for column in columns_to_run:
            if column not in df.columns:
                print(f"⚠️  Columna '{column}' no existe en CSV")
                error_count += 1
                continue
            
            try:
                # Procesar la columna
                result = self.process_column(column, df[column])
                
                if result is None:
                    error_count += 1
                    continue
                
                # Crear DataFrame con resultados
                result_df = pd.DataFrame({
                    'index': range(len(result['values'])),
                    'value': result['values'],
                    'residual': result['residuals'],
                    'outlier_score': result['outlier_score'],
                    'change_score': result['change_score'],
                    'label': result['labels']
                })
                
                # Guardar CSV
                output_path = self.output_dir / f"{column}_labeled.csv"
                result_df.to_csv(output_path, index=False)
                
                # Estadísticas
                outlier_count = (result['labels'] == 'outlier').sum()
                change_count = (result['labels'] == 'change').sum()
                normal_count = (result['labels'] == 'normal').sum()
                
                print(f"  ✓ Normal: {normal_count}, Outliers: {outlier_count}, Changes: {change_count}")
                print(f"  ✓ Guardado: {output_path}")
                
                processed_count += 1
                
            except Exception as e:
                print(f"  ✗ Error procesando {column}: {str(e)}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ Pipeline completado")
        print(f"  - Columnas procesadas: {processed_count}")
        print(f"  - Errores: {error_count}")
        print(f"  - Resultados en: {self.output_dir}")
        print(f"{'='*60}")