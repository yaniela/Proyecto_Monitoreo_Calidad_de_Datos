import pandas as pd
import json
import numpy as np
from pathlib import Path
from time_series_models import ARModel, MAModel, ARMAModel
from outlier_detectors import AdaptiveVarianceDetector, DiffDetector
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
        
        # Mapeo de detectores
        self.detector_map = {
            'adaptive_variance': AdaptiveVarianceDetector,
            'diff': DiffDetector
        }
    
    def normalize_column_name(self, name):
        """Normaliza nombres de columnas para comparación"""
        normalized = unicodedata.normalize('NFKD', name)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def find_column_in_config(self, column_name):
        """Busca una columna en el config, manejando diferencias de encoding"""
        if column_name in self.config:
            return column_name
        
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
        config_key = self.find_column_in_config(column_name)
        
        if config_key is None:
            raise ValueError(f"Columna '{column_name}' no encontrada en config")
        
        col_config = self.config[config_key]
        
        # Detector de outliers
        detector_type = col_config.get('outlier_detector', 'adaptive_variance')
        
        # Modelo de serie de tiempo (solo para adaptive_variance)
        ts_model = None
        if detector_type == 'adaptive_variance':
            ts_model_type = col_config.get('ts_model', 'MA')
            ts_params = col_config.get('ts_params', {})
            q = ts_params.get('q', 2)
            
            if ts_model_type == 'ARMA':
                p = ts_params.get('p', 1)
                ts_model = ARMAModel(p=p, q=q)
            else:
                TSModelClass = self.ts_model_map[ts_model_type]
                ts_model = TSModelClass(q=q)
        
        # Configurar detector según tipo
        if detector_type == 'adaptive_variance':
            ts_params = col_config.get('ts_params', {})
            detector_params = {
                'alpha': ts_params.get('alpha', 0.005),
                'quantile': ts_params.get('quantile', 0.995),
                'factor_olvido': ts_params.get('factor_olvido', 0.02),
                'lag_cambio': ts_params.get('lag_cambio', 2),
                'suavizado': ts_params.get('suavizado', 7),
                'change_quantile': ts_params.get('change_quantile', 0.99)
            }
        elif detector_type == 'diff':
            outlier_params = col_config.get('outlier_params', {})
            detector_params = {
                'lambda_centrada': outlier_params.get('lambda_centrada', None),
                'k': outlier_params.get('k', 0)
            }
        else:
            raise ValueError(f"Detector desconocido: {detector_type}")
        
        DetectorClass = self.detector_map[detector_type]
        detector = DetectorClass(**detector_params)
        
        return ts_model, detector, detector_type
    
    def process_column(self, column_name, data, date_time=None):
        """Procesa una columna individual"""
        print(f"\nProcesando columna: {column_name}")
        
        # Obtener modelos específicos
        ts_model, detector, detector_type = self.get_models_for_column(column_name)
        
        # Limpiar datos (remover NaN)
        clean_indices = data.notna()
        clean_data = data[clean_indices].astype(float).values
        
        if date_time is not None:
            clean_date_time = date_time[clean_indices].values
        else:
            clean_date_time = None
        
        if len(clean_data) < 10:
            print(f"  ⚠️  Advertencia: muy pocos datos válidos ({len(clean_data)})")
            return None
        
        # Procesar según el tipo de detector
        if detector_type == 'adaptive_variance':
            print(f"  - Ajustando modelo: {ts_model.__class__.__name__} (q={ts_model.q})")
            residuals = ts_model.get_residuals(clean_data)
            
            print(f"  - Detectando outliers con: {detector.__class__.__name__}")
            result = detector.detect(clean_data, residuals)
            
            return {
                'date_time': clean_date_time,
                'values': clean_data,
                'residuals': result['residuals'],
                'labels': result['labels'],
                'outlier_score': result['outlier_score'],
                'change_score': result['change_score'],
                'detector_type': detector_type
            }
        
        elif detector_type == 'diff':
            print(f"  - Detectando outliers con: {detector.__class__.__name__}")
            result = detector.detect(clean_data)
            
            return {
                'date_time': clean_date_time,
                'values': clean_data,
                'valores_sin_outliers': result['valores_sin_outliers'],
                'labels': result['labels'],
                'diff_centrada_score': result['diff_centrada_score'],
                'diff_score': result['diff_score'],
                'lambda_centrada': result['lambda_centrada_usado'],
                'k_usado': result['k_usado'],
                'detector_type': detector_type
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
        
        # Verificar si existe columna date_time
        has_datetime = 'date_time' in df.columns
        if has_datetime:
            df['date_time'] = pd.to_datetime(df['date_time'])
        
        # Determinar qué columnas procesar
        if self.columns_to_process is None:
            columns_to_run = []
            for csv_col in df.columns:
                if csv_col == 'date_time':
                    continue
                if self.find_column_in_config(csv_col) is not None:
                    columns_to_run.append(csv_col)
            
            print(f"📋 Procesando TODAS las columnas del config ({len(columns_to_run)} columnas)")
        else:
            columns_to_run = self.columns_to_process
            print(f"📋 Procesando columnas especificadas: {columns_to_run}")
            
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
                date_time_col = df['date_time'] if has_datetime else None
                result = self.process_column(column, df[column], date_time_col)
                
                if result is None:
                    error_count += 1
                    continue
                
                # Crear DataFrame según el tipo de detector
                if result['detector_type'] == 'adaptive_variance':
                    result_dict = {
                        'index': range(len(result['values'])),
                        'value': result['values'],
                        'residual': result['residuals'],
                        'outlier_score': result['outlier_score'],
                        'change_score': result['change_score'],
                        'label': result['labels']
                    }
                    if result['date_time'] is not None:
                        result_dict = {'date_time': result['date_time'], **result_dict}
                    
                elif result['detector_type'] == 'diff':
                    result_dict = {
                        'value': result['values'],
                        'valores_sin_outliers': result['valores_sin_outliers'],
                        'label': result['labels']
                    }
                    if result['date_time'] is not None:
                        result_dict = {'date_time': result['date_time'], **result_dict}
                
                result_df = pd.DataFrame(result_dict)
                
                # Guardar CSV
                output_path = self.output_dir / f"{column}_labeled.csv"
                result_df.to_csv(output_path, index=False)
                
                # Estadísticas
                if result['detector_type'] == 'adaptive_variance':
                    outlier_count = (result['labels'] == 'outlier').sum()
                    change_count = (result['labels'] == 'change').sum()
                    normal_count = (result['labels'] == 'normal').sum()
                    print(f"  ✓ Normal: {normal_count}, Outliers: {outlier_count}, Changes: {change_count}")
                elif result['detector_type'] == 'diff':
                    outlier_count = (result['labels'] == 'outlier').sum()
                    normal_count = (result['labels'] == 'normal').sum()
                    print(f"  ✓ Normal: {normal_count}, Outliers: {outlier_count}")
                    print(f"  ✓ Lambda centrada usado: {result['lambda_centrada']:.4f}")
                    print(f"  ✓ K usado: {result['k_usado']:.4f}")  
                
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