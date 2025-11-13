import pandas as pd
import json
import argparse
from pathlib import Path
import copy

class ConfigGenerator:
    def __init__(self, csv_path, output_path='config.json'):
        self.csv_path = csv_path
        self.output_path = output_path
        
        # Columnas a excluir
        self.exclude_columns = ['date_time']
        
        # Configuración por defecto
        self.default_config = {
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
            "outlier_detector": "diff",
            "outlier_params": {
                "lambda_centrada": 12,
                "k": 0
            }
        }
    
    def generate(self, overwrite=False):
        """Genera el archivo de configuración"""
        
        # Verificar si ya existe
        if Path(self.output_path).exists() and not overwrite:
            print(f"⚠️  El archivo {self.output_path} ya existe.")
            response = input("¿Deseas sobrescribirlo? (s/n): ")
            if response.lower() != 's':
                print("Operación cancelada.")
                return
        
        # Leer CSV
        print(f"Leyendo CSV: {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("⚠️  Error con UTF-8, intentando con latin-1...")
            df = pd.read_csv(self.csv_path, encoding='latin-1')
        
        # Filtrar columnas (excluir date_time y similares)
        valid_columns = [col for col in df.columns if col not in self.exclude_columns]
        
        # Generar configuración para cada columna válida
        config = {}
        for column in valid_columns:
            config[column] = copy.deepcopy(self.default_config)
        
        # Guardar JSON
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Configuración generada: {self.output_path}")
        print(f"✓ Columnas detectadas: {len(df.columns)}")
        print(f"✓ Columnas excluidas: {[col for col in df.columns if col in self.exclude_columns]}")
        print(f"✓ Columnas incluidas: {len(valid_columns)}")
        print(f"\nColumnas procesadas:")
        for i, col in enumerate(valid_columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\n📝 Puedes editar manualmente {self.output_path} para ajustar parámetros específicos")
    
    def generate_with_presets(self, presets=None):
        """
        Genera configuración con presets para columnas específicas
        
        presets: dict con formato {
            'nombre_columna': {
                'ts_model': 'ARMA',
                'ts_params': {'p': 1, 'q': 1},
                ...
            }
        }
        """
        # Leer CSV
        df = pd.read_csv(self.csv_path)
        
        # Filtrar columnas
        valid_columns = [col for col in df.columns if col not in self.exclude_columns]
        
        # Generar configuración
        config = {}
        for column in valid_columns:
            if presets and column in presets:
                config[column] = presets[column]
            else:
                config[column] = copy.deepcopy(self.default_config)
        
        # Guardar JSON
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Configuración generada con presets: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generador de archivo de configuración para el pipeline'
    )
    parser.add_argument('csv_path', help='Ruta al CSV de entrada')
    parser.add_argument('--output', '-o', default='config.json',
                       help='Nombre del archivo de configuración (default: config.json)')
    parser.add_argument('--overwrite', '-f', action='store_true',
                       help='Sobrescribir sin preguntar si ya existe')
    
    args = parser.parse_args()
    
    # Crear generador y ejecutar
    generator = ConfigGenerator(args.csv_path, args.output)
    generator.generate(overwrite=args.overwrite)

if __name__ == "__main__":
    main()