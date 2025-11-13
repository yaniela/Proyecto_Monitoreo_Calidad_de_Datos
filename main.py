import argparse
from pipeline import DataPipeline

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline de detección de outliers en series de tiempo'
    )
    parser.add_argument('input_csv', help='Ruta al CSV de entrada')
    parser.add_argument('--config', required=True,
                       help='Archivo de configuración JSON')
    parser.add_argument('--output-dir', default='output',
                       help='Directorio de salida')
    parser.add_argument('--columns', nargs='+', default=None,
                       help='Columnas específicas a procesar. Si no se especifica, procesa todas las del config')
    parser.add_argument('--all', action='store_true',
                       help='Procesar todas las columnas del config')
    
    args = parser.parse_args()
    
    # Determinar qué columnas procesar
    if args.all or args.columns is None:
        columns_to_process = None  # None significa procesar todas
    else:
        columns_to_process = args.columns
    
    pipeline = DataPipeline(
        args.input_csv, 
        args.output_dir, 
        args.config,
        columns_to_process=columns_to_process
    )
    pipeline.run()

if __name__ == "__main__":
    main()