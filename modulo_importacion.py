import pandas as pd
import sqlite3

def importar_archivo(file_path):
    """
    Función para cargar un archivo CSV, Excel o SQLite y devolver un DataFrame.
    Maneja los casos en los que el archivo esté corrupto, vacío o no sea compatible.
    """
    try:
        extension = file_path.split('.')[-1].lower()

        if extension == 'csv':
            data_frame = pd.read_csv(file_path)
        elif extension in ['xlsx', 'xls']:
            data_frame = pd.read_excel(file_path, engine='openpyxl')
        elif extension in ['sqlite', 'db']:
            conn = sqlite3.connect(file_path)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            if tables.empty:
                raise ValueError("El archivo no contiene tablas.")
            table_name = tables['name'].iloc[0]
            data_frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            conn.close()
        else:
            raise ValueError("Tipo de archivo no soportado.")

        # Validar que el DataFrame no esté vacío
        if data_frame.empty:
            raise ValueError("El archivo no contiene información.")

        # Validar que las columnas contengan los tipos de datos esperados
        if not all(isinstance(val, (int, float, str)) for col in data_frame.columns for val in data_frame[col]):
            raise ValueError("El archivo contiene datos malformados o ilegibles.")
        
        return data_frame

    except pd.errors.EmptyDataError:
        raise ValueError("El archivo está vacío o no contiene datos legibles.")
    except pd.errors.ParserError:
        raise ValueError("El archivo tiene un formato incorrecto o está dañado.")
    except Exception as e:
        raise ValueError(f"Error al leer el archivo: {str(e)}")
