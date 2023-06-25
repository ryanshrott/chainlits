import datetime
import json
def get_table_names(cursor):
    """Return a list of table names."""
    table_names = []
    cursor.execute("""SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'""")
    tables = cursor.fetchall()
    for table in tables:
        table_names.append(table[0])
    return table_names

def get_unique_values(cursor, table_name, column_name):
    """Return a list of unique values in a given column."""
    cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name};")
    unique_values = cursor.fetchall()
    return [val[0] for val in unique_values]


def get_column_names(cursor, table_name):
    """Return a list of column names."""
    column_names = []
    cursor.execute(f"""SELECT column_name FROM information_schema.columns
                      WHERE table_name = '{table_name}';""")
    columns = cursor.fetchall()
    for col in columns:
        column_names.append(col[0])
    return column_names
def get_column_names_and_types(cursor, table_name):
    """Return a list of tuples with column names and types."""
    column_info = []
    cursor.execute(f"""SELECT column_name, data_type FROM information_schema.columns
                      WHERE table_name = '{table_name}';""")
    columns = cursor.fetchall()
    for col in columns:
        if col[0] == "_id":  # Skip the "_id" column
            continue
        column_info.append(col)
    return column_info

def get_example_row(cursor, table_name, column_names):
    """Return an example row from the table as a dict mapping column names to values."""
    cursor.execute(f"SELECT {', '.join(column_names)} FROM {table_name} WHERE house_category = 'Condo' ORDER BY {', '.join(f'{name} IS NULL' for name in column_names)} LIMIT 1")
    row = cursor.fetchone()
    row_dict = {column_names[i]: ((str(row[i])[:100] + "...") if isinstance(row[i], str) and len(str(row[i])) > 100 else row[i]) for i in range(len(row))}
    return row_dict
def default(o):
    """Custom serializer for unsupported types."""
    if isinstance(o, datetime.datetime):
        return o.isoformat()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def get_database_info(cursor):
    """Return a string containing detailed information for each table in the database."""
    table_info = []
    for table_name in get_table_names(cursor):
        if table_name != 'chat':
            continue
        columns_info = []
        for info in get_column_names_and_types(cursor, table_name):
            column_info = {"column_name": info[0], "data_type": info[1]}
            if info[0] == 'house_category':
                column_info['unique_values'] = get_unique_values(cursor, table_name, info[0])
            columns_info.append(column_info)

        column_names = [info["column_name"] for info in columns_info]
        example_row = get_example_row(cursor, table_name, column_names)
        description = f"The {table_name} table holds information about properties currently for sale in Ontario, BC and Alberta."
        table_info.append({
            "table_name": table_name,
            "description": description,
            "columns": columns_info,
            "example_row": example_row
        })
    return json.dumps(table_info, indent=2, default=default)


