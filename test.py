from nogse_table_tools.params import read_sequence_params_xlsx

params = read_sequence_params_xlsx("Parámetros secuencias.xlsx")
# print(params.head())
# print(params.columns)
print(params[['sheet','seq','Hz','bmax','d_ms']].head(20))