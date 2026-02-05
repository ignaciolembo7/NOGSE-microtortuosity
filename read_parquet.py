# import pandas as pd
# df=pd.read_parquet(r'OGSE_signal\data\20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz000_b2000_QC-ROUTINE_20230619152408_10_results.long.parquet')
# print("Data shape (rows/cols):", df.shape)
# print('Ncols=', df.shape[1])
# print('\n'.join(df.columns.astype(str)))
# print(df[['stat','direction','b_step','bvalue','roi','value','value_norm','g']].iloc[0:200].to_string(index=False))
# import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 220)
# df=pd.read_parquet(r'OGSE_signal\data\20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz000_b2000_QC-ROUTINE_20230619152408_10_results.long.parquet')
# print(df.head(10).to_string(index=False))
import pandas as pd
df=pd.read_parquet(r'C:\Users\ignacio\docs\GitHub\NOGSE-microtortuosity\OGSE_signal\contrast\20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_QC-ROUTINE_20230619152408_12_results\contrast_N8-N4\20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_QC-ROUTINE_20230619152408_12_results.contrast_N8-N4.long.parquet')
df.to_excel(r'C:\Users\ignacio\docs\GitHub\NOGSE-microtortuosity\OGSE_signal\contrast\20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_QC-ROUTINE_20230619152408_12_results\contrast_N8-N4\20230619_BRAIN-3_ep2d_advdiff_AP_919D_OGSE_10bval_06dir_d55_Hz040_b0505_12_results.contrast_N8-N4.xlsx', index=False)