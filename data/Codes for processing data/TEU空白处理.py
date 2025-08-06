"""
    此代码用于处理到港数据中teu的空白行
"""
import pandas as pd

file_path = "洛杉矶_长滩到港.csv"
df = pd.read_csv(file_path)

#处理teu的空白行
df['teu'] = df.groupby('shiptype')['teu'].transform(lambda x: x.fillna(x.mean()))
df['teu'] = df['teu'].fillna(0)
#将时间转变为标准格式
df['start_postime'] = pd.to_datetime(df['start_postime'], format='mixed')
output_path = "处理后的_洛杉矶_长滩到港.csv"
df.to_csv(output_path, index=False)



