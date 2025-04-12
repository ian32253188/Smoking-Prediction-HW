import pandas as pd

# 讀取原始檔案
input_file = 'submission_20250411_160503.csv'
output_file = 'submission_binary.csv'

# 讀取 CSV 檔案
data = pd.read_csv(input_file)

# 將 `smoking` 欄位進行二值化處理
data['smoking'] = (data['smoking'] > 0.5).astype(int)

# 將結果儲存到新的檔案
data.to_csv(output_file, index=False)

print(f"已將二值化後的資料儲存到 {output_file}")