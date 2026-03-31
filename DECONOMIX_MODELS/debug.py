import pandas as pd
metrics = pd.read_csv('deconvolution_metrics.csv')
print(metrics.sort_values('spearman_correlation', ascending=False).head(15))