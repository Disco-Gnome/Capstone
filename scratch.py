import csv
import zipfile

with zipfile.ZipFile('csv_pus.zip', 'r') as z:
    with z.open('psam_pusa.csv', 'r') as f:
        vars = next(f)

#%%