import pandas as pd

# Replace the predicted measurement error with the original error value from the data
mocap = pd.read_csv('mocap.csv')
mocap = mocap.replace(-2147483648, -999999000)
mocap.to_csv('mocap.csv', index=False, header=True)