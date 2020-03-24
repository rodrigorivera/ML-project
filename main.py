import pandas as pd
import numpy as np
from XXXXXXXXXX.preprocessing import preprocess
from XXXXXXXXXX.theta_method.ses_theta import sesThetaF
from XXXXXXXXXX.theta_method.general_theta import sThetaF


FORECAST_LENGTH = 28

path = 'data/'
calendar = pd.read_csv(path+'calendar.csv')
sales = pd.read_csv(path+'sales_train_validation.csv')
# sample = pd.read_csv(path+'sample_submission.csv')
# sell_prices = pd.read_csv(path+'sell_prices.csv')

sales = preprocess(sales, flag_trim=True, history_period=730)

dim_vert, dim_hor = sales.shape
forecast_df = pd.DataFrame(np.zeros((dim_vert*2, FORECAST_LENGTH+1)),
                           columns=['id'] + ['F' + str(i + 1) for i in range(FORECAST_LENGTH)])
ids = np.concatenate((sales['id'].values, sales['id'].str.replace('_validation', '_evaluation')))
forecast_df['id'] = ids

sales.set_index('id', inplace=True)

dates_sales = calendar.loc[calendar.d.isin(sales.columns), 'date'].values
for index, row in sales.iterrows():
    data = pd.DataFrame(row.values, index=dates_sales)
    forecast = sThetaF(data, s_period=7, h=28)
    forecast_df.loc[forecast_df.id==index, :] = np.concatenate(([index], forecast['mean'].flatten()))

forecast_df.to_csv('Theta.Nikita.csv', index=False)
