import numpy as np
import pandas as pd


def SIR_model(SIR, beta, gamma, max_population):
    '''
    Simple SIR model
        S: susceptible population
        I: infected people
        R: recovered people
        beta: infection rate
        gamma: recovery rate

        overall condition is that the sum of changes (differences) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    '''

    N0 = max_population
    S, I, R = SIR
    dS_dt = -beta * S * I / N0  #
    dI_dt = beta * S * I / N0 - gamma * I
    dR_dt = gamma * I

    return ([dS_dt, dI_dt, dR_dt])


# df = pd.read_csv("data/raw/country_populations.csv")[["Country Name", "Country Code", "2019 [YR2019]"]]
# pop.rename(columns={'Country Name': 'country', '2019 [YR2019]': 'count'}, inplace = True)
# # pop.country = pop.country.map(lambda x: str(x))
# df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')
# common_countries = set(df_input_large.country.map(lambda x: x.lower())).intersection(set(pop.country.map(lambda x: x.lower())))
# df_input_large_filtered = df_input_large[df_input_large.country.map(lambda x: x.lower()).isin(common_countries)]
# pop = pop[pop.country.map(lambda x: x.lower()).isin(common_countries)]
# country_vs_pop = {row['country']:row['count'] for row in pop.to_dict('records')}

df = pd.read_csv("data/raw/populations_2019.csv")
df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')

print(df_input_large.columns)