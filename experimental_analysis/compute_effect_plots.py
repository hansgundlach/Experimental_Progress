


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%

#helper functionality to get results fast 
def compute_effect(loss_1, loss_2):
    irreducible = 1.7
    return np.exp(-(np.log(loss_1-irreducible)-np.log(loss_2-irreducible))/.155)




# %%
# compare experiment with sgd and lstm vs just lstm 
# compare expeiment with sgd and 

loss_sgd_no_lstm = pd.read_csv("../experimental_data_folder/Two_Changes_Experiemnts/LST.csv")



