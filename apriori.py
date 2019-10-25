# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)


#visualising the results
final_results=pd.DataFrame(columns=['Products', 'Likely', 'Support', 'Confidence', 'Lift'])
for i in results:
  final_results=final_results.append({'Products':list(i.ordered_statistics[0].items_base),
                                      'Likely':list(i.ordered_statistics[0].items_add),
                                      'Support':i[1],
                                      'Confidence':i.ordered_statistics[0].confidence,
                                      'Lift':i.ordered_statistics[0].lift,},
                                        ignore_index=True,)

final_results=final_results.sort_values(by='Lift', ascending=False)



