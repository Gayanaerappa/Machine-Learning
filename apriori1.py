import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter']
]

te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)

df = pd.DataFrame(te_data, columns=te.columns_)

frequent_items = apriori(df, min_support=0.5, use_colnames=True)

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)

print(frequent_items)
print(rules)
