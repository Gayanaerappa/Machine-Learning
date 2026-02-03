import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 1. Prototype Dataset (List of Lists)
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# 2. One-Hot Encoding
# The Apriori algorithm requires a Boolean DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 3. Generate Frequent Itemsets
# We set a minimum support of 60%
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# 4. Generate Association Rules
# We use 'lift' as the metric to find strong relationships
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Display the results
print("--- Frequent Itemsets ---")
print(frequent_itemsets)
print("\n--- Association Rules ---")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])