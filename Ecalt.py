dataset = {
    "T1": {"Milk", "Bread", "Butter"},
    "T2": {"Bread", "Butter"},
    "T3": {"Milk", "Bread"},
    "T4": {"Milk", "Butter"}
}

vertical = {}

for tid, items in dataset.items():
    for item in items:
        vertical.setdefault(item, set()).add(tid)

min_support = 2

items = list(vertical.keys())

for i in range(len(items)):
    for j in range(i + 1, len(items)):
        itemset = {items[i], items[j]}
        common_tids = vertical[items[i]] & vertical[items[j]]

        if len(common_tids) >= min_support:
            print(itemset, "Support:", len(common_tids))
