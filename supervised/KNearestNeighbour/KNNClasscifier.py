import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_table('https://storage.googleapis.com/kagglesdsdata/datasets/9590/13660/fruit_data_with_colors.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240817%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240817T081626Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=095699c18047fb6ff40994e5ed82f47718babbeefcc3be97494a3443fc1a1610da4f7b45027d5ff4e51a3f275b92883849c02b88d387b4aa78ddcad697b6b73de0e337b9e2a7b699e111269ac066c8c91fb544cb02a0bc51220272ceb3b727080ca0ea63e1500eaa5ba20f8d3fb59bc726848831a1db72d3ec4fa192f4ff23ae19e8bf7abd8372b99963485e357b1aadb5e448c7bf35f43469f1a874fc28e5e99e22f3b9f9171a4a635bc02c0fe33f0150e24a409d1d48a85963abb3b8b7362a8e816a6c019919e3eb969256451ce5c46d77a496aceb1997a2e37aa8b74085f7ce8fcc99e5bf5bcaddedae876cb487605e002464ab1fc36146165fb1e6b7cf1e')

X = df[['mass','width','height']]
y = df['fruit_label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_train,y_train)

predict = model.predict(x_test)
accuarcy = accuracy_score(predict,y_test)


#zip label and name then change to dict type
lookup_fruit_name = dict(zip(df.fruit_label.unique(), df.fruit_name.unique()))


new_fruit = model.predict([[20, 3.4 , 5]])
print(f"Your fruit is : {lookup_fruit_name[new_fruit[0]]}")