import pandas as pd

#this is just Example of how to input Cross-Tabulation.
labels = []
varieties = []
#labels is  predicted values from model and varieties giving the grain variety for each sample.
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)