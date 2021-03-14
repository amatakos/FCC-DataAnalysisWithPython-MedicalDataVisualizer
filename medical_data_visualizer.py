import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df.weight / (df.height / 100) ** 2
df['overweight'] = [0 if x <= 25 else 1 for x in bmi]

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
def normalize(x):
  if x == 1:
    return 0
  elif x > 1 :
    return 1

df.gluc = [normalize(gluc) for gluc in df.gluc]
df.cholesterol = [normalize(gluc) for gluc in df.cholesterol]

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x='variable', hue='value', data=df_cat, col='cardio', kind='count')

    g.set(ylabel='total')
    fig = g.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    cleaner = (df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))
    df_heat = df[cleaner]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    for i in range(corr.shape[0]):
      for j in range(i+1):
        mask[j][i] = True



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(14,14))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, center=0, vmin=-0.16, vmax=0.32, annot=True, fmt='.1f')


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
