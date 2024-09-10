import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re 

# Load the data
results_df = pd.read_csv('soma_neurite_separate_data.csv')

# Helper functions for cleaning and categorizing
def categorize_p_value(name):
    match = re.search(r'p(\d+)', name.lower())
    if match:
        return f'p{match.group(1)}'
    return 'flat'

def categorize_image_group(letter):
    if letter in 'abc':
        return 'Thin'
    elif letter in 'def':
        return 'Mushroom'
    elif letter in 'ghi':
        return 'Stubby'
    elif letter == 'flat':  # Assuming 'flat' is an acceptable letter
        return 'Flat'
    return None

# Apply functions
results_df['P Group'] = results_df['Key Image Name'].apply(categorize_p_value)
results_df['Image Group'] = results_df['Key Image Name'].apply(lambda name: name[0])#.lower())
results_df['Image Group'] = results_df['Image Group'].apply(categorize_image_group)

# Filter out rows where Image Group is None
results_df = results_df[results_df['Image Group'].notna()]

# Define category order for P Group and set as categorical
p_group_order = ['p4', 'p10', 'p30', 'flat']
results_df['P Group'] = pd.Categorical(results_df['P Group'], categories=p_group_order, ordered=True)

# Extract flat data and replicate it for each image group
flat_data = results_df[results_df['P Group'] == 'flat'].copy()
for group in ['Thin', 'Mushroom', 'Stubby']:
    replicated_flat = flat_data.copy()
    replicated_flat['Image Group'] = group
    results_df = pd.concat([results_df, replicated_flat])

# Group and calculate mean fluorescence intensity for each channel
grouped = results_df.groupby(['Image Group', 'P Group', 'Channel']).agg({
    'Normalized Fluorescence Intensity COMPOUND': 'mean'
}).unstack('Channel')

# Define color palettes
paxillin_palette = sns.color_palette("Reds")
integrin_palette = sns.color_palette("Blues")

# Plotting with specific color palettes
for channel in [0, 1, 2]:  # Assuming three channels as before
    plt.figure(figsize=(8, 5))
    heatmap_data = grouped['Normalized Fluorescence Intensity COMPOUND', channel].unstack().fillna(0)
    palette = paxillin_palette if channel == 0 else integrin_palette if channel == 1 else sns.color_palette("Greens")
    sns.heatmap(heatmap_data, annot=False, cmap=palette, fmt=".4f",  linewidths=8)
    plt.title(f'Mean Fluorescence {"Paxillin" if channel == 0 else "Integrin" if channel == 1 else "Other"} Across Different Substrate Conditions')
    plt.ylabel('Fluorescence Intensity')
    plt.xlabel('Pitch')
    plt.savefig(f'figure_{channel}.pdf')

    plt.show()