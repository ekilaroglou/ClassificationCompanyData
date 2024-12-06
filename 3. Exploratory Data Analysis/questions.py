# 3.6.1	What categories are the most employed companies?


# Filter the data where total_employee_estimate is greater than 5000
filtered_df = df[df['total_employee_estimate'] > 5000]

# Get the value counts for categories in the filtered data
category_counts = filtered_df['category'].value_counts() 

ax = category_counts.plot(kind='bar')
                        
ax.set_xlabel('Category')
ax.set_ylabel('Count')


# 3.6.2 What’s the distribution of companies founded over the years?

plt.figure(figsize=(12, 6))
# Get the count of companies founded in each year
year_counts = df['year_founded'].value_counts().sort_index()

# Create the bar plot
ax = plt.bar(year_counts.index, year_counts.values, color='lightgreen')

# Get years divisible by 10 in the range of years present in the dataset
years_divisible_by_10 = [year for year in year_counts.index if year % 10 == 0]

# Get min and max value
max_year = df['year_founded'].max()
min_year = df['year_founded'].min()

# Append everything in a list
x_values = years_divisible_by_10
x_values.append(max_year)
x_values.append(min_year)

# Set the x-ticks to the years divisible by 10
plt.xticks(x_values, rotation=45)

# Labels and title
plt.title('Distribution of Companies Founded Over the Years')
plt.xlabel('Year Founded')
plt.ylabel('Number of Companies')

# Make the layout tighter to prevent label clipping
plt.tight_layout()

# Show the plot
plt.show()

# 3.6.3	How have the distributions of different company categories changed over time?

# For clarity reason choose only year_founded after 1950
filtered_df = df[df['year_founded'] > 1950]

# Group by 'year_founded' and 'category' to get the counts of companies in each category for each year
category_counts = filtered_df.groupby(['year_founded', 'category']).size().reset_index(name='count')

# Pivot the data to make categories the columns, and years as the index
category_pivot = category_counts.pivot_table(index='year_founded', columns='category', values='count', aggfunc='sum', fill_value=0)

# Use a larger and more diverse color palette
colors = sns.color_palette("tab20c", n_colors=len(category_pivot.columns))  # Adjust n_colors to the number of categories

# Plotting the category distribution over the years
plt.figure(figsize=(14, 8))

category_pivot.plot(kind='line', marker='o', figsize=(14, 8), color=colors)

plt.title('Category Distribution Over the Years')
plt.xlabel('Year Founded')
plt.ylabel('Number of Companies')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 3.6.4	What’s the top word in Information Technology category?
filtered_df = df[df['category'] == 'Information Technology']

generate_word_cloud(filtered_df['homepage_text'])
generate_word_cloud(filtered_df['homepage_keywords'])
generate_word_cloud(filtered_df['meta_description'])
