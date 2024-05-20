import pandas as pd

def normalize_to_range(value, min_val, max_val, target_min=1, target_max=7):
    return ((value - min_val) / (max_val - min_val)) * (target_max - target_min) + target_min

# Read the Excel file
file_path = "./Dataset/essay_set_descriptions.xlsx"
df = pd.read_excel(file_path)

# Extract relevant columns
questions = df['question']
min_scores = df['min_domain1_score']
max_scores = df['max_domain1_score']

# Calculate global min and max for normalization
global_min = min(min_scores.min(), max_scores.min())
global_max = max(min_scores.max(), max_scores.max())

# Normalize the scores to range 1-7
df['min_domain1_score_normalized'] = min_scores.apply(normalize_to_range, args=(global_min, global_max))
df['max_domain1_score_normalized'] = max_scores.apply(normalize_to_range, args=(global_min, global_max))

# Round the normalized values
df['min_domain1_score_normalized'] = df['min_domain1_score_normalized'].round()
df['max_domain1_score_normalized'] = df['max_domain1_score_normalized'].round()

# Extract the relevant columns
result_df = df[['essay_set', 'question', 'max_domain1_score_normalized', 'min_domain1_score_normalized']]

# Save the result to a new Excel file
result_df.to_excel("./Dataset/normalized_scores.xlsx", index=False)
print("Normalization completed and results saved to 'normalized_scores.xlsx'")

# Load and process another CSV file
temp = pd.read_csv("./Dataset/processed-train-data.csv")
temp.drop("Unnamed: 0", inplace=True, axis=1)
temp.reset_index(drop=True, inplace=True)

# Merge result_df with temp on 'essay_set'
merged_df = pd.merge(temp, result_df, on='essay_set', how='left')

# Save the merged data to a new CSV file
merged_df.to_csv("./Dataset/merged-processed-train-data.csv", index=False)
print("Merged data saved to './Dataset/merged_processed_train_data.csv'")

# Load the main dataset
main_df = pd.read_csv('./Dataset/training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
main_df.dropna(axis=1, inplace=True)
main_df.drop(columns=['domain1_score', 'rater1_domain1', 'rater2_domain1'], inplace=True)

# Add the processed scores
main_df['domain1_score'] = temp['final_score']

# Normalize the scores to range 1-7
main_df['final_score'] = min_scores.apply(normalize_to_range, args=(global_min, global_max))


# Round the normalized values
main_df['final_score'] = main_df['final_score'].round()


# Save the updated main dataset to a new CSV file
main_df.to_csv("./Dataset/updated-training-set.csv", index=False)
print("Updated training set saved to './Dataset/training-set-for-genai.csv'")
