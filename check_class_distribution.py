from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("your_data_path/prepared_dataset")

# Count class occurrences in train and test datasets
train_counts = dataset["train"].to_pandas()["label"].value_counts()
test_counts = dataset["test"].to_pandas()["label"].value_counts()

print("Train Dataset Class Distribution:")
print(train_counts)

print("\nTest Dataset Class Distribution:")
print(test_counts)

# Train Dataset Class Distribution:
# label
# 1    10596
# 4    10584
# 5    10560
# 2    10554
# 0    10551
# 3    10545
# 6    10535
# Name: count, dtype: int64

# Test Dataset Class Distribution:
# label
# 6    2666
# 3    2656
# 0    2650
# 2    2647
# 5    2641
# 4    2617
# 1    2605
# Name: count, dtype: int64