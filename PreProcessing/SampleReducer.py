import csv
import os  # Import the 'os' module for path operations
from collections import defaultdict

# Function to delete N samples associated with a specified label
def delete_samples_with_label(input_csv, output_csv, label_to_delete, num_to_delete):
    with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header row
        header = next(reader)
        writer.writerow(header)
        
        # Initialize counts for the number of samples before and after deletion
        samples_before_deletion = 0
        samples_deleted = 0
        
        # Iterate through the rows and delete the specified number of rows with the label
        for row in reader:
            label = row[2]  # Assuming label is in the third column (index 2)
            
            if label == label_to_delete and samples_deleted < num_to_delete:
                samples_deleted += 1
            else:
                samples_before_deletion += 1
                writer.writerow(row)

    return samples_before_deletion, samples_deleted

# Function to count the samples for each unique label in the input CSV file
def count_samples_by_label(input_csv):
    label_counts = defaultdict(int)
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            label = row[2]  # Assuming label is in the third column (index 2)
            label_counts[label] += 1
    return label_counts

# Prompt the user for the input and output CSV filenames
input_csv = input("Enter the path to the input CSV file: ").strip("'")
output_csv = input("Enter the path to the output CSV file: ").strip("'")

# Get the count of samples in the input CSV
samples_before = count_samples_by_label(input_csv)

# Display the count of samples for each unique label
label_counts = count_samples_by_label(input_csv)
for label, count in label_counts.items():
    print(f"Label '{label}': {count} samples")

# Specify the label and the number of samples to delete for that label
label_to_delete = input("Enter the label you want to delete samples for: ").strip("'")
num_to_delete = int(input("Enter the number of samples to delete for the label: "))

# Call the function to delete N samples with the specified label and get counts
samples_deleted, remaining_samples = delete_samples_with_label(input_csv, output_csv, label_to_delete, num_to_delete)

print(f"Samples in '{input_csv}': {samples_before} samples")
print(f"Samples with label '{label_to_delete}':")
print(f"Deleted: {samples_deleted} samples")
print(f"After deletion: {remaining_samples} samples")
print(f"{num_to_delete} samples with label '{label_to_delete}' deleted from the CSV.")



