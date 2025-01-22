import csv
from collections import defaultdict
import os

# Function to delete N samples not equal to 'control' or 'depressed'
def delete_non_control_depressed_samples(input_csv, output_csv, num_to_delete):
    with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header row
        header = next(reader)
        writer.writerow(header)
        
        # Initialize counts for the number of samples before and after deletion
        samples_before_deletion = 0
        samples_deleted = 0
        
        # Iterate through the rows and delete the specified number of rows not equal to 'control' or 'depressed'
        for row in reader:
            label = row[2]  # Assuming label is in the third column (index 2)
            
            if label in ['control', 'depression']:
                samples_before_deletion += 1
                writer.writerow(row)
            elif samples_deleted < num_to_delete:
                samples_deleted += 1

    return samples_before_deletion, samples_deleted

# Function to count the samples in the input CSV file
def count_samples(input_csv):
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        return sum(1 for _ in reader)

# Prompt the user for the input and output CSV filenames
input_csv = input("Enter the path to the input CSV file: ").strip("'")
output_csv = input("Enter the path to the output CSV file: ").strip("'")

# Get the count of samples in the input CSV
samples_before = count_samples(input_csv)

# Specify the number of samples to delete that are not 'control' or 'depressed'
num_to_delete = int(input("Enter the number of samples to delete (not 'control' or 'depression'): "))

# Call the function to delete N samples not equal to 'control' or 'depressed' and get counts
samples_deleted, remaining_samples = delete_non_control_depressed_samples(input_csv, output_csv, num_to_delete)

print(f"Samples in '{input_csv}': {samples_before} samples")
print(f"Samples not equal to 'control' or 'depression':")
print(f"Deleted: {samples_deleted} samples")
print(f"After deletion: {remaining_samples} samples")
print(f"{num_to_delete} samples not equal to 'control' or 'depression' deleted from the CSV.")
