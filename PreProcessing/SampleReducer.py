import csv
import os  # For path operations
from collections import defaultdict

#==============================
def delete_samples_with_label(input_csv, output_csv, label_to_delete, num_to_delete):
    """
    Deletes the specified number of samples with a particular label from the input CSV 
    and writes the result to the output CSV. Returns the count of samples before deletion 
    and the number of samples deleted.
    """
    samples_before_deletion = 0
    samples_deleted = 0

    # Open the input CSV for reading and the output CSV for writing
    with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header row to the output CSV
        header = next(reader)
        writer.writerow(header)

        # Iterate through the rows in the input CSV
        for row in reader:
            label = row[2]  # Assuming label is in the third column (index 2)
            
            # Delete rows that match the specified label
            if label == label_to_delete and samples_deleted < num_to_delete:
                samples_deleted += 1
            else:
                samples_before_deletion += 1
                writer.writerow(row)

    return samples_before_deletion, samples_deleted

#==============================
def count_samples_by_label(input_csv):
    """
    Counts and returns the number of samples for each unique label in the input CSV.
    """
    label_counts = defaultdict(int)

    # Open the input CSV for reading
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header row

        # Count the occurrences of each label
        for row in reader:
            label = row[2]  # Assuming label is in the third column (index 2)
            label_counts[label] += 1

    return label_counts

#==============================
def main():
    """
    Main program logic: 
    - Prompts the user for input/output CSV paths,
    - Displays label counts,
    - Deletes samples with the specified label,
    - Displays statistics.
    """
    # Prompt user for the input and output CSV filenames
    input_csv = input("Enter the path to the input CSV file: ").strip("'")
    output_csv = input("Enter the path to the output CSV file: ").strip("'")

    # Get and display the count of samples in the input CSV
    label_counts = count_samples_by_label(input_csv)
    for label, count in label_counts.items():
        print(f"Label '{label}': {count} samples")

    # Prompt the user for the label to delete and the number of samples to delete
    label_to_delete = input("Enter the label you want to delete samples for: ").strip("'")
    num_to_delete = int(input("Enter the number of samples to delete for the label: "))

    # Call function to delete samples and get before/after counts
    samples_before, samples_deleted = delete_samples_with_label(input_csv, output_csv, label_to_delete, num_to_delete)

    # Display results to the user
    print(f"\nSamples before deletion: {samples_before + samples_deleted}")
    print(f"Samples with label '{label_to_delete}' deleted: {samples_deleted}")
    print(f"Remaining samples after deletion: {samples_before}")
    print(f"{num_to_delete} samples with label '{label_to_delete}' have been deleted from the CSV.")

#==============================
# Run the main function
if __name__ == "__main__":
    main()
