import csv
from collections import defaultdict
import os

#==============================
def delete_non_control_depressed_samples(input_csv, output_csv, num_to_delete):
    """
    Deletes the specified number of samples that are neither 'control' nor 'depressed' from the input CSV
    and writes the remaining data to the output CSV. Returns the count of samples before deletion
    and the number of samples deleted.
    """
    samples_before_deletion = 0
    samples_deleted = 0

    # Open input CSV for reading and output CSV for writing
    with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header row to the output CSV
        header = next(reader)
        writer.writerow(header)

        # Iterate through the rows and delete those that are not 'control' or 'depressed'
        for row in reader:
            label = row[2]  # Assuming label is in the third column (index 2)

            if label in ['control', 'depression']:  # Keep these rows
                samples_before_deletion += 1
                writer.writerow(row)
            elif samples_deleted < num_to_delete:  # Delete non-'control'/'depressed' samples
                samples_deleted += 1

    return samples_before_deletion, samples_deleted

#==============================
def count_samples(input_csv):
    """
    Counts the total number of rows (samples) in the input CSV file, excluding the header row.
    """
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header row
        return sum(1 for _ in reader)  # Count remaining rows

#==============================
def main():
    """
    Main program logic: 
    - Prompts the user for input/output CSV paths,
    - Counts the total samples,
    - Deletes the specified number of non-'control'/'depressed' samples,
    - Displays statistics.
    """
    # Prompt user for the input and output CSV filenames
    input_csv = input("Enter the path to the input CSV file: ").strip("'")
    output_csv = input("Enter the path to the output CSV file: ").strip("'")

    # Get the count of samples in the input CSV
    samples_before = count_samples(input_csv)

    # Prompt the user for the number of samples to delete that are not 'control' or 'depressed'
    num_to_delete = int(input("Enter the number of samples to delete (not 'control' or 'depression'): "))

    # Call function to delete samples and get before/after counts
    samples_deleted, remaining_samples = delete_non_control_depressed_samples(input_csv, output_csv, num_to_delete)

    # Display results to the user
    print(f"\nSamples in '{input_csv}': {samples_before} samples")
    print(f"Samples not equal to 'control' or 'depression':")
    print(f"Deleted: {samples_deleted} samples")
    print(f"Remaining samples after deletion: {remaining_samples}")
    print(f"{num_to_delete} samples not equal to 'control' or 'depression' deleted from the CSV.")

#==============================
# Run the main function
if __name__ == "__main__":
    main()
