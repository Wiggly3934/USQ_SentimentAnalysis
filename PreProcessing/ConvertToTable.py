import json
import csv

#==============================
def get_json_data(file_path):
    """
    Reads JSON data from the specified file and returns a list of user data.
    If the file is not found, it exits the program with an error message.
    """
    users_data = []
    try:
        with open(file_path, 'r') as json_file:
            for line in json_file:
                user_data = json.loads(line)
                users_data.append(user_data)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        exit(1)
    return users_data

#==============================
def prepare_csv_data(users_data):
    """
    Prepares data to be written into a CSV file.
    Extracts the first 40 posts for each user and formats the data for CSV output.
    """
    csv_data = [["Posts", "ID", "Label"]]  # Header for the CSV file
    
    for user_data in users_data:
        posts_text = ""  # Initialize a string for storing the posts text
        
        # Access and concatenate the first 40 posts for the user
        for post in user_data[0]["posts"][:40]:  
            if posts_text:
                posts_text += "\n"
            posts_text += post[1]  # Accessing the text of each post

        # Append the user data (posts, ID, label) to the CSV data
        csv_data.append([f'"{posts_text}"', user_data[0]["id"], user_data[0]["label"]])

    return csv_data

#==============================
def write_to_csv(csv_data, output_file_path):
    """
    Writes the CSV data to the specified file.
    """
    try:
        with open(output_file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_data)
        print(f"Data has been successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

#==============================
# Main program flow
def main():
    # Prompt the user for the JSON file path
    json_file_path = input("Enter the path to the JSON file: ").strip("'")

    # Read and process the JSON data
    users_data = get_json_data(json_file_path)

    # Prepare the data for the CSV file
    csv_data = prepare_csv_data(users_data)

    # Output CSV file path
    output_csv_path = "/Users/User/Documents/RSDD_zip/RSDD/CSV/testing_40.csv"

    # Write the data to the CSV file
    write_to_csv(csv_data, output_csv_path)

#==============================
# Run the main function
if __name__ == "__main__":
    main()
