import json
import csv

# Prompt the user for the JSON file path
json_file_path = input("Enter the path to the JSON file: ").strip("'")

# Initialize a list to store the user data
users_data = []

# Read the JSON data from the specified file
try:
    with open(json_file_path, 'r') as json_file:
        for line in json_file:
            user_data = json.loads(line)
            users_data.append(user_data)
except FileNotFoundError:
    print(f"File not found: {json_file_path}")
    exit(1)

# Prepare the data for the CSV file
csv_data = [["Posts", "ID", "Label"]]

for user_data in users_data:
    # Initialize a string to store the first 40 posts for each user
    posts_text = ""
    for post in user_data[0]["posts"][:40]:  # Access the first X posts for the user
        if len(posts_text) > 0:
            posts_text += "\n"
        posts_text += post[1]  # Access the text in the nested list
    csv_data.append(['"' + posts_text + '"', user_data[0]["id"], user_data[0]["label"]])

# Write the data to a CSV file
with open("/Users/User/Documents/RSDD_zip/RSDD/CSV/testing_40.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)

print("Data has been converted and saved to /Users/User/Documents/RSDD_zip/RSDD/CSV/testing_40.")

