import os
import json

# --- 1. CONFIGURE YOUR PATHS HERE ---

# Path to your template Jupyter Notebook
template_notebook_path = '../Template.ipynb'

# Directory where your CSV data files are stored
data_directory = '../data/InventWood/'

# Directory where the new notebooks will be saved
output_directory = '../notebooks/WoodBooks/'


# --- 2. THE SCRIPT LOGIC ---

def create_notebooks():
    """
    Generates a new Jupyter notebook for each CSV file in the data directory
    based on a template notebook.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Load the content of the template notebook
    try:
        with open(template_notebook_path, 'r', encoding='utf-8') as f:
            template_content = json.load(f)
    except FileNotFoundError:
        print(f"Error: Template notebook not found at '{template_notebook_path}'")
        return

    # Find all CSV files in the data directory
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{data_directory}'")
        return

    print(f"Found {len(csv_files)} CSV files. Generating notebooks...")

    # Loop through each CSV file and create a new notebook
    for data_file in csv_files:
        # Create a deep copy of the template content for modification
        new_notebook_content = json.loads(json.dumps(template_content))

        # Modify the code cell containing the filename
        for cell in new_notebook_content['cells']:
            if cell['cell_type'] == 'code':
                # Check each line in the code cell
                for i, line in enumerate(cell['source']):
                    # Use .strip() and .startswith() to find the line robustly
                    if line.strip().startswith('File ='):
                        # Replace the line with the new filename
                        cell['source'][i] = f'File = "{data_file}"\n'
                        break  # Move to the next cell once the line is found

        # Define the new notebook's name and path
        new_notebook_name = f"{os.path.splitext(data_file)[0]}.ipynb"
        new_notebook_path = os.path.join(output_directory, new_notebook_name)

        # Write the modified content to the new notebook file
        with open(new_notebook_path, 'w', encoding='utf-8') as f:
            json.dump(new_notebook_content, f, indent=2)

        print(f"  -> Created '{new_notebook_path}'")

    print("\nNotebook generation complete!")


# Run the script
if __name__ == '__main__':
    create_notebooks()