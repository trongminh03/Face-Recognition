import os

# Define the directory containing the files to be renamed
path = 'dataset/Person Name'
# folder_path = 'Le_Trong_Minh/'

# Define the base name for the renamed files
# new_name = 'mindy_kaling_0000'

# Initialize a counter to keep track of the index
# counter = 1
new_name = '_'.join(path.split())
new_name = path + '_0000'
print("new_name", new_name)
counter = 1
# Iterate over each file in the folder
for filename in os.listdir("Le Trong Minh/"): 
    tmp_new_name = new_name[:-len(str(counter))]
    # Build the new filename using the base name and the current index
    new_filename = tmp_new_name + str(counter) + '.jpg'
    print(new_filename)

    # Create the full path to the old and new files
    old_path = os.path.join(path, filename)
    new_path = os.path.join(path, new_filename)

    # Rename the file
    os.rename(old_path, new_path)

    # Increment the counter
    counter += 1