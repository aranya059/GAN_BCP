# Generate a list of HDF5 filenames and save it to a text file

# List of HDF5 filenames
h5_filenames = [f"seed{str(i).zfill(4)}.h5" for i in range(1, 10000)]

# Write the filenames to a text file
output_file_path = "h5_filenames.txt"
with open(output_file_path, "w") as file:
    for filename in h5_filenames:
        file.write(f"{filename}\n")

print(f"File saved as {output_file_path}")
