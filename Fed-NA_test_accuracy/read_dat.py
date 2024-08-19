
file_path = r''

with open(file_path, 'r') as file:
    lines = file.readlines()

data_array = [float(line.strip()) for line in lines]

print(data_array)

