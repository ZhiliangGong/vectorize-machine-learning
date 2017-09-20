import csv

def read_csv_matrix(filename):
    with open(filename) as csvFile:
        data = csv.reader(csvFile, delimiter = ',')
        matrix = []
        for row in data:
            matrix.append(row)
        return matrix

def find_indices(the_string, the_character):
    return [ i for i, letter in enumerate(the_string) if letter == the_character ]

if __name__ == "__main__":
    import sys
    print(read_csv_matrix(sys.argv[1]))
