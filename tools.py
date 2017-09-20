import csv

def read_csv(filename):
    with open('data.csv') as csvFile:
        data = csv.reader(csvFile, delimiter = ',')
        matrix = []
        for row in data:
            matrix.append(row)
        print matrix
