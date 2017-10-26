import csv

def ReadCsv(path,delimiter = ","):
    stream = open(path , 'r' )
    return csv.reader(stream, delimiter = delimiter)


if __name__ == "__main__":
    filepath = r"data\1-1_Refelctivity.csv"
    resr = ReadCsv(filepath)
    aa = list(resr)[300:600]

    for line in aa:
        print([ one.rjust(20) for one in line])

    b = aa[0]
    print(aa)
    print()




