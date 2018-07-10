
import csv
import sys
import re

infile = sys.argv[1]
outdir = sys.argv[2]
column = int(sys.argv[3])

csv.field_size_limit(sys.maxsize)
try:
    lines = []
    with open(infile, 'r', encoding = 'utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        for line in csv_reader:
            if re.match('^[0-9]+$',line[column]): 
                filename = outdir+line[column]+'.txt'
                with open(filename,'a',encoding='utf-8') as out:
                    out.write('|'.join(line) + '\n')

except:
    lines = []
    csvfile = open(infile, 'r', encoding = 'utf-8')
    csv_reader = csv.reader(line.replace('\0','') for line in csvfile.readlines())
    for line in csv_reader:
        if re.match('^[0-9]+$',line[column]): 
            filename = outdir+line[column]+'.txt'
            with open(filename,'a',encoding='utf-8') as out:
                out.write('|'.join(line)+'\n')
