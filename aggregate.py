from os import listdir
import csv
import numpy as np
import sys

directory = (sys.argv[1]+"/") if len(sys.argv) > 1 else "baseline/"
dir_name = directory[:-1]

aggregate = open(dir_name+".csv", "w")
wrlist = ["Category", "Sequence", "File", "TP", "TN", "FP", "FN", "Rec", "Spec", "FPR", "FNR", "PWr", "F-Meas", "Prec", "ProcTime", "OVERALL", "N", "R", "DIS", "numMin", "phi", "POST or TargetSeg"]
aggregate.write(','.join(wrlist)+'\n')
aggregate.close()

aggregate = open(dir_name+".csv", "a")
agg_write = csv.writer(aggregate, delimiter=",")

results = listdir(directory)
for result_file in results:
    print result_file
    with open (directory+result_file, 'r') as csvfile:
        result_read = csv.reader(csvfile)
        overalls = []
        for row in result_read:
            if "OVERALL" in row:
                overalls.append(row)
                agg_write.writerow(row)
        if overalls:
            overalls = np.asarray(overalls)
            sums = np.sum(np.asarray(overalls[:,[3,4,5,6]], dtype=np.float64), axis=0)
            if (len(row) == 22):
                means = np.mean(np.asarray(overalls[:,[7,8,9,10,11,12,13,14]], dtype=np.float64), axis=0)
                agg_write.writerow((row[0],"OVERALL","OVERALL",sums[0],sums[1],sums[2],sums[3],means[0],means[1],means[2],means[3],means[4],means[5],means[6],means[7],str(2),row[16],row[17],row[18],row[19],row[20],row[21]))
            else:
                means = np.mean(np.asarray(overalls[:,[7,8,9,10,11,12,13]], dtype=np.float64), axis=0)
                agg_write.writerow((row[0],"OVERALL","OVERALL",sums[0],sums[1],sums[2],sums[3],means[0],means[1],means[2],means[3],means[4],means[5],means[6],"",str(2),row[15],row[16],row[17],row[18],row[19],row[20]))
    csvfile.close()

aggregate.close()
