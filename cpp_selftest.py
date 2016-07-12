import numpy as np
from os import listdir
import subprocess
from random import randint, random
from multiprocessing import Process, Lock, Pool
import time

def run_test(id, root, dataset, folder, i_N=20, i_R=20, i_DIS=2, i_numMin=2, i_phi=4, post=1):
    seq_path = dataset+folder+"/"

    lock.acquire()
    print "["+time.strftime("%Y-%m-%d %H:%M:%S")+"]"
    print "NOW TESTING " + folder + " ID: " + str(id) + "                       "
    print "Parameters: N="+str(i_N)+" R="+str(i_R)+" DIS="+str(i_DIS)+" numMin="+str(i_numMin)+" phi="+str(i_phi)+" post="+str(post)
    lock.release()

    command = ["./superbe",seq_path,str(i_N),str(i_R),str(i_DIS),str(i_numMin),str(i_phi),str(post),str(id)]
    subprocess.call(command)

    lock.acquire()
    print "["+time.strftime("%Y-%m-%d %H:%M:%S")+"]"
    print "ALL SEQUENCES ID: " + str(id) + " COMPLETE!                          "
    lock.release()
    return 1 #This really doesn't matter the result gets thrown away anyway

def pool_init(l):
    lock = l

def main():
    #Set up eight pools
    global lock
    lock = Lock()
    pool = Pool(initializer=pool_init, initargs=(lock,), processes=8)

    root = "../dataset2014/"
    dataset = root + "dataset/"
    #categories = listdir(dataset)
    #categories.sort()
    categories = ["baseline"]

    subprocess.call(["rm", "-rf", "cdwresults"])
    subprocess.call(["mkdir", "cdwresults"])
    subprocess.call(["rm", "-rf", "resimg"])
    subprocess.call(["mkdir", "resimg"])

    results = []
    count = 0
    for N in xrange(20, 41, 10):
        for R in xrange(40, 80, 5):
            for DIS in np.arange(10.0,20.01,1.0):
                for numMin in xrange(2, 6, 1):
                    for phi in xrange(4, 17, 4):
                        for post in [0, 1]:
                            for folder in categories:
                                count += 1
                                resfile = open("cdwresults/"+folder+str(count)+".csv", "w")
                                wrlist = ["Category", "Sequence", "File", "TP", "TN", "FP", "FN", "Rec", "Spec", "FPR", "FNR", "PWr", "F-Meas", "Prec", "Mask Err", "N", "R", "DIS", "numMin", "phi", "post"]
                                resfile.write(','.join(wrlist)+"\n")
                                resfile.close()
                                results.append(pool.apply_async(run_test, (count,root,dataset,folder,N,R,DIS,numMin,phi, post)))
                                #run_test(count, root, dataset, folder, N, R, DIS, numMin, phi)

    print "TOTAL NUMBER OF TESTS: " + str(count) + "                   "

    for res in results:
        res.get()

    pool.close()
    pool.join()
    print "TESTS COMPLETE                                              "

if __name__=='__main__':
    main()
