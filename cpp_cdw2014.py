import cv2
import numpy as np
from os import listdir
import subprocess
from multiprocessing import Process, Lock, Pool
from random import randint, random
import time

def check_segmentation(input_a, groundtruth):
    height, width = input_a.shape

    tp, tn, fp, fn = 0, 0, 0, 0 #Initialise True Positive, False Positive, False Negative, True Negative
    groundtruth[groundtruth == 50] = 255
    for idx, i in np.ndenumerate(input_a):
        if groundtruth[idx] not in [85, 170]: #Outside of RoI or unknown motion
            if i == groundtruth[idx]: #Correct
                if i:
                    tp += 1 #True Positive
                else:
                    tn += 1 #True Negative
            else: #Wrong
                if i:
                    fp += 1 #False Positive
                else:
                    fn += 1 #False Negative

    return calc_metrics(tp, tn, fp, fn)

def calc_metrics(tp, tn, fp, fn):
    denom = float(tp + fp)
    precision = float(tp) / denom if denom else 1.0

    denom = float(tp + fn)
    recall = float(tp) / denom if denom else 1.0

    denom = float(tn + fp)
    specificity = float(tn) / denom if denom else 1.0
    fprate = float(fp) / denom if denom else 0
    denom = float(tp + fn)
    fnrate = float(fn) / denom if denom else 0
    total_num_pixels = float(fn + fp + tn + tp)
    pcwrong = (100 * float(fn + fp) / total_num_pixels) if total_num_pixels else 0
    denom = float(precision + recall)
    fmeas = float(2 * precision * recall) / denom if denom else 1.0
    return [tp, tn, fp, fn, recall, specificity, fprate, fnrate, pcwrong, fmeas, precision]


def run_test(id, root, dataset, folder, i_N=20, i_R=20, i_DIS=2, i_numMin=2, i_phi=4, post=1):
    seq_path = dataset+folder+"/"
    sequences = listdir(seq_path)
    sequences.sort()
    for sequence in sequences:
        test_dir = seq_path+sequence+"/"
        files = listdir(test_dir + "input/")
        files.sort()
        num_samples = len(files)

        rangefile = open(test_dir+"temporalROI.txt", "r")
        test_range = rangefile.read()
        test_range = [int(n) for n in test_range.split(' ')];
        rangefile.close()

        lock.acquire()
        print "["+time.strftime("%Y-%m-%d %H:%M:%S")+"]"
        print "NOW TESTING " + folder + ": " + sequence + " ID: " + str(id) + "                       "
        print "Parameters: N="+str(i_N)+" R="+str(i_R)+" DIS="+str(i_DIS)+" numMin="+str(i_numMin)+" phi="+str(i_phi)
        lock.release()

        #result_dir = root+"results/"+folder+"/"+sequence+"/"
        result_dir = "resimg/" + str(id) + sequence + "/"
        subprocess.call(["mkdir", result_dir])

        command = ["./superbe",test_dir+"input/",result_dir,str(i_N),str(i_R),str(i_DIS),str(i_numMin),str(i_phi), str(post)]
        subprocess.call(command)

        lock.acquire()
        print "ID: " + str(id) + " PROCESSING COMPLETE, NOW ANALYSING"
        lock.release()

        seq_scores = []
        for idx, file_in in enumerate(files[0::1]):
            if idx >= test_range[0] and idx <= test_range[1]:
            #if True:

                test_img = cv2.imread(result_dir+"b"+file_in)
                try:
                    valid = True if test_img else False
                except ValueError:
                    valid = True

                if valid: #Otherwise there is no test image found
                    groundtruth = cv2.imread(test_dir+"groundtruth/"+"gt"+file_in[2:-3]+"png")
                    if len(test_img.shape) > 2:
                        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                    if len(groundtruth.shape) > 2:
                        groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)

                    score = check_segmentation(test_img, groundtruth)
                    seq_scores.append(score)

                    results = open("cdwresults/"+folder+str(id)+".csv", "a")
                    wrlist = [folder,sequence,file_in,str(score[0]),str(score[1]),str(score[2]),
                        str(score[3]),str(score[4]),str(score[5]),str(score[6]),str(score[7]),
                        str(score[8]),str(score[9]),str(score[10]), str(0),
                        str(i_N), str(i_R), str(i_DIS), str(i_numMin), str(i_phi), str(post)]
                    results.write(','.join(wrlist)+"\n")
                    results.close()

        sums = np.sum(seq_scores, axis=0)
        avgs = np.mean(seq_scores, axis=0)
        results = open("cdwresults/"+folder+str(id)+".csv", "a")
        try:
            wrlist = [folder,sequence,"OVERALL",str(sums[0]),str(sums[1]),str(sums[2]),
                str(sums[3]),str(avgs[4]),str(avgs[5]),str(avgs[6]),str(avgs[7]),
                str(avgs[8]),str(avgs[9]),str(avgs[10]), str(1),
                str(i_N), str(i_R), str(i_DIS), str(i_numMin), str(i_phi), str(post)]
            results.write(','.join(wrlist)+"\n")
        except IndexError: #There are no results to write?!
            pass
        results.close()

        lock.acquire()
        print "["+time.strftime("%Y-%m-%d %H:%M:%S")+"]"
        print folder + ": " + sequence + " ID: " + str(id) + " TEST COMPLETE                        "
        subprocess.call(["rm", "-rf", result_dir])
        lock.release()

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
    for N in xrange(10, 41, 10):
        for R in xrange(20, 201, 40):
            for DIS in np.arange(4.0,20.01,2.0):
                for numMin in xrange(2, 4, 1):
                    for phi in xrange(8, 17, 8):
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
