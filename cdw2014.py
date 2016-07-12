import cv2
import numpy as np
import sys
from superBE import superbe_engine
from os import listdir
from multiprocessing import Process, Lock, Pool
from random import randint, randrange
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


def run_test(id, root, dataset, folder, i_N=20, i_R=20, i_DIS=2, i_numMin=2, i_phi=4, post=0):
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
        print "Parameters: N="+str(i_N)+" R="+str(i_R)+" DIS="+str(i_DIS)+" numMin="+str(i_numMin)+" phi="+str(i_phi)+" post="+str(post)
        print sequence + " testing range: " + str(test_range) + "                        "
        lock.release()

        initial_bg = test_dir + "input/in000001.jpg"
        result_dir = root+"results/"+folder+"/"+sequence+"/"
        try:
            engine = superbe_engine(N=i_N, R=i_R, DIS=i_DIS, numMin=i_numMin, phi=i_phi)
            engine.initialise_background(initial_bg)

            seq_scores = []
            for idx, file_in in enumerate(files[0::1]):
                lock.acquire()
                sys.stdout.write("Test ID: " + str(id) + " " + sequence + " ------------ " + file_in + " /" + str(num_samples) + "            " + '\r')
                sys.stdout.flush()
                lock.release()

                [superbe_mask, superbe_closed] = engine.process_frame(test_dir+ "input/" + file_in, -1)
                superbe = superbe_mask if post == 0 else superbe_closed
                error = cv2.imwrite(result_dir+str(post)+"b"+file_in, superbe)
                error = 0

                if idx >= test_range[0] and idx <= test_range[1]:
                #if True:
                    groundtruth = cv2.imread(test_dir+"groundtruth/"+"gt"+file_in[2:-3]+"png")
                    score = check_segmentation(superbe, cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY))
                    seq_scores.append(score)

                    results = open("cdwresults/"+folder+str(id)+".csv", "a")
                    wrlist = [folder,sequence,file_in,str(score[0]),str(score[1]),str(score[2]),
                        str(score[3]),str(score[4]),str(score[5]),str(score[6]),str(score[7]),
                        str(score[8]),str(score[9]),str(score[10]), str(1 if error else 0),
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
            lock.release()

        except: #An error, probably the targetSeg is too high for the sequence
            lock.acquire()
            print folder + ": " + sequence + " ID: " + str(id) + " TEST ERROR                        "
            lock.release()

    lock.acquire()
    print "["+time.strftime("%Y-%m-%d %H:%M:%S")+"]"
    print "ALL SEQUENCES ID: " + str(id) + " COMPLETE!                          "
    lock.release()
    return 1 #This really doesn't matter the result gets thrown away anyway

def pool_init(l):
    global lock
    lock = l

def main():
    #Set up four pools
    lock = Lock()
    pool = Pool(initializer=pool_init, initargs=(lock,), processes=8)

    root = "../dataset2014/"
    dataset = root + "dataset/"
    categories = listdir(dataset)
    categories.sort()
    #categories = ["baseline"]

    results = []
    count = 0

    for i in [0, 1]:
        N = 20
        R = 80
        DIS = 2.0
        numMin = 1.8
        phi = 16
        for folder in categories:
            count += 1
            resfile = open("cdwresults/"+folder+str(count)+".csv", "w")
            wrlist = ["Category", "Sequence", "File", "TP", "TN", "FP", "FN", "Rec", "Spec", "FPR", "FNR", "PWr", "F-Meas", "Prec", "Mask Err", "N", "R", "DIS", "numMin", "phi", "post"]
            resfile.write(','.join(wrlist)+"\n")
            resfile.close()
            results.append(pool.apply_async(run_test, (count,root,dataset,folder,N,R,DIS,numMin,phi,i)))

    print "TOTAL NUMBER OF TESTS: " + str(count) + "                   "

    for res in results:
        try:
            res.get()
        except:
            pass

    pool.close()
    pool.join()
    print "TESTS COMPLETE                                              "

if __name__=='__main__':
    main()
