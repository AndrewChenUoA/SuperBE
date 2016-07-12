import cv2
import numpy as np
from random import randint
from skimage.segmentation import slic
from scipy.spatial.distance import euclidean
import warnings
#from math import pow

"""SuperBE algorithm, uses superpixels for simple background estimation"""

class superbe_engine():
    def __init__(self, N=20, R=40, DIS=10, numMin=2, phi=4):
        #Algorithm parameters
        self.N = N #Number of historical samples per superpixel
        self.R = R #Maximum distance in Euclidian colour space
        self.DIS = DIS #Maximum dissimilarity score
        self.numMin = numMin #Number of close samples for being part of the bg
        self.phi = phi #Amount of random subsampling when updating neighbours
        self.SPSIZE = 100 #Average number of pixels in each superpixel

        #Initial value
        self.frameNumber = 1

        #Dynamically calculated parameters
        self.numSegments = 0 #Actual number of segments in a particular frame
        self.height = 0 #Height of image
        self.width = 0 #Width of image
        self.structA = 0 #Kernel for morphological closing
        self.structB = 0 #Kernel for morphological opening

        warnings.filterwarnings("ignore")

    def get_img(self, img):
        if img == None:
            return False
        try:
            raw_img = cv2.imread(img)
        except TypeError:
            raw_img = img #Probably the raw data being passed in

        return raw_img

    def filter_equalise(self, raw_img):
        #filt_img = cv2.bilateralFilter(raw_img,5,150,150) #Filtering for noise reduction
        filt_img = cv2.GaussianBlur(raw_img,(15,15),0)

        #Do colour equalisation correctly by flattening the lightness channel
        filt_img = cv2.cvtColor(filt_img, cv2.COLOR_BGR2YCrCb)
        filt_img[:,:,0] = cv2.equalizeHist(filt_img[:,:,0])
        filt_img = cv2.cvtColor(filt_img, cv2.COLOR_YCrCb2BGR)
        #for colour in xrange(0,3): #Equalise each channel separately to flatten image overall
        #    filt_img[:,:,colour] = cv2.equalizeHist(filt_img[:,:,colour])

        return filt_img

    def draw_segments(self, raw_img):
        segmented = raw_img.copy()
        segmented[self.edges == 255] = (0, 0, 255)
        return segmented

    def process_model(self, filt_img):
        #Build a list where each element is a list of pixel values for that segment
        #Build a neighbour list, just looking at the edges (rather than iterating across entire image)
        segment_pixels = [[] for i in range(self.numSegments)]
        neighbours = [[] for i in range(self.numSegments)]
        for idx, i in np.ndenumerate(self.segments):
            segment_pixels[i].append(filt_img[idx])
            if self.edges.item(idx) == 255:
                if idx[0] != 0:
                    neighbours[i].append(self.segments[idx[0]-1,idx[1]])
                if idx[1] != 0:
                    neighbours[i].append(self.segments[idx[0],idx[1]-1])
                if idx[0] != self.height-1:
                    neighbours[i].append(self.segments[idx[0]+1,idx[1]])
                if idx[1] != self.width-1:
                    neighbours[i].append(self.segments[idx[0],idx[1]+1])

        #Remove duplicate neighbours
        for idx, i in np.ndenumerate(neighbours):
            x = set(i)
            neighbours[idx[0]] = list(x)
            #It's okay for the seg itself to be a "neighbour" because it
            #can be used for initialisation and also bg updates
            #try:
                #neighbours[idx[0]].remove(idx)
            #except ValueError:
            #    pass #It wasn't in the list so we don't need to remove it

        avgs = [None] * self.numSegments
        covar = [None] * self.numSegments
        for seg in xrange(0, self.numSegments):
            if segment_pixels[seg]:
                avgs[seg] = np.asarray(np.mean(segment_pixels[seg], axis = 0), dtype=np.uint8)
                covar[seg] = np.asarray(np.cov(segment_pixels[seg], rowvar=False)) #Using default 1/(N-1) normalisation
            else: #segment_pixels[i] was empty for whatever reason, no elements in that superpixel?
                try:
                    avgs[seg] = self.bgmodel[seg][self.N-1][0]
                    covar[seg] = self.bgmodel[seg][self.N-1][1:]
                except AttributeError: #No background model exists yet?
                    pass

        return neighbours, avgs, covar

    #@profile
    def pull_img(self, img):
        raw_img = self.get_img(img)

        #For HDA+ we should crop off the time at the top of the image
        #raw_img = raw_img[20:,:]

        filt_img = self.filter_equalise(raw_img)
        segmented = self.draw_segments(raw_img)
        neighbours, avgs, covar = self.process_model(filt_img)

        return raw_img, segmented, neighbours, avgs, covar

    def initialise_background(self, img):

        raw_img = self.get_img(img)

        self.height, self.width, cols = raw_img.shape
        self.targetSegments = (self.height * self.width) / self.SPSIZE

        filt_img = self.filter_equalise(raw_img)

        #Convert to CIE LAB, better approximation of human vision
        #filt_img = cv2.cvtColor(filt_img, cv2.COLOR_BGR2LAB)

        #Process SLIC
        self.segments = slic(filt_img, n_segments = self.targetSegments, sigma = 0)
        self.numSegments = np.amax(self.segments)+1 #Get the maximum value +1 since starting at 0
        #Find the edges of the segments and draw them to help with visualising the segments
        edge_segments = self.segments.astype(np.uint8)
        self.edges = cv2.Canny(edge_segments,0,1)

        neighbours, averages, covars = self.process_model(filt_img)

        self.bgmodel = np.empty((self.numSegments, self.N, 4, 3), np.float32);

        for seg in xrange(0, self.numSegments):
            num_neighbours = len(neighbours[seg])
            for N in xrange(0, self.N):
                #Initialise with random neighbour (including self) values
                if len(neighbours[seg]) != 0:
                    rand = randint(0, num_neighbours-1)
                    self.bgmodel[seg][N][0] = averages[neighbours[seg][rand]]
                    self.bgmodel[seg][N][1:] = covars[neighbours[seg][rand]]
                else: #no neighbours found, just fill with self-values
                    self.bgmodel[seg][N][0] = averages[seg]
                    self.bgmodel[seg][N][1:] = covars[seg]

        #Compute kernel sizes based on (average) superpixel size
        self.openSize = int(np.sqrt((self.height * self.width) / self.numSegments) * 2)
        self.closeSize = self.openSize * 3

        self.structA = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.closeSize,self.closeSize))
        self.structB = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.openSize,self.openSize))

        #print "Engine initialised"

    #@profile
    def process_frame(self, img, waitTime=250):

        raw_img, segmented, neighbours, averages, covars = self.pull_img(img)

        mask = np.zeros(raw_img.shape[:2], dtype = "uint8")

        #Iterate through each superpixel
        for seg in xrange(0,self.numSegments):

            #1) Compare pixel to background model
            count, index, dist = 0, 0, 0

            while count < self.numMin and index < self.N:
                #euc_dist = round(np.sqrt(pow(averages[seg][0].astype(np.int16)-self.bgmodel[seg][index][0][0].astype(np.int16),2)+pow(averages[seg][1].astype(np.int16)-self.bgmodel[seg][index][0][1].astype(np.int16),2)+pow(averages[seg][2].astype(np.int16)-self.bgmodel[seg][index][0][2].astype(np.int16),2)),0)
                euc_dist = round(euclidean(averages[seg].astype(np.int16), self.bgmodel[seg][index][0].astype(np.int16)), 0)

                A = np.matrix(covars[seg])
                B = np.matrix(self.bgmodel[seg][index][1:])

                try:
                    X = np.linalg.cholesky(A)
                    Y = np.linalg.cholesky(B)
                    Z = np.linalg.cholesky((A+B)/2)
                    dissimilarity = 2*np.log(np.linalg.det(Z)) - (np.log(np.linalg.det(X)) - np.log(np.linalg.det(Y)))
                except np.linalg.LinAlgError:
                    #print A
                    #print B
                    dissimilarity = np.inf
                #print dissimilarity

                #try:
                    #Jensen-Bregman LogDet Divergence [Cherian et al, 2011] [Eq (7)] https://lear.inrialpes.fr/people/cherian/papers/metricICCV.pdf
                #    dissimilarity = np.log(np.linalg.det((A+B)*0.5)) - 0.5*np.log(np.linalg.det(A*B))
                #except:
                #    dissimilarity = np.inf

                #If the colour covariance and the mean (LAB) colours are similar enough, it counts towards the background
                if (dissimilarity < self.DIS and euc_dist < self.R) or np.isnan(dissimilarity) or np.isinf(dissimilarity):
                    count += 1
                index += 1

            #2) Classify pixel and update model
            if count >= self.numMin:
                #3) Update current pixel model
                #Update a random past value
                #Always updating causes ghosts!
                rand = randint(0, self.N-1)
                self.bgmodel[seg][rand][0] = averages[seg]
                self.bgmodel[seg][rand][1:] = covars[seg]

                #4) Update neighbouring superpixel model(s)
                update = randint(0, self.phi-1)
                if update == 0:
                    num_neighbours = len(neighbours[seg])
                    if len(neighbours[seg]):
                        rand_neigh = randint(0, num_neighbours-1)
                        rand_bgmodel = randint(0, self.N-1)
                        self.bgmodel[rand_neigh][rand_bgmodel][0] = averages[seg]
                        self.bgmodel[rand_neigh][rand_bgmodel][1:] = covars[seg]
                    else:
                        pass #no neighbours, so don't update
            else:
                mask[self.segments == seg] = 255

        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.structA)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.structB)
        #closed = cv2.dilate(closed, np.ones((self.openSize,self.openSize),np.uint8), iterations=1)

        masked_img = cv2.bitwise_and(raw_img, raw_img, mask=closed)

        output = np.zeros((self.height*2, self.width*2, 3), np.uint8)

        if waitTime >= 0:
            output[:self.height, :self.width] = segmented
            output[:self.height, self.width:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            output[self.height:, :self.width] = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
            output[self.height:, self.width:] = masked_img

            cv2.imshow("output", output)
            cv2.waitKey(waitTime)

        self.frameNumber += 1
        return [mask, closed]

def main():
    engine = superbe_engine()

    from os import listdir
    #directory = "../HDA+/camera18/"
    directory = "../dataset2014/dataset/baseline/highway/input/"
    #directory = "../dataset2014/dataset/lowFramerate/turnpike_0_5fps/input/"
    files = listdir(directory)
    files.sort()

    #initial_bg = "../HDA+/camera18/I00000.jpg"
    initial_bg = directory + "in000001.jpg"

    engine.initialise_background(initial_bg)

    for file_in in files[0::1]:
        print "------------------------------" + file_in
        engine.process_frame(directory+file_in, 50)

    print "PROCESSING FINISHED"

if __name__ == '__main__':
    main()
