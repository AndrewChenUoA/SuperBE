#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <set>
#include <iostream>
#include <stdlib.h>
#include "helper.h"

#define EPSILON 1e-100

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

class superbe_engine {
private:
    int N, R, DIS, numMin, phi, post;
    int numSegments, height, width;

    int frameNumber;

    Mat image, segmented;
    Mat segments, edges;

    vector<vector<Point> > segment_pixels;
    vector<vector<Vec3b> > segment_pixvals;
    vector<vector<int> > neighbours;
    vector<Mat> avgs;
    vector<Mat> devs;

    //Background Model
    vector<vector<Mat> > bgavgs; //Easier later to just keep these two separate
    vector<vector<Mat> > bgdevs;

    Mat structOpen, structClose; //Structuring elements for morphological operations

    //Memory placeholder for random numbers
    int randint, rand_neigh, rand_bgmodel;

    Mat filter_equalise();
    void process_vals(Mat);

public:
    void set_init(int, int, double, int, int, int);
    void initialise_background(String);
    void initialise_background(Mat);
    Mat process_frame(String, int);
    Mat process_frame(Mat, int);
};
