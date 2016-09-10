#include "helper.h"

//https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
//Proper histogram equalization for colour images
Mat equalizeIntensity(const Mat& inputImage)
{
    if (inputImage.channels() >= 3)	{
        Mat ycrcb;
        cvtColor(inputImage, ycrcb, CV_BGR2YCrCb);
        vector<Mat> channels;
        split(ycrcb, channels);

        equalizeHist(channels[0], channels[0]);
        Mat result;
        merge(channels, ycrcb);
        cvtColor(ycrcb, result, CV_YCrCb2BGR);

        return result;
    }
    return Mat();
}

//calcCovarMatrix expects single-dimension values
Mat castVec3btoMat(vector<Vec3b> input) {
    int in_size = input.size();
    Mat output(in_size, 3, CV_8UC1);
    for (int i=0; i<in_size; i++) {
        output.at<uchar>(i,0) = input.at(i)[0];
        output.at<uchar>(i,1) = input.at(i)[1];
        output.at<uchar>(i,2) = input.at(i)[2];
    }
    return output;
}

//https://github.com/opencv/opencv/blob/master/modules/ml/src/inner_functions.cpp
//Modified from above to produce lower triangular matrix S
void Cholesky(const Mat& A, Mat& S)
{
    CV_Assert(A.type() == CV_32F);

    S = A.clone();
    cv::Cholesky ((float*)S.ptr(),S.step, S.rows,NULL, 0, 0);
    for (int i=0;i<S.rows;i++)
        for (int j=i+1;j<S.rows;j++)
            S.at<float>(i,j)=0;
}


//For splitting the directory URL
//https://stackoverflow.com/questions/236129/split-a-string-in-c
vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}
