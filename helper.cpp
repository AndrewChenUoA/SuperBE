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

//Calculate Cholesky Decomposition/Factorisation
//https://github.com/Itseez/opencv/blob/master/modules/ml/src/inner_functions.cpp
void Cholesky( const Mat& A, Mat& S )
{
    CV_Assert(A.type() == CV_32F);

    int dim = A.rows;
    S.create(dim, dim, CV_32F);

    int i, j, k;

    for( i = 0; i < dim; i++ )
    {
        for( j = 0; j < i; j++ )
        S.at<float>(i,j) = 0.f;

        float sum = 0.f;
        for( k = 0; k < i; k++ )
        {
            float val = S.at<float>(k,i);
            sum += val*val;
        }

        S.at<float>(i,i) = std::sqrt(std::max(A.at<float>(i,i) - sum, 0.f));
        float ival = 1.f/S.at<float>(i, i);

        for( j = i + 1; j < dim; j++ )
        {
            sum = 0;
            for( k = 0; k < i; k++ )
            sum += S.at<float>(k, i) * S.at<float>(k, j);

            S.at<float>(i, j) = (A.at<float>(i, j) - sum)*ival;
        }
    }
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
