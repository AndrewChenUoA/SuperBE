#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace cv;
using namespace std;

//Function declarations
Mat equalizeIntensity(const Mat& inputImage);
Mat castVec3btoMat(vector<Vec3b> input);
void Cholesky(const Mat& A, Mat& S);
vector<string> &split(const string &s, char delim, vector<string> &elems);
vector<string> split(const string &s, char delim);
