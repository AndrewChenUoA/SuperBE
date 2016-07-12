#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

vector<double> check_segmentation(Mat input_a, Mat groundtruth);
vector<double> calc_metrics(long tp, long tn, long fp, long fn);
