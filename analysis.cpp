#include "analysis.h"

//Analysis functions ported from Python
vector<double> check_segmentation(Mat input_a, Mat groundtruth) {
	int test_height = input_a.rows;
	int test_width = input_a.cols;

  long tp = 0;
	long tn = 0;
	long fp = 0;
	long fn = 0;
	uchar test_val, gt_val;

	for(int i=0; i<test_height; i++) {
    int* pi = input_a.ptr<int>(i);
		int* gi = groundtruth.ptr<int>(i);
    for (int j=0; j<test_width; j++) {
			test_val = pi[j];
			gt_val = gi[j];
			if (gt_val == 50) gt_val = 255; //Turn shadows into "movement"

			if (gt_val != 85 && gt_val != 170) {
				if (test_val == gt_val) {
					if (test_val != 0) { //Correct
						tp += 1;
					} else {
						tn += 1;
					}
				} else { //Incorrect
					if (test_val != 0) {
						fp += 1;
					} else {
						fn += 1;
					}
				}
			}
    }
  }

  return calc_metrics(tp, tn, fp, fn);
}

vector<double> calc_metrics(long tp, long tn, long fp, long fn) {
	double precision, recall, specificity, fprate, fnrate, total_num_pixels, pcwrong, fmeas;
  double denom = (double)(tp + fp);
	if (denom != 0) {
		precision = (double)(tp) / denom;
	} else {
		precision = 1.0;
	}

  denom = (double)(tp + fn);
	if (denom != 0) {
  	recall = (double)(tp) / denom;
	} else {
		recall = 1.0;
	}

  denom = (double)(tn + fp);
	if (denom != 0) {
  	specificity = (double)(tn) / denom;
		fprate = (double)(fp) / denom;
	} else {
		specificity = 1.0;
		fprate = 0.0;
	}

  denom = (double)(tp + fn);
	if (denom != 0) {
  	fnrate = float(fn) / denom;
	} else {
		fnrate = 0.0;
	}

  total_num_pixels = (double)(fn + fp + tn + tp);
	if (total_num_pixels != 0) {
  	pcwrong = (100 * (double)(fn + fp) / total_num_pixels);
	} else {
		pcwrong = 0;
	}

  denom = (double)(precision + recall);
	if (denom != 0) {
  	fmeas = (double)(2 * precision * recall) / denom;
	} else {
		fmeas = 0.0;
	}

	double array[] = {tp, tn, fp, fn, recall, specificity, fprate, fnrate, pcwrong, fmeas, precision};
	vector<double> metrics (array, array+sizeof(array) / sizeof(array[0]));

	return metrics;
}
