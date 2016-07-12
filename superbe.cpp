#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <ctime>
#include <fstream>
#include <sstream>

//http://stackoverflow.com/a/5590404/3093549
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

#include "helper.h"
#include "analysis.h"
#include "superbe_core.h"

int main(int argc, char** argv) {
	if (argc < 2) {
		printf(("\nUsage: %s <read directory> <N> <R> <DIS> <numMin> <phi> <post> <ID>\n"), argv[0]);
		return -1;
	}

	superbe_engine engine;

  if (argc >= 8) {
    engine.set_init(atoi(argv[2]),atoi(argv[3]),atof(argv[4]),atoi(argv[5]),atoi(argv[6]), atoi(argv[7]));
  } else if (argc == 7) {
		engine.set_init(atoi(argv[2]),atoi(argv[3]),atof(argv[4]),atoi(argv[5]),atoi(argv[6]), 1);
	} else {
		engine.set_init(20, 60, 18.0, 4, 16, 1);
	}

	String sequences[] = {"highway", "office", "pedestrians", "PETS2006"};
	vector<String> filenames;
	String directory = argv[1];
	vector<string> dirs = split(directory, '/');

	String ID = (argc == 9) ? argv[8] : "0";
	String command;

	fstream fs;
	String resfile = "cdwresults/"+dirs.back()+ID+".csv";
	fs.open(resfile.c_str(), fstream::out);
	fs << "Category,Sequence,File,TP,TN,FP,FN,Rec,Spec,FPR,FNR,PWr,F-Meas,Prec,Overall,N,R,DIS,numMin,phi,post" << "\n";
	fs.close();

	//Initialise memory for tests, avoiding redefinition of large items
	Mat result, groundtruth;
	vector<double> metrics;
	vector<vector<double> > seq_scores;
	double sums[11];
	double avgs[11];
	int ignorewarning;

	for (int seq=0; seq<4; seq++) {
		String sequence = sequences[seq];
		String write_dir = "resimg/"+ID+sequence+"/";
		command = "mkdir " + write_dir;
		ignorewarning = system(command.c_str());
		glob(directory+"/"+sequence+"/input", filenames); //Pulls the filenames out of the directory

		//Check what the range for this sequence is
	  vector<int> range;
		String tempfile = directory+"/"+sequence+"/"+"temporalROI.txt";
		ifstream rangefile(tempfile.c_str());
		string rangeline;
		while(getline(rangefile, rangeline)) {
			stringstream lineStream(rangeline);
			int value;
			while(lineStream >> value) {
				range.push_back(value);
			}
		}

		//Initialise background for engine, clear appending vectors
		engine.initialise_background(filenames[0]);
		seq_scores.clear();

		//for (int i = 1; i <= filenames.size(); i+=1) {
  	for (int i = 1500; i <= 1510; i+=1) {
			//cout << "------------- Image: " << i << "\n";
			result = engine.process_frame(filenames[i-1], -1);

			//Write image to file with correct name
			string num_in = SSTR(i);
			string file_num = string(6 - num_in.length(), '0') + num_in;
			imwrite(write_dir+"bin"+file_num+".jpg", result);

			if (i >= range[0] && i <= range[1]) {
				//Read in ground truth image and compare to the SuperBE result
				groundtruth = imread(directory+"/"+sequence+"/groundtruth/gt"+file_num+".png");
				if (result.channels() > 2) cvtColor(result, result, CV_BGR2GRAY);
				if (groundtruth.channels() > 2) cvtColor(groundtruth, groundtruth, CV_BGR2GRAY);
				metrics = check_segmentation(result, groundtruth);
				seq_scores.push_back(metrics);

				//Write result for this image to file
				fs.open(resfile.c_str(), fstream::out|fstream::app);
			  fs << dirs.back() << "," << sequence << "," << file_num << ",";
				for (int k=0; k<11; k++) fs << metrics[k] << ",";
				fs << 0 << ",";
				for (int k=2; k<7; k++) fs << argv[k] << ",";
				fs << argv[7] << "\n";
			  fs.close();
			}
		}

		for (int k=0; k<11; k++) sums[k] = 0; //Reset sums before accumulating
		//Calculate aggregate statistics
		for (int j=0; j<seq_scores.size(); j++) {
			for (int k=0; k<11; k++) {
				sums[k] += seq_scores.at(j).at(k);
			}
		}
		cout << sequence << "\n";
		for (int k=0; k<11; k++) {
			avgs[k] = sums[k] / seq_scores.size();
			cout << sums[k] << ", " << avgs[k] << "\n";
		}

		//Write overall result for this sequence to file
		fs.open(resfile.c_str(), fstream::out|fstream::app);
		fs << dirs.back() << "," << sequence << "," << "OVERALL" << ",";
		for (int k=0; k<4; k++) fs << (int)sums[k] << ",";
		for (int k=4; k<11; k++) fs << avgs[k] << ",";
		fs << 1 << ",";
		for (int k=2; k<7; k++) fs << argv[k] << ",";
		fs << argv[7] << "\n";
		fs.close();

		//Clean up and remove the folder (to avoid running out of disk space)
		command = "rm -rf " + write_dir;
		ignorewarning = system(command.c_str());
	}
  //cout << "PROCESSING FINISHED\n";
	return 0;
}
