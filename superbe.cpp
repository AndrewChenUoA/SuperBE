#include <iostream>
#include <stdlib.h>
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

    String directory = argv[1];

    int post = 1;
    if (argc >= 8) {
        post = atoi(argv[7]);
    }

    String categories[] = {"badWeather", "baseline", "cameraJitter", "dynamicBackground", "intermittentObjectMotion", "lowFramerate", "nightVideos", "PTZ", "shadow", "thermal", "turbulence"};
    vector<vector<String> > sequences;
    String badWeather[] = {"blizzard", "skating", "snowFall", "wetSnow"};
    String baseline[] = {"highway", "office", "pedestrians", "PETS2006"};
    String cameraJitter[] = {"badminton", "boulevard", "sidewalk", "traffic"};
    String dynamicBackground[] = {"boats", "canoe", "fall", "fountain01", "fountain02", "overpass"};
    String intermittentObjectMotion[] = {"abandonedBox", "parking", "sofa", "streetLight", "tramstop", "winterDriveway"};
    String lowFramerate[] = {"port_0_17fps", "tramCrossroad_1fps", "tunnelExit_0_35fps", "turnpike_0_5fps"};
    String nightVideos[] = {"bridgeEntry", "busyBoulvard", "fluidHighway", "streetCornerAtNight", "tramStation", "winterStreet"};
    String PTZ[] = {"continuousPan", "intermittentPan", "twoPositionPTZCam", "zoomInZoomOut"};
    String shadow[] = {"backdoor", "bungalows", "busStation", "copyMachine", "cubicle", "peopleInShade"};
    String thermal[] = {"corridor", "diningRoom", "lakeSide", "library", "park"};
    String turbulence[] = {"turbulence0", "turbulence1", "turbulence2", "turbulence3"};
    sequences.push_back(vector<String> (badWeather, badWeather+sizeof(badWeather) / sizeof(badWeather[0])));
    sequences.push_back(vector<String> (baseline, baseline+sizeof(baseline) / sizeof(baseline[0])));
    sequences.push_back(vector<String> (cameraJitter, cameraJitter+sizeof(cameraJitter) / sizeof(cameraJitter[0])));
    sequences.push_back(vector<String> (dynamicBackground, dynamicBackground+sizeof(dynamicBackground) / sizeof(dynamicBackground[0])));
    sequences.push_back(vector<String> (intermittentObjectMotion, intermittentObjectMotion+sizeof(intermittentObjectMotion) / sizeof(intermittentObjectMotion[0])));
    sequences.push_back(vector<String> (lowFramerate, lowFramerate+sizeof(lowFramerate) / sizeof(lowFramerate[0])));
    sequences.push_back(vector<String> (nightVideos, nightVideos+sizeof(nightVideos) / sizeof(nightVideos[0])));
    sequences.push_back(vector<String> (PTZ, PTZ+sizeof(PTZ) / sizeof(PTZ[0])));
    sequences.push_back(vector<String> (shadow, shadow+sizeof(shadow) / sizeof(shadow[0])));
    sequences.push_back(vector<String> (thermal, thermal+sizeof(thermal) / sizeof(thermal[0])));
    sequences.push_back(vector<String> (turbulence, turbulence+sizeof(turbulence) / sizeof(turbulence[0])));

    for (int cat=0; cat<=sequences.size(); cat++) {
        String category = categories[cat];
        vector<String> filenames;

        String ID = (argc == 9) ? argv[8] : "0";
        String command;

        fstream fs;
        String resfile = "cdwresults/"+category+ID+".csv";
        fs.open(resfile.c_str(), fstream::out);
        fs << "Category,Sequence,File,TP,TN,FP,FN,Rec,Spec,FPR,FNR,PWr,F-Meas,Prec,procTime,Overall,N,R,DIS,numMin,phi,post" << "\n";
        fs.close();

        //Initialise memory for tests, avoiding redefinition of large items
        Mat result, groundtruth;
        vector<double> metrics;
        vector<vector<double> > seq_scores;
        double sums[12];
        double avgs[12];
        int ignorewarning;
        int start, stop;
        double procTime;

        for (int seq=0; seq<sequences[cat].size(); seq++) {
            String sequence = sequences[cat][seq];

            //Initialise engine for each sequence with command line arguments
            superbe_engine engine;
            if (argc < 7) {
                printf(("\nUsage: %s <read directory> <N> <R> <DIS> <numMin> <phi> <post> <ID>\n"), argv[0]);
                return -1;
            } else {
                engine.set_init(atoi(argv[2]),atoi(argv[3]),atof(argv[4]),atoi(argv[5]),atoi(argv[6]), post);
            }
            
            //Commented out code is if you want to save the result masks
            //String write_dir = "resimg/"+ID+sequence+"/";
            //command = "mkdir " + write_dir;
            //ignorewarning = system(command.c_str());
            glob(directory+"/"+category+"/"+sequence+"/input", filenames); //Pulls the filenames out of the directory

            //Check what the range for this sequence is
            vector<int> range;
            String tempfile = directory+"/"+category+"/"+sequence+"/"+"temporalROI.txt";
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

            for (int i = 1; i <= filenames.size(); i+=1) {
                //cout << "------------- Image: " << i << "\n";
                
                start = clock();
                
                //PROCESS THE FRAME
                result = engine.process_frame(filenames[i-1], -1);
                
                stop = clock();
                procTime = (double)(stop-start)/CLOCKS_PER_SEC;

                //Write image to file with correct name
                string num_in = SSTR(i);
                string file_num = string(6 - num_in.length(), '0') + num_in;
                //imwrite(write_dir+"bin"+file_num+".jpg", result);

                if (i >= range[0] && i <= range[1]) {
                    //Read in ground truth image and compare to the SuperBE result
                    groundtruth = imread(directory+"/"+category+"/"+sequence+"/groundtruth/gt"+file_num+".png");
                    if (result.channels() > 2) cvtColor(result, result, CV_BGR2GRAY);
                    if (groundtruth.channels() > 2) cvtColor(groundtruth, groundtruth, CV_BGR2GRAY);
                    metrics = check_segmentation(result, groundtruth);
                    metrics.push_back(procTime);
                    seq_scores.push_back(metrics);

                    //Write result for this image to file
                    fs.open(resfile.c_str(), fstream::out|fstream::app);
                    fs << category << "," << sequence << "," << file_num << ",";
                    for (int k=0; k<12; k++) fs << metrics[k] << ",";
                    fs << 0 << ",";
                    for (int k=2; k<7; k++) fs << argv[k] << ",";
                    fs << post << "\n";
                    fs.close();
                }
            }

            for (int k=0; k<12; k++) sums[k] = 0; //Reset sums before accumulating
            //Calculate aggregate statistics
            for (int j=0; j<seq_scores.size(); j++) {
                for (int k=0; k<12; k++) {
                    sums[k] += seq_scores.at(j).at(k);
                }
            }

            for (int k=0; k<12; k++) {
                avgs[k] = sums[k] / (double) seq_scores.size();
            }

            //Write overall result for this sequence to file
            fs.open(resfile.c_str(), fstream::out|fstream::app);
            fs << category << "," << sequence << "," << "OVERALL" << ",";
            for (int k=0; k<4; k++) fs << (int)sums[k] << ",";
            for (int k=4; k<12; k++) fs << avgs[k] << ",";
            fs << 1 << ",";
            for (int k=2; k<7; k++) fs << argv[k] << ",";
            fs << argv[7] << "\n";
            fs.close();

            //Clean up and remove the folder (to avoid running out of disk space)
            //command = "rm -rf " + write_dir;
            //ignorewarning = system(command.c_str());
        }
    }
    //cout << "PROCESSING FINISHED\n";
    return 0;
}
