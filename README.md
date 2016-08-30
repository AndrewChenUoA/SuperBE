# SuperBE
Superpixel-level Background Estimation Algorithm

Known to compile on Ubuntu 14.04

Dependencies:
- OpenCV 3.1.0 including opencv_contrib (with corrected slic.cpp (https://github.com/opencv/opencv_contrib/pull/483))
- CMake 2.8

Instructions:
Use SuperBE.cpp to write your test program, instantiating a superbe_engine object (described in superbe_core.cpp).

The second argument of superbe_engine.process_frame() can be set to 0 in order to visually show the results with the user pressing any key to progress.

Use check_segmentation (described in analysis.cpp) to test SuperBE output images against a ground truth.

Usage: ./superbe <directory to dataset> <N> <R> <DIS> <numMin> <phi> <post> <ID>

Example: ./superbe ../dataset2014/dataset/baseline 20 60 18.0 4 8 1 test

CSV scripts for each test are stored in /cdwresults

Image files are temporarily stored in /resimg but deleted by the program