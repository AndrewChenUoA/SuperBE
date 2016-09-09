# SuperBE
Superpixel-level Background Estimation Algorithm

Known to compile on Ubuntu 14.04

Dependencies:
- OpenCV 3.1.0 including opencv_contrib (with corrected slic.cpp (https://github.com/opencv/opencv_contrib/pull/483))
- CMake 2.8

Instructions:
- Use SuperBE.cpp to write your test program, instantiating a superbe_engine object (described in superbe_core.cpp).
- The second argument of superbe_engine.process_frame() can be set to 0 in order to visually show the results with the user pressing any key to progress, or -1 to not show anything.
- Use check_segmentation (described in analysis.cpp) to test SuperBE output images against a ground truth.

For test purposes with the CDW2014 dataset (http://wordpress-jodoin.dmi.usherb.ca/dataset2014/):

Usage: ./superbe [directory to dataset] [N] [R] [DIS] [numMin] [phi] [post] [ID]

Example: ./superbe ../dataset2014/dataset/ 20 60 18.0 4 8 1 test_ID

CSV result files for each test are stored in /cdwresults. By default, the storage of the output masks is commented out.

superbe_webcam allows you to build a working demonstration that uses your computer's camera. Modify CMakeLists.txt to compile superbe_webcam instead, and call (example parameter settings may need to be modified):

./superbe_webcam 20 20 8.0 2 16 1