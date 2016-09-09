//Make sure that automatic webcam settings have been disabled
//Use cam.sh if necessary
#include "helper.h"
#include "superbe_core.h"

//Code for detecting if a key has been pressed
//cboard.cprogramming.com/c-programming/63166-kbhit-linux.html
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
int kbhit(void) {
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);
  if(ch != EOF){
    ungetc(ch, stdin);
    return 1;
  }
  return 0;
}


int main(int argc, char** argv) {
    superbe_engine engine;
    if (argc < 7) {
        printf(("\nUsage: %s <N> <R> <DIS> <numMin> <phi> <post>n"), argv[0]);
        return -1;
    } else {
        engine.set_init(atoi(argv[1]),atoi(argv[2]),atof(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6]));
    }

    //http://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    cout << "Camera Opened" << "\n";

    int frameNumber = 0;
    Mat frame;
    for (int i=0;i<10;i++) {
      cap >> frame; //Make sure camera comes out of USB suspend mode
    }
    engine.initialise_background(frame);
    cout << "Background Model Initialised" << "\n";

    while(1){ //Create infinte loop for live streaming
        cap >> frame; // get a new frame from camera
        engine.process_frame(frame, 10);
        frameNumber++;
        cout << "Frame: " << frameNumber << "\n";
        if (kbhit()) {
          getchar();
          cout << "Initialising Background" << "\n";
          engine.initialise_background(frame);
          cout << "Background Model Initialised" << "\n";
          frameNumber = 0;
        }
    }

    return 0;
}
