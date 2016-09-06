#include "superbe_core.h"

void superbe_engine::set_init(int i_N, int i_R, double i_DIS, int i_numMin, int i_phi, int i_post) {
    N = i_N;
    R = i_R;
    DIS = i_DIS;
    numMin = i_numMin;
    phi = i_phi;
    post = i_post;
    frameNumber = 1;
    srand(time(NULL));
}

Mat superbe_engine::filter_equalise() {
    //Filtering for noise reduction
    Mat filt_img;
    GaussianBlur(image, filt_img, Size(5, 5), 0, 0); //Should this be parameterised based on image size? Or not turned on at all?
    filt_img = equalizeIntensity(filt_img); //Equalise in YCbCr space
    return filt_img;
}

void superbe_engine::process_vals(Mat filt_img) {
    //Reset these containers because we are appending, not overwriting
    segment_pixels.clear();
    segment_pixvals.clear();
    segment_pixels.resize(numSegments); //An array of groups, each element is a vector of pixel positions
    segment_pixvals.resize(numSegments); //An array of groups, each element is a vector of pixel values

    //Build arrays of pixel positions and pixel values sorted by groups
    for(int i=0; i<height; i++) {
        int* pi = segments.ptr<int>(i);
        for(int j=0; j<width; j++) {
            Point loc = Point(j, i); //For some reason this has to be the other way around, not sure why!
            segment_pixels[pi[j]].push_back(loc); //Append the (x,y) location of the pixel to the appropriate group
            Vec3b vals = filt_img.at<Vec3b>(i, j);
            segment_pixvals[pi[j]].push_back(vals); //Append the (B,G,R) values of the pixel to the appropriate group
        }
    }

    //Calculate average and covariance matrix for each superpixel and store
    Mat vec3bplaceholder;
    for(int i=0; i<numSegments; i++) {
        if (segment_pixvals[i].size() != 0) {
            vec3bplaceholder = castVec3btoMat(segment_pixvals[i]);
            calcCovarMatrix(vec3bplaceholder, covars[i], avgs[i], CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_32F);
            covars[i] = covars[i]/(vec3bplaceholder.rows-1); //Normalise by 1/(N-1)
        } else { //For whatever reason, no elements in the superpixel (so can't calculate)
            if (frameNumber != 1) { //Fill with most recent values
                bgavgs.at(i).at(N-1).copyTo(avgs.at(i));
                bgcovars.at(i).at(N-1).copyTo(covars.at(i));
            } else { //No background model initialised yet, copy from a neighbour
                avgs.at(i-1).copyTo(avgs.at(i));
                covars.at(i-1).copyTo(covars.at(i));
            }
        }
    }
}

void superbe_engine::initialise_background(String filename) {
    initialise_background(imread(filename));
}

void superbe_engine::initialise_background(Mat image_in) {
    image = image_in;
    height = image.rows;
    width = image.cols;
    Mat filt_img = filter_equalise();

    //Pass the image through SLIC - just do this once per video sequence!
    //Superpixel size needs to be small enough, even when objects are small due to perspective distortion
    Ptr<SuperpixelSLIC> slic_engine = createSuperpixelSLIC(filt_img, SLICO); //SLICO has no parameters
    slic_engine->iterate();
    numSegments = slic_engine->getNumberOfSuperpixels();
    //cout << "Number of segments: " << numSegments << "\n";

    slic_engine->getLabels(segments);
    //Note: original slic.cpp must be fixed for this to work
    //https://github.com/Itseez/opencv_contrib/pull/483/files
    slic_engine->getLabelContourMask(edges);
    slic_engine.release(); //Release memory

    neighbours.clear();
    neighbours.resize(numSegments); //An array of groups, each element is a vector of neighbouring group numbers
    //Build neighbours, skipping the outside 1px border of pixels to avoid segmentation fault
    //This is fine because the neighbouring superpixels are likely to be captured elsewhere
    for(int i=1; i<height-1; i++) {
        int* pi = segments.ptr<int>(i);
        for (int j=1; j<width-1; j++) {
            //If neighbouring superpixel label is not the same as the current superpixel label, store it in the neighbour list
            if (segments.at<int>(i-1,j) != segments.at<int>(i,j)) neighbours.at(pi[j]).push_back(segments.at<int>(i-1,j));
            if (segments.at<int>(i,j-1) != segments.at<int>(i,j)) neighbours.at(pi[j]).push_back(segments.at<int>(i,j-1));
            if (segments.at<int>(i+1,j) != segments.at<int>(i,j)) neighbours.at(pi[j]).push_back(segments.at<int>(i+1,j));
            if (segments.at<int>(i,j+1) != segments.at<int>(i,j)) neighbours.at(pi[j]).push_back(segments.at<int>(i,j+1));
        }
    }

    //Remove duplicates - Use sort then unique for "small" vectors as evidenced in
    //https://stackoverflow.com/questions/1041620/whats-the-most-efficient-way-to-erase-duplicates-and-sort-a-vector
    //These vectors are typically around 100 elements long
    for(int i=0; i<numSegments; i++) {
        sort(neighbours.at(i).begin(), neighbours.at(i).end());
        neighbours.at(i).erase(unique(neighbours.at(i).begin(), neighbours.at(i).end()), neighbours.at(i).end());
        neighbours.at(i).push_back(i); //Add self to list to increase chances of adding self-values in
    }

    //Resize vectors of pixel averages and covariance matrices for number of superpixels
    avgs.resize(numSegments);
    covars.resize(numSegments);

    process_vals(filt_img); //Calculate means and covariance matrices

    bgavgs.resize(numSegments); //Allocate space for background model
    bgcovars.resize(numSegments);

    for (int i=0; i<numSegments; i++) {
        bgavgs.at(i).resize(N);
        bgcovars.at(i).resize(N);
        for (int j=0; j<N; j++) {
            if (neighbours.at(i).size() != 0) {
                //Select a random neighbouring superpixel (including potentially, self)
                randint = rand() % (neighbours.at(i).size());
                avgs.at(neighbours.at(i)[randint]).copyTo(bgavgs.at(i).at(j));
                covars.at(neighbours.at(i)[randint]).copyTo(bgcovars.at(i).at(j));
            } else { //No neighbours found, just fill with self-values
                avgs.at(i).copyTo(bgavgs.at(i).at(j));
                covars.at(i).copyTo(bgcovars.at(i).at(j));
            }
        }
    }

    //Compute morphological operation kernel sizes based on (average) superpixel size
    int openSize = (int)(sqrt(height * width / (double)numSegments) * 1.5);
    int closeSize = openSize * 3;
    structOpen = getStructuringElement(MORPH_ELLIPSE, Size(openSize, openSize));
    structClose = getStructuringElement(MORPH_ELLIPSE, Size(closeSize, closeSize));

    //cout << "Engine Initialised\n";
}

Mat superbe_engine::process_frame(String filename, int waitTime) {
    return process_frame(imread(filename), waitTime);
}

Mat superbe_engine::process_frame(Mat image_in, int waitTime) {
    image = image_in;
    Mat filt_img = filter_equalise();
    image.copyTo(segmented); //Help with visualising results
    segmented.setTo(Scalar(0, 0, 255), edges);
    process_vals(filt_img);

    Mat mask(height, width, CV_8UC1, Scalar(0)); //Initialise to zeros

    int euc_dist;
    double dissimilarity;
    for (int i=0;i<numSegments;i++) {
        //1) Compare pixel to background model
        int count = 0;
        int index = 0;

        while(count < numMin && index < N) {
            //Calculate euclidean distance between averages
            euc_dist = (int)round(norm(avgs.at(i), bgavgs.at(i).at(index)));

            //Jensen-Bregman LogDet Divergence [Cherian et al, 2011] [Eq (7)] https://lear.inrialpes.fr/people/cherian/papers/metricICCV.pdf
            //Use Cholesky factorisation to reduce computation
            //https://stackoverflow.com/questions/30223101/jensen-bregman-logdet-divergence/30236092#30236092
            Mat A, B, C;
            Cholesky(covars.at(i), A);
            Cholesky(bgcovars.at(i).at(index), B);
            Cholesky((covars.at(i) + bgcovars.at(i).at(index))*0.5, C);
            dissimilarity = abs(2*log(determinant(C)) - (log(determinant(A)) - log(determinant(B))));
            //dissimilarity = log( determinant((covars.at(i)+bgcovars.at(i).at(index)) * 0.5) - 0.5*log(determinant(covars.at(i)*bgcovars.at(i).at(index))) );

            //Check if enough similar samples have been found (if so, break)
            //If the colour covariance and the mean (LAB) colours are similar enough, it counts towards the background
            //if(euc_dist < R && (dissimilarity < DIS || !isfinite(dissimilarity))) {
            if(euc_dist < R && dissimilarity) {
                //if(dissimilarity < DIS) {
                count++;
            }
            index++;
        }

        //2) Classify superpixel and update model
        if(count >= numMin) {
            //3) Update current pixel model
            randint = rand() % (N-1);
            avgs.at(i).copyTo(bgavgs.at(i).at(randint));
            covars.at(i).copyTo(bgcovars.at(i).at(randint));

            //4) Update neighbouring superpixel model(s)
            randint = rand() % (phi-1);
            if (randint == 0) {
                if (neighbours.at(i).size() != 0) {
                    rand_neigh = neighbours.at(i).at(rand() % (neighbours.at(i).size()-1));
                    rand_bgmodel = rand() % (N-1);
                    avgs.at(i).copyTo(bgavgs.at(rand_neigh).at(rand_bgmodel));
                    covars.at(i).copyTo(bgcovars.at(rand_neigh).at(rand_bgmodel));
                }
            }
        } else {
            for(int j=0; j<segment_pixels.at(i).size(); j++) {
                mask.at<uchar>(segment_pixels.at(i).at(j)) = 255;
            }
        }
    }

    Mat masked_img(height, width, CV_8UC3, Scalar(0,0,0)); //Initialise to zeros
    Mat closed, opened;
    if (post == 1) {
        //Apply morphological operation
        morphologyEx(mask, closed, MORPH_CLOSE, structClose);
        morphologyEx(closed, opened, MORPH_OPEN, structOpen);

        image.copyTo(masked_img, opened); //Copy and mask
    } else {
        image.copyTo(masked_img, mask); //Copy and mask
    }

    if (waitTime >= 0) {
        //Concatenate together for display purposes
        Mat output(height*2, width*2, CV_8UC3);
        //For whatever reason, Rects are defined opposite to Mats
        Mat quad1(output, Rect(0, 0, width, height));
        Mat quad2(output, Rect(width, 0, width, height));
        Mat quad3(output, Rect(0, height, width, height));
        Mat quad4(output, Rect(width, height, width, height));

        segmented.copyTo(quad1);
        cvtColor(mask, quad2, CV_GRAY2RGB); //Handy to just output to desired location!
        if (post == 1) cvtColor(opened, quad3, CV_GRAY2RGB);
        masked_img.copyTo(quad4);

        imshow("Final", output);
        waitKey(waitTime);
    }

    frameNumber++;
    if (post == 1) {
        return opened;
    } else {
        return mask;
    }
}
