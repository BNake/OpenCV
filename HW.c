// flandmarks

// Run this example to check that your opencv and dlib libraries are up and ready to run
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

//using namespace dlib;
using namespace std;
//using namespace cv;

int main(int argc, char** argv)
{
  // dlib's face detector
  dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

  // facial landmark detector is a particular case of shape_predictor
  dlib::shape_predictor facial_landmark_detector;

  // loading  the facial landmark model
  dlib::deserialize("../../shared/shape_predictor_68_face_landmarks.dat") >> facial_landmark_detector;

  // reading OpenCV image
  cv::Mat image = cv::imread("../images/images.jpeg");

  // converting OpenCV image format to Dlib's image format
  dlib::cv_image<dlib::bgr_pixel> dlib_image(image);

  // detecting all the faces in the image
  std::vector<dlib::rectangle> face_rectangles = face_detector(dlib_image);

  // for all faces...
  for (int i = 0; i < face_rectangles.size(); i++)
  {
    // find landmarks for face_rectangles[i]
    dlib::full_object_detection landmarks = facial_landmark_detector(dlib_image, face_rectangles[i]);

    // draw rectangle and landmarks on face
    for(int i = 0; i < landmarks.num_parts(); i++)
    {
	    int x = landmarks.part(i).x();
	    int y = landmarks.part(i).y();
	    circle(image, cv::Point(x, y), 21, cv::Scalar(0, 0, 255));
	}
  }

  // show resultant image
  cv::imshow("Latuta and Dicaprio", image);
  cv::waitKey(0);
  return 0;
}



// week03
// access_pixel.cpp
// Mat.at<Vec3>(y,x)[channel], here 0=Blue, 1=Green, 2=Red, for RGB images
// Mat.at<uchar>(y,x), unsigned char value, for grayscale images
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(){
    // for 3-channel pixels
    Mat A = imread("../images/butterfly.png");
    A.at<Vec3b>(30,50)[0] = 0;   // B
    A.at<Vec3b>(30,50)[1] = 253; // G
    A.at<Vec3b>(30,50)[2] = 255; // R
    imshow("A",A);
    // for 1-channel pixels
    Mat B = imread("../images/butterfly.png",IMREAD_GRAYSCALE);
    B.at<uchar>(30,50) = 255;
    imshow("B",B);
    // printing the above pixel using pointer arithmetics based CRAZY method :)
    cout << " A[30][50] (Blue)  = " << (int) A.data[ A.step[0]*30 + A.step[1]*50 + 0] << endl; // B
    cout << " A[30][50] (Green) = " << (int) A.data[ A.step[0]*30 + A.step[1]*50 + 1] << endl; // G
    cout << " A[30][50] (Red)   = " << (int) A.data[ A.step[0]*30 + A.step[1]*50 + 2] << endl; // R
    cout << " A[30][50] (BGR)   = " << *((Vec3b*) (A.data + A.step[0]*30 + A.step[1]*50)) << endl; // BGR
    cout << " B[30][50] (Gray) = " <<  (int) B.data[ B.step[0]*30 + B.step[1]*50 + 0] << endl; // Gray
    waitKey(0);
    return 0;
}


// week03
// assign_region.cpp
// sometimes we need only a particular region of an image
// it is called a "region of interest" (ROI) in computer vision literature

#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    Mat A = imread("../images/butterfly.png");

    Mat B(A, Rect(50,10,150,100)); // x, y, width, height
    Mat C = A(Range(10,10+100), Range(50,50+150)); // rows, cols
    imshow("A",A);
    imshow("B",B);
    imshow("C",C);
    waitKey(0);

    B.setTo(Scalar(255,0,0)); // make it blue

    imshow("A changed",A);
    imshow("B changed",B);
    imshow("C changed",C);
    waitKey(0);

    return 0;
}


// week03
// assign.cpp
// assignment operator only copies header around
// physically data is not copied

#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    Mat A = imread("../images/butterfly.png");
    Mat B(A);   // initializing B with A
    Mat C = A;  // A.data, B.data and C.data point to the same memory physically

    imshow("A",A);
    imshow("B",B);
    imshow("C",C);
    waitKey(0);

    // making a small region of A black
    A(Range(0, 50), Range(50,100)).setTo(Scalar(0,0,0));

    imshow("A(black)",A);
    imshow("B(black)",B);
    imshow("C(black)",C);
    waitKey(0);

    // A, B and C all have the black region painted
    // this proves that they point to the same image physically
    return 0;
}


#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

const int NUMQ=20; // number of questions per column
const int NUMC=4;

int ans[NUMQ*NUMC+1];
string variant;
vector<Point> p;
Mat A, B, C, D, E, F, G, H, I, J, K, L,M;
vector<vector<Point> > blob;
vector<Rect> bRect;
vector<Rect> xRect;
vector<vector<Point> > xblob;

//const char ENTER_KEY = 10;
const char ESC_KEY = 27;
//const char BACKSPACE_KEY = 8;
//const char SPACE_KEY = 32;

const int WIN_WIDTH = 1366;
const int WIN_HEIGHT = 768;

//const int WIN_WIDTH = 768;
//const int WIN_HEIGHT = 1366;

//const int A4_WIDTH = 840;
//const int A4_HEIGHT = 1188;
const int A4_WIDTH = 1188;
const int A4_HEIGHT = 840;

int sz = 65;
int c = 138; // 128 + 10
int morph_sz = 5;

double alpha=1.0,beta = 20.0;

int read_answers(int j){
    variant = "./variants/A";
    variant[variant.length()-1]+=j;
    cout << "Variant = " << variant << endl;
    ifstream fin(variant);
    if (!fin.is_open()) {
        cout << "Answers file was not opened." << endl;
        return -1;
    }
    for(int i=0;i<NUMQ*NUMC+1;i++){
        fin >> ans[i];
    }
    fin.close();
    return 0;
}

void get_binary(){
    adaptiveThreshold(D,E,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,sz/2*2+1,c-128);
    erode(E,F,getStructuringElement(MORPH_RECT,Size(morph_sz,morph_sz)));
    dilate(F,G,getStructuringElement(MORPH_RECT,Size(morph_sz,morph_sz)));
    bitwise_not(G,H);
    I = H.clone();
    floodFill(I, Point(0,0), Scalar(255));
    Mat I_inv;
    I_inv = 255-I;
    J = H + I_inv;
    imshow("window",J);
}

int fillIntensity(Rect bb){
    Mat W = D(bb);
    int t = W.at<uchar>(0,0)/4 + W.at<uchar>(0,W.cols-1)/4 + W.at<uchar>(W.rows-1,0)/4 + W.at<uchar>(W.rows-1,W.cols-1)/4;
    threshold(W,W,t-15,255,THRESH_BINARY);
    int white = countNonZero(W);
    cout << "w = " << white << " b = " << (bb.width*bb.height)-white << " w+b = "<< (bb.width*bb.height) << endl;
    return white <= (bb.width*bb.height)/2 ? (bb.width*bb.height) - white : 0;
}

int get_blobs() {
    blob.clear();
    bRect.clear();
    xRect.clear();
    xblob.clear();

    vector<vector<Point> > contours;
    cvtColor(D, K, COLOR_GRAY2BGR);
    findContours(J, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++) {
        Rect bb = boundingRect(contours[i]); // bounding box
        if (bb.width > 20 && bb.width < 30 && bb.height > 20 && bb.height < 30) {
            blob.push_back(contours[i]);
            bRect.push_back(bb);
        }
    }
    cout << "blob.size() = " << blob.size() << endl;

    drawContours(K,blob,-1,Scalar(0,0,255),2);

    ostringstream sout;
    sout << "blobs count = " << blob.size();
    putText(K,sout.str(),Point(30,50),FONT_HERSHEY_COMPLEX_SMALL,1.5,Scalar(0,0,255),2);
    imshow("window", K);
    return blob.size();
}


void get_score() {
    if (blob.size()+xblob.size()-1==(NUMQ*NUMC+1)*5){
        if (xblob.size()>0){
            blob.insert(blob.end(),xblob.begin()+1, xblob.end());
            bRect.insert(bRect.end(),xRect.begin()+1, xRect.end());
            xblob.clear();
            xRect.clear();
        }
    }

    L = C.clone();

    int score = 0;
    ostringstream sout;
    if (blob.size() == ( NUMQ*NUMC + 1 )*5){
//        Mat K=C.clone();
        /*for(int i=0;i<bRect.size();i++){
            rectangle(K,bRect[i],Scalar(0,200,100),2);
            imshow("window",K);
            waitKey(100);
        }*/

        for (int i = 0; i < blob.size() - 1; i++) {
            for (int j = i + 1; j < blob.size(); j++)
                if (bRect[i].y > bRect[j].y) {
                    vector<Point> td = blob[i];
                    blob[i] = blob[j];
                    blob[j] = td;
                    Rect tb = bRect[i];
                    bRect[i] = bRect[j];
                    bRect[j] = tb;
                }
            /*
            if (bRect[i].y - bRect[j].y > bRect[j].height/2) {
                vector<Point> td = blob[i];
                blob[i] = blob[j];
                blob[j] = td;
                Rect tb = bRect[i];
                bRect[i] = bRect[j];
                bRect[j] = tb;
            } else
            if (std::abs(bRect[i].y - bRect[j].y) <= bRect[i].height/2 && bRect[i].x > bRect[j].x+bRect[j].width) {
                vector<Point> td = blob[i];
                blob[i] = blob[j];
                blob[j] = td;

                Rect tb = bRect[i];
                bRect[i] = bRect[j];
                bRect[j] = tb;
            }*/
        }

        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 5; j++)
                if (bRect[i].x > bRect[j].x) {
                    vector<Point> td = blob[i];
                    blob[i] = blob[j];
                    blob[j] = td;
                    Rect tb = bRect[i];
                    bRect[i] = bRect[j];
                    bRect[j] = tb;
                }
        }

        for(int t=5;t<(NUMQ*NUMC+1)*5;t+=NUMC*5)
            for (int i = t; i < t+NUMC*5 - 1; i++) {
                for (int j = i + 1; j < t+NUMC*5; j++)
                    if (bRect[i].x > bRect[j].x) {
                        vector<Point> td = blob[i];
                        blob[i] = blob[j];
                        blob[j] = td;
                        Rect tb = bRect[i];
                        bRect[i] = bRect[j];
                        bRect[j] = tb;
                    }
            }

        M=C.clone();

        /*for(int i=0;i<bRect.size();i++){
            rectangle(K,bRect[i],Scalar(100,200,200),2);
            imshow("window",K);
            waitKey(100);
        }*/

        for(int i=0;i<A4_HEIGHT;i+=10){
            if (i/10%2==0)
                line(M,Point(0,i),Point(A4_WIDTH-1,i),Scalar(100,255,0),1);
            else
                line(M,Point(0,i),Point(A4_WIDTH-1,i),Scalar(255,100,0),1);
        }


        Scalar col[5] = {Scalar(0,0,255),Scalar(0,255,0),Scalar(0,255,255),Scalar(255,255,0),Scalar(255,255,255)};
        for(int nv = 0; nv < NUMQ*NUMC+1; nv++) { // nomer voprosa
            for (int vo = 0; vo < 5; vo++) {
                if (nv==0){
                    //rectangle(K,bRect[vo],Scalar(255,255,255),2);
                    ostringstream s1,s2;
                    s1 << nv;
                    s2 <<"ABCDE"[vo];
                    putText(M,s1.str(), Point(bRect[vo].x,bRect[vo].y+5), FONT_HERSHEY_PLAIN, 1, col[vo], 2);
                    putText(M,s2.str(), Point(bRect[vo].x+15,bRect[vo].y+20), FONT_HERSHEY_PLAIN, 1, col[vo], 2);
                } else{
                    int ii = (nv-1) % NUMQ * NUMC * 5 + (nv-1)/NUMQ*5 + vo + 5;
                    ostringstream s1,s2;
                    s1 << nv;
                    s2 <<"ABCDE"[vo];
                    putText(M,s1.str(), Point(bRect[ii].x,bRect[ii].y+5), FONT_HERSHEY_PLAIN, 1, col[vo],2);
                    putText(M,s2.str(), Point(bRect[ii].x+15,bRect[ii].y+20), FONT_HERSHEY_PLAIN, 1, col[vo], 2);
                }
            }
        }

        imshow("window",M);
        waitKey(0);

        for(int nv = 0; nv < NUMQ*NUMC+1; nv++) { // nomer voprosa
            cout << "vopros " << nv << endl;
            int w[5] = {0, 0, 0, 0, 0};
            for (int vo = 0; vo < 5; vo++) { // variant otv
                if (nv==0)
                    w[vo] = fillIntensity(bRect[vo]);
                else
                    w[vo] = fillIntensity(bRect[(nv-1) % NUMQ * NUMC*5 + (nv-1)/NUMQ*5 + vo + 5]);
            }

            int j = 0;
            for (int i = 1; i < 5; i++) if (w[i] > w[j]) j = i;

            if (nv==0){
                if (read_answers(j)<0) break; // read answers of variant "ABCDE"[j]
            }

            int correct_index = (nv-1) % NUMQ * NUMC * 5 + (nv-1) / NUMQ * 5 + ans[nv]-1 + 5;
            int answered_index = (nv-1) % NUMQ * NUMC * 5 + (nv-1) / NUMQ * 5 + j + 5;

            if (ans[nv]-1==j && w[j] > 0) {
                if (nv>0){
                    score++;
                    // correct answer
                    cout << "correct" << endl;
                    drawContours(L, blob, correct_index , Scalar(0, 255, 255), 2);
                }
                else{
                    // variant
                    cout << "variant" << endl;
                    drawContours(L, blob, j, Scalar(255, 255, 255), 2);
                }

            } else {
                if (w[j] > 0) {
                    cout << "wrong1" << endl;
                    drawContours(L, blob, answered_index, Scalar(0, 0, 255), 2);
                    cout << "wrong2" << endl;
                    drawContours(L, blob, correct_index, Scalar(0, 255, 0), 2);
                    cout << "wrong3" << endl;
                } else {
                    cout << "empty" << endl;
                    drawContours(L, blob, correct_index, Scalar(255, 255, 0), 2);
                }

            }
        }
        sout << "score = " << score;
    }
    else {
        cout << "all wrong" << endl;
        drawContours(L, blob,-1, Scalar(0,0, 255), 2);
        sout << "score = " << score << " :(";
    }


    cout << sout.str() << endl;
    cout << blob.size() << endl;
    putText(L, variant, Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1.5, Scalar(0, 0, 255), 2);
    putText(L, sout.str(), Point(600, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.5, Scalar(0, 0, 255), 2);

    imshow("window",L);
}

void onMouse(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        if (p.size()<4)
        {
            p.push_back(Point(x,y));
            cout << "p["<< p.size()-1 << "].x = " << p[p.size()-1].x << " and p["<< p.size()-1 << "].y = " << p[p.size()-1].y << endl;
            circle(B,p[p.size()-1],40,Scalar(0,0,255),-1);
            imshow("window",B);
        }
        cout << "on mouse..." << Point(x,y) << endl;
    }

}


void onMouse2(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        cout << "on mouse 2 ... "<< Point(x,y) << endl;
        if (xRect.size()==0){
            for(int j=0;j<bRect.size();j++){
                if (bRect[j].contains(Point(x,y))){
                    cout << bRect[j] << " contains " << Point(x,y) << endl;
                    drawContours(K,blob,j,Scalar(0,255,255),2);
                    xRect.push_back(bRect[j]);
                    xblob.push_back(blob[j]);
                    break;
                }
            }
        } else{
            int j;
            for(j=0;j<bRect.size();j++)
                if (bRect[j].contains(Point(x,y))) break;
            if (j<bRect.size()) {
                drawContours(K,blob,j,Scalar(255,255,255),2);
                bRect.erase(bRect.begin()+j);
                blob.erase(blob.begin()+j);
                imshow("window",K);
                return;
            }

            for(j=0;j<xRect.size();j++)
                if (xRect[j].contains(Point(x,y))) break;
            if (j<xRect.size()) {
                drawContours(K,xblob,j,Scalar(255,255,255),2);
                xRect.erase(xRect.begin()+j);
                xblob.erase(xblob.begin()+j);
                imshow("window",K);
                return;
            }

            xRect.push_back(Rect(x-xRect[0].width/2,y-xRect[0].height/2,xRect[0].width,xRect[0].height));
            xblob.push_back(xblob[0]);
            int n = xblob.size();
            int m = xblob[n-1].size();
            for(int i=0;i<m;i++){
                Point offset = xblob[0][i] - Point(xRect[0].x,xRect[0].y);
                xblob[n-1][i] = Point(xRect[n-1].x,xRect[n-1].y) + offset;
            }
            drawContours(K,xblob,n-1,Scalar(255,0,255),2);
        }
    }
    imshow("window",K);

}

void onThreshConstantChange(int newc, void*){
    c = newc;
    get_binary();
    get_blobs();
}

void onThreshSizeChange(int newsz, void*){
    sz = newsz;
    get_binary();
    get_blobs();
}

void onMorphSizeChange(int newsz, void*){
    morph_sz = newsz;
    get_binary();
    get_blobs();
}


int main(int argc, char*argv[])
{
    // step 0: read image and correct answers

    if (argc<2) return -1;
    A = imread(argv[1]);
    if (A.empty()) {
        cout << "Image file was not opened." << endl;
        return -1;
    }

    cout << "Step 0 was done." << endl;

    // step 1: select points and warpPerspective, convert to grayscale
    char ch;
    B = A.clone();
    namedWindow("window",WINDOW_KEEPRATIO);
    resizeWindow("window",WIN_WIDTH,WIN_HEIGHT);
    imshow("window",B);
    setMouseCallback("window", onMouse);

    ch = (char) waitKey(0);
    if (ch==ESC_KEY) return -1;
    if (p.size()!=4) return -1;

    Point2f src[4] = { p[0], p[1], p[2], p[3]};
    Point2f dst[4] = {Point2f(0,0),Point2f(A4_WIDTH-1,0),Point2f(A4_WIDTH-1,A4_HEIGHT-1),Point2f(0,A4_HEIGHT-1)};
    Mat T = getPerspectiveTransform(src,dst);
    warpPerspective(A, C, T, Size(A4_WIDTH,A4_HEIGHT));
    imshow("window",C);
    ch = (char) waitKey(0);
    if (ch==ESC_KEY) return -1;

    cvtColor(C,D,COLOR_BGR2GRAY);
    imshow("window",D);

    ch = (char) waitKey(0);
    if (ch==ESC_KEY) return -1;
    //step 2: getting binaries
    createTrackbar("sz","window",&sz,A4_WIDTH/2,onThreshSizeChange);
    createTrackbar("c+128","window",&c,255,onThreshConstantChange);
    createTrackbar("morph_sz","window",&morph_sz,31,onMorphSizeChange);
    get_binary();
    get_blobs();

    setMouseCallback("window", nullptr);
    setMouseCallback("window", onMouse2);
    get_blobs();
    ch = (char) waitKey(0);
    if (ch==ESC_KEY) return -1;
    // step 3: getting sorted list of blobs
    destroyWindow("window");
    namedWindow("window",WINDOW_KEEPRATIO);
    resizeWindow("window",WIN_WIDTH,WIN_HEIGHT);
    get_score();

    ch = (char) waitKey(0);
    if (ch==ESC_KEY) return -1;

    imwrite(string(argv[1])+string("_SCORE.jpg"),L);
    cout << "Saved as " << string(argv[1])+string("_SCORE.jpg") << endl;
}


// week03
// blob_analysis.cpp
// drawing connected components

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

Mat get_painted_blobs(Mat blobs)
{
    // copy of the blob matrix
    Mat labels = blobs.clone();
    // find the min and max values in imLabels
    Point minpoint, maxpoint;
    double min, max;
    minMaxLoc(labels, &min, &max, &minpoint, &maxpoint);
    // normalize the matrix so that min value is 0 and max value is 255.
    labels  = 255 * (labels - min) / (max - min);
    // convert matrix to 8-bits to be an "image"
    labels .convertTo(labels, CV_8U);
    // apply a color map
    Mat painted;
    applyColorMap(labels, painted, COLORMAP_JET);
    return painted;
}

int main(){
    // original
    Mat image = imread("../images/answers.png");
//    image = image(Rect(0,0,image.cols,35)); // crop out
    // converting to grayscale
    Mat gray;
    cvtColor(image,gray,COLOR_BGR2GRAY);
    // inverse binary thresholding
    Mat binary;
    threshold(gray,binary,100,255,THRESH_BINARY_INV);

    // this is not an image in 0-255 range
    // because the number of blobs can be greater than 256
    Mat blobs;
    connectedComponents(binary,blobs);

    imshow("original", binary);
    imshow("blobs", get_painted_blobs(blobs));
    waitKey(0);
}

// week03
// camera_read_show.cpp
// how to read video from file/camera

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(){
    //VideoCapture vin("../videos/bike.avi");
    VideoCapture vin(0); // access to laptop built-in or USB camera
    //VideoCapture vin("rtsp://192.168.12.100/media/live/1/1"); // access to IP camera
    Mat frame;

    cout << "FPS = " << vin.get(CAP_PROP_FPS) << endl; // this one will not work for cameras :(
    cout << "COUNT = " << vin.get(CAP_PROP_FRAME_COUNT) << endl; // we don't know future
    cout << "WIDTH = " << vin.get(CAP_PROP_FRAME_WIDTH) << endl; // this may work
    cout << "HEIGHT = " << vin.get(CAP_PROP_FRAME_HEIGHT) << endl; // this may work

    vin >> frame;
    while (!frame.empty()){
        imshow("video",frame);
        waitKey(33);
        vin >> frame;
    }

    return 0;
}



// week03
// clone_copy.cpp
// when we need to make a copy of an image OR a copy of a part of an image

#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    Mat A = imread("../images/sdu.png");
    Mat B = A.clone(); // a new physical copy of A
    Mat C;             // empty matrix
    A.copyTo(C);       // another copy of A

    C(Range(185,215),Range(245,270)).setTo(Scalar(0,0,255)); // set to Red
    B(Range(185,215),Range(245,270)).setTo(Scalar(0,255,255)); // set to Yellow

    imshow("A",A);
    imshow("B",B);
    imshow("C",C);
    waitKey(0);
    // B and C changed, while A didn't => all 3 images are physically different
    return 0;
}


// week03
// clone_mask.cpp
// creating a mask and copy using the mask

#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    Mat A = imread("../images/sdu.png");
    Mat mask(A.rows,A.cols,CV_8UC1,Scalar(0)); // grayscale image of black color
    mask(Rect(130,55,250,200)).setTo(Scalar(255));  // make a white square

    Mat B;
    A.copyTo(B,mask); // copy with mask

    imshow("A",A);
    imshow("mask",mask);
    imshow("B",B);
    waitKey(0);

    return 0;
}


// week03
// crop_resize.cpp
// resizing images and cropping
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(){
    Mat A = imread("../images/sdu.png");
    Mat B1,B2,C1,C2;
    double fx1 = 1.3; // scaling factor by x axis
    double fy1 = 0.5; // scaling factor by y axis
    double fx2 = 0.5; // scaling factor by x axis
    double fy2 = 1.3; // scaling factor by y axis
    resize(A,B1,Size(A.cols*fx1, A.rows*fy1),0,0,INTER_LINEAR); // interpolation method in the end
    resize(A,B2,Size(),fx2,fy2,INTER_CUBIC);
    C1 = A(Rect(400,60,130,200)); // cropping with Rect(x,y,width,height)
    C2 = A(Range(90,230),Range(170,345)); // cropping with Range()

    imshow("A",A);
    imshow("B1",B1);
    imshow("B2",B2);
    imshow("C1",C1);
    imshow("C2",C2);
    cout << "A.size() = " << A.size() << endl;
    cout << "B1.size() = " << B1.size() << endl;
    cout << "B2.size() = " << B2.size() << endl;
    cout << "C1.size() = " << C1.size() << endl;
    cout << "C2.size() = " << C2.size() << endl;
    waitKey(0);
    return 0;
}


// week03
// drawing.cpp
#include <opencv2/opencv.hpp>
using namespace cv;
int main(void)
{
    Mat image(300,300,CV_8UC3,Scalar(255,255,255));
    line(image,Point(20,100),Point(50,250),Scalar(0,150,50),2,LINE_AA); // anti-aliasing ON
    line(image,Point(280,150),Point(50,250),Scalar(150,50,50),2); // anti-aliasing OFF
    circle(image,Point(150,150),50,Scalar(0,0,150),2,LINE_AA);
    rectangle(image,Rect(10,10,100,50),Scalar(200,0,0),1,LINE_AA);
    putText(image,"hello world!",Point(130,30),FONT_HERSHEY_COMPLEX_SMALL,1,Scalar(55,55,0),1,LINE_AA);
    ellipse(image,Point(150,150),Size(20, 40), 45, 0, 360, Scalar(0, 150, 0), -1, LINE_AA);
    imshow("image",image);
    waitKey(0);
}

// Run this example to check that your opencv and dlib libraries are up and ready to run
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

//using namespace dlib;
using namespace std;
//using namespace cv;

int main(int argc, char** argv)
{
  // dlib's face detector
  dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

  // facial landmark detector is a particular case of shape_predictor
  dlib::shape_predictor facial_landmark_detector;

  // loading  the facial landmark model
  dlib::deserialize("../../shared/shape_predictor_68_face_landmarks.dat") >> facial_landmark_detector;

  // reading OpenCV image
  cv::Mat image = cv::imread("../images/dicaprio.png");

  // converting OpenCV image format to Dlib's image format
  dlib::cv_image<dlib::bgr_pixel> dlib_image(image);

  // detecting all the faces in the image
  std::vector<dlib::rectangle> face_rectangles = face_detector(dlib_image);

  // for all faces...
  for (int i = 0; i < face_rectangles.size(); i++)
  {
    // find landmarks for face_rectangles[i]
    dlib::full_object_detection landmarks = facial_landmark_detector(dlib_image, face_rectangles[i]);

    // draw rectangle and landmarks on face
    for(int i = 0; i < landmarks.num_parts(); i++)
    {
	    int x = landmarks.part(i).x();
	    int y = landmarks.part(i).y();
	    circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255));
	}
  }

  // show resultant image
  cv::imshow("Latuta and Dicaprio", image);
  cv::waitKey(0);
  return 0;
}

// week03
// keyboard.cpp
// waitKey

#include <iostream>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

int main(){
    Mat A(300,200,CV_8UC3,Scalar(150,150,150));
    imshow("window",A);

    // until escape
    char key = (char) waitKey(0);
    while( key != 27) {
        if (key<=32)
            cout << "#" << (int)key << " was pressed" << endl;
        else
            cout << "'" << key << "' was pressed" << endl;
        key = (char) waitKey(0);
    }
    cout << "#" << (int)key << " was pressed" << endl;
}


// week03
// mat_create_more.cpp
// other ways to create matrices
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(){
    Mat A;
    A.create(2, 3, CV_8UC1);
    cout << "A = " << endl << A << endl;
    Mat B = Mat::eye(3, 3, CV_8UC1);
    cout << "B = " << endl << B << endl;
    Mat C = Mat::ones(1, 3, CV_8UC1);
    cout << "C = " << endl << C << endl;
    Mat D = Mat::zeros(3, 7, CV_8UC1);
    cout << "D = " << endl << D << endl;

    return 0;
}

// week03
// mat_create.cpp
// CV_<bit-depth>{U|S|F}C<number_of_channels> pattern of matrix data type
// bit-depth = 8,16,32,64
// U=unsigned, S=signed, F=floating
// number_of_channels = 1,2,3,4 ...
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(){
    // 8 bit unsigned, 1 channel
    Mat A(3,4,CV_8UC1,13); // rows, cols, type, color
    cout << "A = " << endl << A << endl;
    // 16 bit unsigned, 3 channels
    Mat B(3,4,CV_16UC3,Scalar(1001,1002,1003));
    cout << "B = " << endl << B << endl;
    // 32 bit float, 4 channels
    Mat C(3,4,CV_32FC4, Scalar(1.0, 2.0, 3.5, 4.0));
    cout << "C = " << endl << C << endl;
    return 0;
}

// week03
// mat_methods.cpp
// some methods and attributes from the zoo of Mat.methods() and Mat.attributes

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(){
    Mat A = imread("../images/sdu.png");
    cout << "A.rows = " << A.rows << endl;
    cout << "A.cols = " << A.cols << endl;
    cout << "A.size() = " << A.size() << endl;
    cout << "A.size().width = " << A.size().width << endl;
    cout << "A.size().height = " << A.size().height << endl;
    cout << "A.channels() = " << A.channels() << endl;
    return 0;
}


// week03
// morphology_ex.cpp
// opening and closing

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(){
    // original
    Mat image = imread("../images/answers.png");
    // converting to grayscale
    Mat gray;
    cvtColor(image,gray,COLOR_BGR2GRAY);
    // inverse binary thresholding
    Mat binary;
    threshold(gray,binary,130,255,THRESH_BINARY_INV);
    // structuring element
    Mat S = getStructuringElement(MORPH_RECT,Size(3,3));
    cout << S << endl;
    // opening
    Mat opened;
    morphologyEx(binary,opened,MORPH_OPEN,S);
    // closing
    Mat closed;
    morphologyEx(binary,closed,MORPH_CLOSE,S);
    // show all
    imshow("original",binary);
    imshow("closed",closed);
    imshow("opened",opened);
    waitKey(0);
}

// week03
// morphology.cpp
// dilation and erosion

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(){
    // original
    Mat image = imread("../images/answers.png");
    // converting to grayscale
    Mat gray;
    cvtColor(image,gray,COLOR_BGR2GRAY);
    // inverse binary thresholding
    Mat binary;
    threshold(gray,binary,100,255,THRESH_BINARY_INV);
    // structuring element
    Mat S = getStructuringElement(MORPH_RECT,Size(3,3));
    cout << S << endl;
    // dilation
    Mat dilated;
    dilate(binary,dilated,S);
    Mat dilated_gray;
    dilate(gray,dilated_gray,S);
    // erosion
    Mat eroded;
    erode(binary,eroded,S);
    Mat eroded_gray;
    erode(gray,eroded_gray,S);
    // show all
    imshow("original",binary);
    imshow("dilated",dilated);
    imshow("eroded",eroded);
    waitKey(0);

    imshow("original",gray);
    imshow("dilated",dilated_gray);
    imshow("eroded",eroded_gray);
    waitKey(0);

}


// week03
// mouse.cpp
// waitKey

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat A(300,200,CV_8UC3,Scalar(150,150,150));

void onMouse(int action, int x, int y, int flags, void *userdata)
{
    // if left button pressed
    if( action == EVENT_LBUTTONDOWN )
    {
        circle(A, Point(x,y), 5, Scalar(0,0,255), -1 );
    }
    if( action == EVENT_RBUTTONDOWN )
    {
        circle(A, Point(x,y), 5, Scalar(0,255,255), -1);
    }

    imshow("window", A);
}

int main(){
    namedWindow("window");
    setMouseCallback("window",onMouse);
    imshow("window",A);
    waitKey(0);
}

// week03
// read_write_show.cpp
// reads, writes, shows images

#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    Mat image1 = imread("../images/butterfly.png",IMREAD_COLOR); // read colorful (default)
    Mat image2 = imread("../images/butterfly.png",IMREAD_GRAYSCALE); // read and convert to gray

    namedWindow("colorful"); // create window "colorful"
    namedWindow("grayscale"); // create window "grayscale"
    waitKey(0); // wait for user to press a key

    imshow("colorful",image1);
    imshow("grayscale",image2);

    waitKey(0); // wait for user to press a key

    imwrite("../butterfly_gray.png",image2); // save image

    return 0;
}

// week03
// thresholding.cpp
// basic thresholding

#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    // original
    Mat sheet = imread("../images/answers.png");

    // converting to grayscale
    Mat gray_sheet;
    cvtColor(sheet,gray_sheet,COLOR_BGR2GRAY);

    // binary thresholding
    Mat bin_sheet;
    threshold(gray_sheet,bin_sheet,100,255,THRESH_BINARY);

    // inverse binary thresholding
    Mat bin_sheet_inv;
    threshold(gray_sheet,bin_sheet_inv,100,255,THRESH_BINARY_INV);

    // show all
    imshow("original",sheet);
    imshow("gray",gray_sheet);
    imshow("binary (t=100)",bin_sheet);
    imshow("inverse (t=100)",bin_sheet_inv);
    waitKey(0);
}

// week03
// trackbar.cpp
// trackbars and callback functions

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat image;
Mat result;

void on_threshold_change(int t, void* data) {
    threshold(image, result, t, 255, THRESH_BINARY);
    imshow("result", result);
}

int main(){
    image = imread("../images/sdu.png",IMREAD_GRAYSCALE);
    imshow("window",image);

    namedWindow("result");
    int T = 100;
    createTrackbar("threshold", "result", &T, 255, on_threshold_change);
    on_threshold_change(T, NULL);
    waitKey(0);
}

// week03
// transform_affine.cpp
// translation, scaling, rotation, shearing

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

Mat image;
Mat result;
vector <Point2f> src;
vector <Point2f> dst;

void collect_src_points(int action, int x, int y, int flags, void *userdata)
{
    if( action == EVENT_LBUTTONDOWN && src.size() < 3)
    {
        circle(image, Point(x,y), 3, Scalar(0,0,255), -1);
        switch(src.size()){
            case 0: putText(image,"A",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 1: putText(image,"B",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 2: putText(image,"C",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2);break;
        }
        src.push_back(Point(x,y));
        imshow("original", image);
    }
}

void collect_dst_points(int action, int x, int y, int flags, void *userdata)
{
    if( action == EVENT_LBUTTONDOWN && dst.size() < 3)
    {
        circle(result, Point(x,y), 3, Scalar(0,255,0), -1);
        switch(dst.size()){
            case 0: putText(result,"A",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 1: putText(result,"B",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 2: putText(result,"C",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2);break;
        }
        dst.push_back(Point(x,y));
        imshow("result", result);
    }
}

int main(){
    image = imread("../images/obama.jpg");
    result.create(image.rows*2,image.cols*2,image.type());

    imshow("original",image);
    imshow("result",result);
    setMouseCallback("original",collect_src_points);
    setMouseCallback("result",collect_dst_points);
    waitKey(0);

    Mat T = getAffineTransform(src,dst);
    warpAffine(image,result,T,result.size(),INTER_CUBIC,BORDER_TRANSPARENT);

    cout << "T = " << endl << T << endl;
    imshow("result",result);
    waitKey(0);
}

// week03
// transform_perspective.cpp
// translation, scaling, rotation, shearing

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

Mat image;
Mat result;
vector <Point2f> src;
vector <Point2f> dst;

void collect_src_points(int action, int x, int y, int flags, void *userdata)
{
    if( action == EVENT_LBUTTONDOWN && src.size() < 4)
    {
        circle(image, Point(x,y), 3, Scalar(0,0,255), -1 );
        switch(src.size()){
            case 0: putText(image,"A",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 1: putText(image,"B",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 2: putText(image,"C",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2);break;
            case 3: putText(image,"D",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2);break;
        }
        src.push_back(Point(x,y));
        imshow("original", image);
    }
}

void collect_dst_points(int action, int x, int y, int flags, void *userdata)
{
    if( action == EVENT_LBUTTONDOWN && dst.size() < 4)
    {
        circle(result, Point(x,y), 3, Scalar(0,255,0), -1);
        switch(dst.size()){
            case 0: putText(result,"A",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 1: putText(result,"B",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2); break;
            case 2: putText(result,"C",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2);break;
            case 3: putText(result,"D",Point(x,y),FONT_HERSHEY_PLAIN,2,Scalar(0,255,255),2);break;
        }
        dst.push_back(Point(x,y));
        imshow("result", result);
    }
}

int main(){
    image = imread("../images/obama.jpg");
    result.create(image.rows*2,image.cols*2,image.type());

    imshow("original",image);
    imshow("result",result);
    setMouseCallback("original",collect_src_points);
    setMouseCallback("result",collect_dst_points);
    waitKey(0);

    Mat T = getPerspectiveTransform(src,dst);
    //Mat T = findHomography(src,dst);
    warpPerspective(image,result,T,result.size(),INTER_CUBIC,BORDER_TRANSPARENT);
    cout << "T = " << endl << T << endl;
    imshow("result",result);
    waitKey(0);
}

// week03
// transform_scaled_rotation.cpp
// rotation + scaling

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat image;
Mat result;
Point center;
int angle = 10;
int scale = 50;

void on_change(int, void* data) {
    Mat T = getRotationMatrix2D(center,angle,scale/100.0);
    cout << "angle = " << angle << endl;
    cout << "T = " << endl << T << endl;
    warpAffine(image,result,T,Size(),INTER_CUBIC,BORDER_CONSTANT,Scalar(255,255,0));
    imshow("result", result);
}

int main(){
    image = imread("../images/putin.jpg");
    center = Point(image.cols/2,image.rows/2);
    imshow("window",image);
    namedWindow("result");
    createTrackbar("angle", "result", &angle, 360, on_change);
    createTrackbar("scale", "result", &scale, 200, on_change);
    on_change(0,NULL);
    waitKey(0);
}


// week03
// transform_translation.cpp
// translation (manually, without sophisticated geometry)

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat image;
Mat result;
int tX = 10;
int tY = 30;

int oldtX = 10;
int oldtY = 30;

const Scalar WHITE(255,255,255);

void on_change(int, void* data) {
    Mat crop = result(Rect(tX,tY,image.cols,image.rows));
    Mat old_crop = result(Rect(oldtX,oldtY,image.cols,image.rows));
    old_crop.setTo(WHITE);
    oldtX = tX; // remember old position to make it WHITE
    oldtY = tY;
    image.copyTo(crop);
    imshow("result", result);
}

int main(){
    image = imread("../images/obama.jpg");
    result.create(image.rows*2, image.cols*2, image.type());
    result.setTo(WHITE);

    namedWindow("result");
    createTrackbar("tX", "result", &tX, image.cols, on_change);
    createTrackbar("tY", "result", &tY, image.rows, on_change);
    on_change(0,NULL);
    waitKey(0);
}

// week03
// type_conversion.cpp
// data type conversion of an image
// 0 .. 255 => 0.0 .. 1.0
// OpenCV assumes floating point image is in the range of 0..1
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(void)
{
    Mat image,image2,image3;
    image = imread("../images/sdu.png");

    double scale = 1/255.0;
    double shift = 0;
    // converting from uchar to 32-bit float
    image.convertTo(image2, CV_32FC3, scale, shift);
    // converting from float to uchar
    image.convertTo(image3, CV_32FC3, 1.0, shift);

    imshow("image",image);
    imshow("image2",image2);
    imshow("image3",image3); // 0.0 .. 255.0 is almost oversaturated

    waitKey(0);
}

// week03
// video_read_show.cpp
// how to read video from file/camera

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(){
    VideoCapture vin("../videos/bike.avi");
    Mat frame;

    cout << "FPS = " << vin.get(CAP_PROP_FPS) << endl;
    cout << "COUNT = " << vin.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "WIDTH = " << vin.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "HEIGHT = " << vin.get(CAP_PROP_FRAME_HEIGHT) << endl;

    vin >> frame;
    while (!frame.empty()){
        imshow("video",frame);
        waitKey(33);
        vin >> frame;
    }

    return 0;
}


// week03
// video_write.cpp
// creating a sequence of frames and storing them to video file

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int main(){
    Mat obama = imread("../images/obama.jpg");
    Mat putin = imread("../images/putin.jpg");
    Mat mix = obama/2 + putin/2;
    imshow("obama",obama);
    imshow("putin",putin);
    imshow("(obama + putin)/2",mix);
    cout << obama.size() << endl;
    cout << putin.size() << endl;
    waitKey(0);

    VideoWriter vout("../videos/obama2putin.mp4",CV_FOURCC('M','J','P','G'),25,putin.size());
    double alpha = 1.0;
    while(alpha > 0.0)
    {
        Mat mix = obama*alpha + putin*(1 - alpha);
        alpha -= 0.005;
        imshow( "mix", mix);
        vout << mix;
        waitKey(20);
    }
    vout.release();
    return 0;
}


// week04
// hsv_color.cpp
// HSV color space

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat bgr_image = imread("../images/capsicam.jpg");\
    Mat hsv_image;
    cvtColor(bgr_image,hsv_image,COLOR_BGR2HSV);
    Mat hsv[3];
    split(hsv_image,hsv);
    imshow("image",hsv_image);
    imshow("H",hsv[0]);
    imshow("S",hsv[1]);
    imshow("V",hsv[2]);
    waitKey(0);
    return 0;
}

// week04
// ycrcb_color.cpp
// LAB color space

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat bgr_image = imread("../images/capsicam.jpg");\
    Mat lab_image;
    cvtColor(bgr_image,lab_image,COLOR_BGR2Lab);
    Mat lab[3];
    split(lab_image,lab);
    imshow("image",lab_image);
    imshow("L",lab[0]);
    imshow("A",lab[1]);
    imshow("B",lab[2]);
    waitKey(0);
    return 0;
}

// week04
// rgb_color.cpp
// RGB color space 

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat image = imread("../images/capsicam.jpg");
    Mat bgr[3];
    split(image,bgr);

    imshow("image",image);
    imshow("blue",bgr[0]);
    imshow("green",bgr[1]);
    imshow("red",bgr[2]);

    waitKey(0);
    return 0;
}

// week04
// ycrcb_color.cpp
// YCrCb color space

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat bgr_image = imread("../images/capsicam.jpg");\
    Mat ycrcb_image;
    cvtColor(bgr_image,ycrcb_image,COLOR_BGR2YCrCb);
    Mat ycrcb[3];
    split(ycrcb_image,ycrcb);
    imshow("image",ycrcb_image);
    imshow("Y",ycrcb[0]);
    imshow("Cr",ycrcb[1]);
    imshow("Cb",ycrcb[2]);
    waitKey(0);
    return 0;
}





