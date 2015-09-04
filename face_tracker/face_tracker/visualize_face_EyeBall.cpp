#include<opencv2\opencv.hpp>
#include "face_Keypoint.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
#define fl at<float>
//==============================================================================

//==============================================================================
int main(int argc,char** argv)
{
  //load detector model
	CascadeClassifier face_classfier;
	face_classfier.load("data/haarcascade_frontalface_alt2.xml");
	  if( face_classfier.empty())
	  {
	   cerr<<"error:couldn't load face detector"<<endl;
	   exit(1);
	  }
  //face_tracker tracker = load_ft<face_tracker>("data/train_face_tracker.yaml");

  //create tracker parameters
  /*face_tracker_params p; p.robust = false;
  p.ssize.resize(3);
  p.ssize[0] = Size(21,21);
  p.ssize[1] = Size(11,11);
  p.ssize[2] = Size(5,5);*/

  //open video stream
  VideoCapture cam; 
  if(argc > 2)cam.open(argv[2]); else cam.open(0);
  if(!cam.isOpened()){
    cout << "Failed opening video file." << endl;
	return 0;
  }
  vector<Rect> faces;
  //detect until user quits
  namedWindow("faceDemo");
  while(cam.get(CV_CAP_PROP_POS_AVI_RATIO) < 0.999999){
    Mat im; cam >> im; 

	Mat gray; if(im.channels()==1)gray = im; else cvtColor(im,gray,CV_RGB2GRAY);
	face_classfier.detectMultiScale(gray,faces,1.1,5,0
                |CV_HAAR_FIND_BIGGEST_OBJECT,Size(50,50));



//获取人脸框之后，调用下面的函数tracker.track(im,faces[0])，其中im为原图像，face[0]为人脸矩形框
//tracker.LeftEye为左眼，tracker.RightEye为右眼
    face_tracker tracker;
	if(faces.size()>0)
	{
       if(tracker.track(im,faces[0]))
	   circle(im,tracker.LeftEye,2,CV_RGB(255,0,0),2,CV_AA);
	   circle(im,tracker.RightEye,2,CV_RGB(255,0,0),2,CV_AA);
	}
    //tracker.timer.display_fps(im,Point(1,im.rows-1));
	flip(im,im,1);
    imshow("faceDemo",im);
/*************************************************************************************************/



    int c = waitKey(10);
    if(c == 'q')break;
    else if(c == 'd')tracker.reset();
  }
  destroyWindow("faceDemo"); cam.release(); return 0;
}
//==============================================================================
