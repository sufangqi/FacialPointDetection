#include "face_detector.hpp"
//#include "ft.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#define fl at<float>
face_detector::face_detector()
{
  detector_offset[0]=2.8527858667075634e-003;
  detector_offset[1]=8.3034694194793701e-002;
  detector_offset[2]=2.6335916519165039e+000;
  reference=(Mat_<float>(152,1)<<-1.65296733e-001, -6.72742724e-002, -1.64978012e-001,
          -1.60028860e-002, -1.59681112e-001, 3.06803677e-002,
          -1.48966506e-001, 7.77749792e-002, -1.24983452e-001,
          1.26513779e-001, -8.93540084e-002, 1.57958075e-001,
          -4.43429351e-002, 1.80292770e-001, 3.69058293e-003,
          1.87625960e-001, 4.96583730e-002, 1.77637532e-001,
          8.52332041e-002, 1.52668282e-001, 1.12488709e-001,
          1.19925812e-001, 1.29193261e-001, 7.78981969e-002,
          1.39485851e-001, 2.71101370e-002, 1.43883228e-001,
          -1.92882363e-002, 1.40999287e-001, -6.84657246e-002,
          1.19044304e-001, -9.99365672e-002, 9.28181186e-002,
          -1.23049714e-001, 6.12762794e-002, -1.22011289e-001,
          3.19975279e-002, -1.02776863e-001, 6.29857406e-002,
          -1.03805214e-001, 9.15271193e-002, -1.04962341e-001,
          -1.25131428e-001, -9.67217013e-002, -9.65818763e-002,
          -1.19907878e-001, -6.20788746e-002, -1.20543532e-001,
          -3.02504748e-002, -1.02868527e-001, -6.36558458e-002,
          -1.02921374e-001, -9.46720466e-002, -1.03235006e-001,
          -9.85844508e-002, -6.66727424e-002, -7.25809559e-002,
          -8.08793455e-002, -4.51488234e-002, -6.66077361e-002,
          -7.21861124e-002, -5.78657575e-002, -7.14182556e-002,
          -7.04673752e-002, 9.58473682e-002, -6.83981627e-002,
          7.09440187e-002, -8.18853155e-002, 4.41815518e-002,
          -6.75117970e-002, 7.11317956e-002, -5.97099811e-002,
          7.11636916e-002, -7.12023601e-002, -1.64285246e-002,
          -6.91920966e-002, -2.04179827e-002, -2.46094316e-002,
          -4.48611304e-002, 2.64196726e-003, -3.28396074e-002,
          2.25352161e-002, 5.62006980e-003, 2.91545987e-002,
          3.98004539e-002, 2.18730439e-002, 4.93510254e-002,
          2.72505963e-003, 2.49567851e-002, -2.82830410e-002,
          1.80943906e-002, -7.05796480e-002, -1.92170665e-002,
          1.49217024e-002, 2.95450818e-002, 1.42575083e-002,
          -5.81411533e-002, 7.79130906e-002, -3.26500349e-002,
          6.06753826e-002, -8.79942533e-003, 5.48121259e-002,
          5.81641309e-003, 5.67407981e-002, 1.87939107e-002,
          5.38917482e-002, 4.12441418e-002, 5.95650524e-002,
          6.18709996e-002, 7.53120854e-002, 4.81018461e-002,
          9.31589827e-002, 2.86645647e-002, 1.02873348e-001,
          5.57450717e-003, 1.05887726e-001, -2.02799886e-002,
          1.04065165e-001, -4.25798595e-002, 9.54934508e-002,
          -2.31898557e-002, 8.37075114e-002, 5.22858230e-003,
          8.69224668e-002, 3.20928656e-002, 8.26704055e-002,
          3.20016704e-002, 7.07344562e-002, 5.30683855e-003,
          7.13810772e-002, -2.31130049e-002, 7.16408268e-002,
          5.30222896e-003, 7.88102746e-002, 6.92879222e-003,
          -2.49633740e-004, -8.59698951e-002, -7.41588846e-002,
          -5.92449121e-002, -7.41239339e-002, -5.90228215e-002,
          -6.25857040e-002, -8.57535303e-002, -6.26327470e-002,
          8.30119699e-002, -7.55266249e-002, 5.71743138e-002,
          -7.50936270e-002, 5.72569706e-002, -6.40020370e-002,
          8.31122473e-002, -6.44418448e-002);
  /*detector.load("data/haarcascade_frontalface_alt2.xml");
  if(detector.empty())
  {
   cerr<<"error:couldn't load face detector"<<endl;
   exit(1);
  }*/
}
vector<Point2f>
face_detector::
detect_Rect(const Mat &im,
       const float scaleFactor,
       const int minNeighbours,
       const Size minSize,Rect&face_Rect)
{
   //convert image to greyscale
  Mat gray; if(im.channels()==1)gray = im; else cvtColor(im,gray,CV_RGB2GRAY);

  //detect faces
  Mat eqIm=gray.clone();
  if(face_Rect.area() ==0){return vector<Point2f>();}
  //predict face placement
  if(face_Rect.area()==0)
  {return vector<Point2f>();}
  Rect R = face_Rect; Vec3f scale = detector_offset*R.width;
  int n = reference.rows/2; vector<Point2f> p(n);
  for(int i = 0; i < n; i++){
    p[i].x = scale[2]*reference.fl(2*i  ) + R.x + 0.5 * R.width  + scale[0];
    p[i].y = scale[2]*reference.fl(2*i+1) + R.y + 0.5 * R.height + scale[1];
  }return p;
}
//==============================================================================
vector<Point2f>
face_detector::
detect(const Mat &im,
       const float scaleFactor,
       const int minNeighbours,
       const Size minSize)
{
  //convert image to greyscale
  Mat gray; if(im.channels()==1)gray = im; else cvtColor(im,gray,CV_RGB2GRAY);

  //detect faces
  vector<Rect> faces; Mat eqIm=gray.clone();
  //equalizeHist(gray,eqIm);
 //   int scaledWidth=320;
	//Mat gray_binary,display_Small,gray_binary_copy;
	//int threshold_lower=140,threshold_lower0=140;
	//int threshold_higher=200,threshold_higher0=200;
	//vector<vector<Point>> contours;
 //   vector<Vec4i> hierarchy;
 //   Mat  displayedFrame=im.clone();
	//float scales=displayedFrame.cols/scaledWidth;
	//if(displayedFrame.cols>scaledWidth)
	//	{
	//		int scaledHeight=cvRound(displayedFrame.rows/scales);
	//		resize(displayedFrame,display_Small,Size(scaledWidth,scaledHeight));
	//	}
	//	else
	//	{
	//		display_Small=displayedFrame;
	//	}
	//    skin_detection(display_Small,gray_binary,threshold_lower,threshold_higher,threshold_lower0,threshold_higher0);
	//	imshow("ss",gray_binary);
	//	gray_binary.copyTo(gray_binary_copy);
	//	Mat dst = Mat::zeros(gray_binary.rows, gray_binary.cols, CV_8UC3);
 //       findContours(gray_binary, contours, hierarchy,
 //       CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
	//	if(contours.size()>0)
 //      {
	//	   for( int idx = 0; idx <contours.size();idx ++ )
	//		{
	//			double tmpara;
	//			tmpara=contourArea(contours[idx]);
	//			if(tmpara>50)
	//			{
	//				Rect r0;
	//				r0 = boundingRect(Mat(contours[idx]));
	//				if(displayedFrame.cols>scaledWidth)
	//				//if(displayedFrame.cols>scaledWidth)
	//				  {
	//			     
	//						  r0.x=cvRound(r0.x*scales);
	//						  r0.y=cvRound(r0.y*scales);
	//						  r0.width=cvRound(r0.width*scales);
	//						  r0.height=cvRound(r0.height*scales);
	//			              
	//				  }
	//				Mat FaceImgRoi=im(r0);
	//				detector.detectMultiScale(FaceImgRoi,faces,scaleFactor,minNeighbours,0
 //               |CV_HAAR_FIND_BIGGEST_OBJECT
 //               ,minSize);
	//				if(faces.size()>0)
	//				{
	//					break;
	//				}
	//				faces[0]=Rect(faces[0].x+r0.x,faces[0].y+r0.y,faces[0].width,faces[0].height);
	//			}
	//	    }

	//	}

  detector.detectMultiScale(eqIm,faces,scaleFactor,minNeighbours,0
              |CV_HAAR_FIND_BIGGEST_OBJECT
               ,minSize);

  if(faces.size() < 1){return vector<Point2f>();}
  //predict face placement


  Rect R = faces[0]; Vec3f scale = detector_offset*R.width;
  int n = reference.rows/2; vector<Point2f> p(n);
  for(int i = 0; i < n; i++){
    p[i].x = scale[2]*reference.fl(2*i  ) + R.x + 0.5 * R.width  + scale[0];
    p[i].y = scale[2]*reference.fl(2*i+1) + R.y + 0.5 * R.height + scale[1];
  }return p;
}
//==============================================================================
//void
//face_detector::
//train(ft_data &data,
//      const string fname,
//      const Mat &ref,
//      const bool mirror,
//      const bool visi,
//      const float frac,
//      const float scaleFactor,
//      const int minNeighbours,
//      const Size minSize)
//{
//  detector.load(fname.c_str()); detector_fname = fname; reference = ref.clone();
//  vector<float> xoffset(0),yoffset(0),zoffset(0);
//  for(int i = 0; i < data.n_images(); i++){
//    Mat im = data.get_image(i,0); if(im.empty())continue;
//    vector<Point2f> p = data.get_points(i,false); int n = p.size();
//    Mat pt = Mat(p).reshape(1,2*n);
//    vector<Rect> faces; Mat eqIm; equalizeHist(im,eqIm);
//    detector.detectMultiScale(eqIm,faces,scaleFactor,minNeighbours,0
//                  |CV_HAAR_FIND_BIGGEST_OBJECT
//                  |CV_HAAR_SCALE_IMAGE,minSize);
//    if(faces.size() >= 1){
//      if(visi){
//    Mat I; cvtColor(im,I,CV_GRAY2RGB);
//    for(int i = 0; i < n; i++)circle(I,p[i],1,CV_RGB(0,255,0),2,CV_AA);
//   
//      }
//      //check if enough points are in detected rectangle
//      if(this->enough_bounded_points(pt,faces[0],frac)){
//    Point2f center = this->center_of_mass(pt); float w = faces[0].width;
//    xoffset.push_back((center.x - (faces[0].x+0.5*faces[0].width ))/w);
//    yoffset.push_back((center.y - (faces[0].y+0.5*faces[0].height))/w);
//    zoffset.push_back(this->calc_scale(pt)/w);
//      }
//    }
//    if(mirror){
//      im = data.get_image(i,1); if(im.empty())continue;
//      p = data.get_points(i,true);
//      pt = Mat(p).reshape(1,2*n);
//      //equalizeHist(im,eqIm);
//	  eqIm=im.clone();
//      detector.detectMultiScale(eqIm,faces,scaleFactor,minNeighbours,0
//                  |CV_HAAR_FIND_BIGGEST_OBJECT
//                ,minSize);
//      if(faces.size() >= 1){
//    if(visi){
//      Mat I; cvtColor(im,I,CV_GRAY2RGB);
//      for(int i = 0; i < n; i++)circle(I,p[i],1,CV_RGB(0,255,0),2,CV_AA);
//      rectangle(I,faces[0].tl(),faces[0].br(),CV_RGB(255,0,0),3);
//      imshow("face detector training",I); waitKey(10);
//    }
//    //check if enough points are in detected rectangle
//    if(this->enough_bounded_points(pt,faces[0],frac)){
//      Point2f center = this->center_of_mass(pt); float w = faces[0].width;
//      xoffset.push_back((center.x - (faces[0].x+0.5*faces[0].width ))/w);
//      yoffset.push_back((center.y - (faces[0].y+0.5*faces[0].height))/w);
//      zoffset.push_back(this->calc_scale(pt)/w);
//    }
//      }
//    }
//  }
//  //choose median value
//  Mat X = Mat(xoffset),Xsort,Y = Mat(yoffset),Ysort,Z = Mat(zoffset),Zsort;
//  cv::sort(X,Xsort,CV_SORT_EVERY_COLUMN|CV_SORT_ASCENDING); int nx = Xsort.rows;
//  cv::sort(Y,Ysort,CV_SORT_EVERY_COLUMN|CV_SORT_ASCENDING); int ny = Ysort.rows;
//  cv::sort(Z,Zsort,CV_SORT_EVERY_COLUMN|CV_SORT_ASCENDING); int nz = Zsort.rows;
//  detector_offset = Vec3f(Xsort.fl(nx/2),Ysort.fl(ny/2),Zsort.fl(nz/2)); return;
//}
//==============================================================================
bool
face_detector::
enough_bounded_points(const Mat &pts,
              const Rect R,
              const float frac)
{
  int n = pts.rows/2,m = 0;
  for(int i = 0; i < n; i++){
    if((pts.fl(2*i  ) >= R.x) && (pts.fl(2*i  ) <= R.x + R.width) &&
       (pts.fl(2*i+1) >= R.y) && (pts.fl(2*i+1) <= R.y + R.height))m++;
  }
  if(float(m)/n >= frac)return true; else return false;
}
//==============================================================================
Point2f
face_detector::
center_of_mass(const Mat &pts)
{
  float mx = 0,my = 0; int n = pts.rows/2;
  for(int i = 0; i < n; i++){
    mx += pts.fl(2*i); my += pts.fl(2*i+1);
  }return Point2f(mx/n,my/n);
}
//==============================================================================
float 
face_detector::
calc_scale(const Mat &pts)
{
  Point2f c = this->center_of_mass(pts); int n = pts.rows/2;
  Mat p(2*n,1,CV_32F);
  for(int i = 0; i < n; i++){
    p.fl(2*i  ) = pts.fl(2*i  ) - c.x;
    p.fl(2*i+1) = pts.fl(2*i+1) - c.y;
  }return reference.dot(p)/reference.dot(reference);
}
//==============================================================================
//void 
//face_detector::
//write(FileStorage &fs) const
//{
//  assert(fs.isOpened()); 
//  fs << "{"
//     << "fname"     << detector_fname
//     << "x offset"  << detector_offset[0]
//     << "y offset"  << detector_offset[1]
//     << "z offset"  << detector_offset[2]
//     << "reference" << reference
//     << "}";
//}
//==============================================================================
//void 
//face_detector::
//read(const FileNode& node)
//{
//  assert(node.type() == FileNode::MAP);
//  //node["fname"]     >> detector_fname;
//  node["x offset"]  >> detector_offset[0];
//  node["y offset"]  >> detector_offset[1];
//  node["z offset"]  >> detector_offset[2];
//  node["reference"] >> reference;
//  detector.load("data/haarcascade_frontalface_alt2.xml");
//  if(detector.empty())
//  {
//   cerr<<"error:couldn't load face detector"<<endl;
//   exit(1);
//  }
//}
//==============================================================================
