#ifndef _FT_FACE_DETECTOR_HPP_
#define _FT_FACE_DETECTOR_HPP_
#include <opencv2/objdetect/objdetect.hpp>
#include "ft_data.hpp"
//==============================================================================
class face_detector{                       //face detector for initialisation
public:
  string detector_fname;                   //file containing cascade classifier
  Vec3f detector_offset;                   //offset from center of detection
  Mat reference;                           //reference shape
  CascadeClassifier detector;              //face detector
  face_detector();
  vector<Point2f>                          //points for detected face in image
  detect(const Mat &im,                    //image containing face
     const float scaleFactor = 1.1,    //scale increment
     const int minNeighbours = 5,      //minimum neighbourhood size
     const Size minSize = Size(30,30));//minimum detection window size
  vector<Point2f>
   detect_Rect(const Mat &im,
       const float scaleFactor,
       const int minNeighbours,
       const Size minSize,Rect&face_Rect);
  void
  train(ft_data &data,                     //training data
    const string fname,                //cascade detector
    const Mat &ref,                    //reference shape
    const bool mirror = false,         //mirror data?
    const bool visi = false,           //visualise training?
    const float frac = 0.8,            //fraction of points in detected rect
    const float scaleFactor = 1.1,     //scale increment
    const int minNeighbours = 2,       //minimum neighbourhood size
    const Size minSize = Size(30,30)); //minimum detection window size

  //void 
  //write(FileStorage &fs) const;            //file storage object to write to

  //void 
  //read(const FileNode& node);              //file storage node to read from

protected:
  bool                                     //are enough points bounded?
  enough_bounded_points(const Mat &pts,    //points to evaluate
            const Rect R,      //bounding rectangle
            const float frac); //fraction of points bounded

  Point2f                                  //center of mass
  center_of_mass(const Mat &pts);          //[x1;y1;...;xn;yn]

  float                                    //scaling from @reference to @pts
  calc_scale(const Mat &pts);              //[x1;y1;...;xn;yn]
};
//==============================================================================
#endif
