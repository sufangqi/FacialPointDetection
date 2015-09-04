#include<opencv2\opencv.hpp>
#include "face_Keypoint.hpp"
#include <opencv2/highgui/highgui.hpp>
#include<afx.h>
#include<atltypes.h>
#include<iostream>
#include<string>
using namespace cv;
#define fl at<float>
using namespace cv;
using namespace std;
vector<CString> m_vecImage;
int CountFile(CString strPath)//批量读入图片
{
	USES_CONVERSION;
	CFileFind finder;
	int nCount = 0;
	m_vecImage.clear();

	// build a string with wildcards 
	CString strWildcard(strPath); 
	strWildcard += _T("\\*");
	// start working for files 
	BOOL bWorking = finder.FindFile(strWildcard);
	while (bWorking) 
	{ 
		bWorking = finder.FindNextFile();
		if (finder.IsDots())
			continue; 
		if(!finder.IsDirectory()) 
		{ 
			nCount++; 
			CString strName = finder.GetFileName();
			CString strFormat = strName.Right(3);
			if (strFormat.CompareNoCase(_T("jpg")) == 0 ||
				strFormat.CompareNoCase(_T("JPG")) == 0 ||
				strFormat.CompareNoCase(_T("bmp")) == 0 ||
				strFormat.CompareNoCase(_T("BMP")) == 0 ||
				strFormat.CompareNoCase(_T("png")) == 0 ||
				strFormat.CompareNoCase(_T("PNG")) == 0 ||
				strFormat.CompareNoCase(_T("gif")) == 0 ||
				strFormat.CompareNoCase(_T("GIF")) == 0 )
			{
				m_vecImage.push_back(strName);
			}
			continue;
		}
	}
	finder.Close(); 
	return nCount;
}
BOOL DirExistW(LPCWSTR pszDirName)//确定当前文件夹是否存在
{
	CFileFind FileFind;
	CString sFilePath(pszDirName);
	if(!FileFind.FindFile(sFilePath))  //路径不存在则创建该路径
	{
		return FALSE;  
	}
	return TRUE;   
}
//==============================================================================
void skin_detection(Mat&input,Mat &dist,int lowerCr,int upperCr,int lowerCb,int upperCb)
{

	Mat img0;
	cvtColor(input,img0,CV_BGR2YCrCb);
	vector<Mat> Ycbcr;
	split(img0,Ycbcr);
	Mat imgCr=Ycbcr[1];
	Mat imgCb=Ycbcr[2];
	int nr=imgCr.rows;
    int nc=imgCb.cols;
	dist.create(imgCb.size(),CV_8UC1);
	 if(imgCb.isContinuous()&&dist.isContinuous())
    {
        nr=1;
        nc=nc*imgCb.rows*imgCb.channels();
    }
    for(int i=0;i<nr;i++)
    {
        const uchar*inDataCr=imgCr.ptr<uchar>(i);
		const uchar*inDataCb=imgCr.ptr<uchar>(i);
		      uchar*outData=dist.ptr<uchar>(i);
	   for(int j=0;j<nc;j++)
 {
	 if((*inDataCr<=upperCr&&*inDataCr>=lowerCr&&*inDataCb<=upperCb&&*inDataCb>=lowerCb))
		{
			*outData=255;
		}
		else
		{
			*outData=0;
		}
		*inDataCr++;
		*inDataCb++;
		*outData++;
	}
	}
	
	Mat element = getStructuringElement(MORPH_RECT,   
                                        Size(3,3)   
                                  );  
    //腐蚀操作  
	medianBlur(dist,dist,3);
	dilate(dist,dist,element,Point(-1, -1),3);
	//erode(dist,dist,element,Point(-1, -1),3);
	
	
}
//==============================================================================
int main(int argc,char** argv)
{
  //load detector model
	CascadeClassifier face_classfier;
    face_classfier.load("haarcascade_frontalface_alt2.xml");
	if(face_classfier.empty())
	{
		cerr<<"error:couldn't load face detector"<<endl;
		return 0;
	}
	/*if(argc!=3)
	{
       cout<<"error:"<<"# usage: ./face_tracker [image_str][image_save_str]" <<endl;
	   return 0;
	}
	string path_input=argv[1];
	CString m_strPath_A=path_input.c_str();;
	string save_path=argv[2];*/
		TCHAR exeFullPath[MAX_PATH + 1]={0}; // 获取当前exe所在文件夹路径 
                memset(exeFullPath,0,MAX_PATH);  
                GetModuleFileName(NULL,exeFullPath,MAX_PATH);
				(_tcsrchr(exeFullPath, _T('\\')))[1] = 0;
		CString str;
					for (int n=0;exeFullPath[n];n++)
					{
					if (exeFullPath[n]!=_T('\\'))
					{
					str +=exeFullPath[n] ;
					}
					else
					{
					str += _T("\\\\");
					}
					}
	CString m_strPath_A=str+ _T("\\Raw_image");
	CString save_path=str+ _T("\\detect_image");
	if (DirExistW(m_strPath_A))
		{
			int nCount = CountFile(m_strPath_A);
			if (nCount == 0)
			{
				std::cout<<("Raw_image路径下不存在JPG文件!")<<std::endl;
				return 0;
			}
		}
		else
		{
			std::cout<<("存储路径不存在，请在工程目录下新建Raw_image文件夹!");
			return 0;
		}
  for (vector<CString>::iterator iter=m_vecImage.begin(); iter!=m_vecImage.end(); ++iter)
  {
    CString strPath(m_strPath_A);
	strPath += _T("\\");
	strPath += *iter;
	USES_CONVERSION;
	string imgName=T2A(*iter);
	string data_path=T2A(strPath);
	Mat im=imread(data_path);
	if(im.empty())
	{
		cout<<"faild load image:"<<data_path<<endl;
		return 0;
	}
	if(im.empty())
		continue;
    vector<Rect> faces;
	Mat gray; if(im.channels()==1)gray = im; 
	else cvtColor(im,gray,CV_RGB2GRAY);
	face_classfier.detectMultiScale(gray,faces,1.1,5,0
                |CV_HAAR_FIND_BIGGEST_OBJECT,Size(30,30));

//获取人脸框之后，调用下面的函数tracker.track(im,faces[0])，其中im为原图像，face[0]为人脸矩形框
//tracker.LeftEye为左眼，tracker.RightEye为右眼
    face_tracker tracker;
	if(faces.size()>0)
	{
       tracker.track(im,faces[0]);
	   tracker.draw(im);
	   circle(im,tracker.LeftEye,faces[0].width/50,CV_RGB(255,0,0),faces[0].width/50,CV_AA);
	   circle(im,tracker.RightEye,faces[0].width/50,CV_RGB(255,0,0),faces[0].width/50,CV_AA);
	}
    //tracker.timer.display_fps(im,Point(1,im.rows-1));
	flip(im,im,1);
    imshow("faceDemo",im);
	string image_save_path_name=T2A(save_path);
	image_save_path_name=image_save_path_name+"\\"+imgName;
	imwrite(image_save_path_name,im);
/*************************************************************************************************/
    int c = waitKey(1);
    if(c == 'q')break;
    else if(c == 'd')tracker.reset();
  }
  destroyWindow("faceDemo"); return 0;
}
//==============================================================================
