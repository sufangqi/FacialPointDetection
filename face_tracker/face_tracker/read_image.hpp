#ifndef _READ_IMAGE_H__
#define _READ_IMAGE_H__
#include<opencv2\opencv.hpp>
#include<afx.h>
#include<atltypes.h>
#include<atlconv.h>
#include<iostream>
#include<fstream>
#include<windows.h> 
#include<string>
using namespace cv;
using namespace std;
vector<CString> m_vecImage;
int CountFile(CString strPath)
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
BOOL DirExistW(LPCWSTR pszDirName)
{
	CFileFind FileFind;
	CString sFilePath(pszDirName);
	if(!FileFind.FindFile(sFilePath))  //路径不存在则创建该路径
	{
		return FALSE;  
	}
	return TRUE;   
}
#endif