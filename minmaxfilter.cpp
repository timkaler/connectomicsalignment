#include "common.h"

#define MACRO_CONCAT(a,b) MACRO_CONCAT_(a,b)
#define MACRO_CONCAT_(a,b) a##b

#define MFILTER_DATA_WIDTH 16
#define V256_MAX MACRO_CONCAT(_mm256_max_epi,MFILTER_DATA_WIDTH)
#define V256_MIN MACRO_CONCAT(_mm256_min_epi,MFILTER_DATA_WIDTH)

namespace maxFilter3x3
{
	void columnMaxFilter(std::vector< cv::Mat > &data)
	{
		assert(sizeof(sift_wt_elem)*MFILTER_DATA_WIDTH*8==256);
		int n=data.size(), rows=data[0].rows, cols=data[0].cols;
		assert(n>=3 && rows>=3 && cols>=3);
		rep(i,0,n-1)
			rep(j,0,rows-1)
			{
				sift_wt* a = data[i].ptr<sift_wt>(j);
				const sift_wt* end = a+cols;
				
				__m256i m1=_mm256_loadu_si256((const __m256i*)a);
				a++;
				__m256i m2=_mm256_loadu_si256((const __m256i*)a);
				
				while (a+2 < end)
				{
					//invarient: a points to m2, *a not done yet
					// 1 2(a) 3 4 
					__m256i m3=_mm256_loadu_si256((const __m256i*)(a+1));
					__m256i m4=_mm256_loadu_si256((const __m256i*)(a+2));
					__m256i m23=V256_MAX(m2, m3);
					_mm256_storeu_si256((__m256i*)a,V256_MAX(m23, m1));
					_mm256_storeu_si256((__m256i*)(a+1),V256_MAX(m23, m4));
					a+=2;
					
					m1=m3; m2=m4;
					/*
					// 3 4(a) 1 2
					m1=_mm256_loadu_si256((const __m256i*)(a+1));
					m2=_mm256_loadu_si256((const __m256i*)(a+2));
					_mm256i m41=V256_MAX(m4, m1);
					_mm256_storeu_si256((__m256i*)a,V256_MAX(m3, m41));
					_mm256_storeu_si256((__m256i*)(a+1),V256_MAX(m41, m2));
					a+=2;
					*/
				}
				
				while (a+1 < end)
				{
					__m256i m3=_mm256_loadu_si256((const __m256i*)(a+1));
					_mm256_storeu_si256((__m256i*)a,V256_MAX(V256_MAX(m1, m2), m3));
					a++; m1=m2; m2=m3;
				}
			}
	}
	
	void rowMaxFilter(std::vector< cv::Mat > &data)
	{
		assert(sizeof(sift_wt_elem)*MFILTER_DATA_WIDTH*8==256);
		int n=data.size(), rows=data[0].rows, cols=data[0].cols;
		assert(n>=3 && rows>=3 && cols>=3);
		rep(i,0,n-1)
		{
			// row 0 will be used as buffer
			int j=1;
			while (j+2<rows)
			{
				sift_wt* buffer = data[i].ptr<sift_wt>(0);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i].ptr<sift_wt>(j+1);
				sift_wt* c = data[i].ptr<sift_wt>(j+2);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					__m256i m4=_mm256_loadu_si256((const __m256i*)c);
					__m256i m23=V256_MAX(m2,m3);
					_mm256_storeu_si256((__m256i*)a, V256_MAX(m23, m1));
					_mm256_storeu_si256((__m256i*)b, V256_MAX(m23, m4));
					_mm256_storeu_si256((__m256i*)buffer, m3);
					buffer++; a++; b++; c++;
				}
				j+=2;
			}
			
			while (j+1 < rows)
			{
				sift_wt* buffer = data[i].ptr<sift_wt>(0);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i].ptr<sift_wt>(j+1);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					_mm256_storeu_si256((__m256i*)a, V256_MAX(V256_MAX(m1, m2), m3));
					_mm256_storeu_si256((__m256i*)buffer, m2);
					buffer++; a++; b++;
				}
			}
		}
	}
	
	void levelMaxFilter(std::vector< cv::Mat > &data)
	{
		assert(sizeof(sift_wt_elem)*MFILTER_DATA_WIDTH*8==256);
		int n=data.size(), rows=data[0].rows, cols=data[0].cols;
		assert(n>=3 && rows>=3 && cols>=3);
		rep(j,0,rows-1)
		{
			//matrix 0 will be used as buffer
			int i=1;
			while (i+2<n)
			{
				sift_wt* buffer = data[0].ptr<sift_wt>(j);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i+1].ptr<sift_wt>(j);
				sift_wt* c = data[i+2].ptr<sift_wt>(j);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					__m256i m4=_mm256_loadu_si256((const __m256i*)c);
					__m256i m23=V256_MAX(m2,m3);
					_mm256_storeu_si256((__m256i*)a, V256_MAX(m23, m1));
					_mm256_storeu_si256((__m256i*)b, V256_MAX(m23, m4));
					_mm256_storeu_si256((__m256i*)buffer, m3);
					buffer++; a++; b++; c++;
				}
				i+=2;
			}
			
			while (i+1 < n)
			{
				sift_wt* buffer = data[0].ptr<sift_wt>(j);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i+1].ptr<sift_wt>(j);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					_mm256_storeu_si256((__m256i*)a, V256_MAX(V256_MAX(m1, m2), m3));
					_mm256_storeu_si256((__m256i*)buffer, m2);
					buffer++; a++; b++;
				}
			}
		}
	}
	
	void applyMaxFilter(std::vector< cv::Mat > &data)
	{
		columnMaxFilter(data);
		rowMaxFilter(data);
		levelMaxFilter(data);
	}
}

namespace minFilter3x3
{
	void columnMinFilter(std::vector< cv::Mat > &data)
	{
		assert(sizeof(sift_wt_elem)*MFILTER_DATA_WIDTH*8==256);
		int n=data.size(), rows=data[0].rows, cols=data[0].cols;
		assert(n>=3 && rows>=3 && cols>=3);
		rep(i,0,n-1)
			rep(j,0,rows-1)
			{
				sift_wt* a = data[i].ptr<sift_wt>(j);
				const sift_wt* end = a+cols;
				
				__m256i m1=_mm256_loadu_si256((const __m256i*)a);
				a++;
				__m256i m2=_mm256_loadu_si256((const __m256i*)a);
				
				while (a+2 < end)
				{
					//invarient: a points to m2, *a not done yet
					// 1 2(a) 3 4 
					__m256i m3=_mm256_loadu_si256((const __m256i*)(a+1));
					__m256i m4=_mm256_loadu_si256((const __m256i*)(a+2));
					__m256i m23=V256_MIN(m2, m3);
					_mm256_storeu_si256((__m256i*)a,V256_MIN(m23, m1));
					_mm256_storeu_si256((__m256i*)(a+1),V256_MIN(m23, m4));
					a+=2;
					
					m1=m3; m2=m4;
					/*
					// 3 4(a) 1 2
					m1=_mm256_loadu_si256((const __m256i*)(a+1));
					m2=_mm256_loadu_si256((const __m256i*)(a+2));
					_mm256i m41=V256_MIN(m4, m1);
					_mm256_storeu_si256((__m256i*)a,V256_MIN(m3, m41));
					_mm256_storeu_si256((__m256i*)(a+1),V256_MIN(m41, m2));
					a+=2;
					*/
				}
				
				while (a+1 < end)
				{
					__m256i m3=_mm256_loadu_si256((const __m256i*)(a+1));
					_mm256_storeu_si256((__m256i*)a,V256_MIN(V256_MIN(m1, m2), m3));
					a++; m1=m2; m2=m3;
				}
			}
	}
	
	void rowMinFilter(std::vector< cv::Mat > &data)
	{
		assert(sizeof(sift_wt_elem)*MFILTER_DATA_WIDTH*8==256);
		int n=data.size(), rows=data[0].rows, cols=data[0].cols;
		assert(n>=3 && rows>=3 && cols>=3);
		rep(i,0,n-1)
		{
			// row 0 will be used as buffer
			int j=1;
			while (j+2<rows)
			{
				sift_wt* buffer = data[i].ptr<sift_wt>(0);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i].ptr<sift_wt>(j+1);
				sift_wt* c = data[i].ptr<sift_wt>(j+2);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					__m256i m4=_mm256_loadu_si256((const __m256i*)c);
					__m256i m23=V256_MIN(m2,m3);
					_mm256_storeu_si256((__m256i*)a, V256_MIN(m23, m1));
					_mm256_storeu_si256((__m256i*)b, V256_MIN(m23, m4));
					_mm256_storeu_si256((__m256i*)buffer, m3);
					buffer++; a++; b++; c++;
				}
				j+=2;
			}
			
			while (j+1 < rows)
			{
				sift_wt* buffer = data[i].ptr<sift_wt>(0);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i].ptr<sift_wt>(j+1);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					_mm256_storeu_si256((__m256i*)a, V256_MIN(V256_MIN(m1, m2), m3));
					_mm256_storeu_si256((__m256i*)buffer, m2);
					buffer++; a++; b++;
				}
			}
		}
	}
	
	void levelMinFilter(std::vector< cv::Mat > &data)
	{
		assert(sizeof(sift_wt_elem)*MFILTER_DATA_WIDTH*8==256);
		int n=data.size(), rows=data[0].rows, cols=data[0].cols;
		assert(n>=3 && rows>=3 && cols>=3);
		rep(j,0,rows-1)
		{
			//matrix 0 will be used as buffer
			int i=1;
			while (i+2<n)
			{
				sift_wt* buffer = data[0].ptr<sift_wt>(j);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i+1].ptr<sift_wt>(j);
				sift_wt* c = data[i+2].ptr<sift_wt>(j);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					__m256i m4=_mm256_loadu_si256((const __m256i*)c);
					__m256i m23=V256_MIN(m2,m3);
					_mm256_storeu_si256((__m256i*)a, V256_MIN(m23, m1));
					_mm256_storeu_si256((__m256i*)b, V256_MIN(m23, m4));
					_mm256_storeu_si256((__m256i*)buffer, m3);
					buffer++; a++; b++; c++;
				}
				i+=2;
			}
			
			while (i+1 < n)
			{
				sift_wt* buffer = data[0].ptr<sift_wt>(j);
				sift_wt* a = data[i].ptr<sift_wt>(j);
				sift_wt* b = data[i+1].ptr<sift_wt>(j);
				const sift_wt* end = a + cols;
				while (a<end)
				{
					__m256i m1=_mm256_loadu_si256((const __m256i*)buffer);
					__m256i m2=_mm256_loadu_si256((const __m256i*)a);
					__m256i m3=_mm256_loadu_si256((const __m256i*)b);
					_mm256_storeu_si256((__m256i*)a, V256_MIN(V256_MIN(m1, m2), m3));
					_mm256_storeu_si256((__m256i*)buffer, m2);
					buffer++; a++; b++;
				}
			}
		}
	}
	
	void applyMinFilter(std::vector< cv::Mat > &data)
	{
		columnMinFilter(data);
		rowMinFilter(data);
		levelMinFilter(data);
	}
}
