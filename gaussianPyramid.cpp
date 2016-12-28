#include "common.h"
#include "fasttime.h"

template <typename T> struct Matrix
{
	int row, col;
	T *data;
	Matrix() 
	{
		row=0; col=0;
		data = NULL;
	}
	Matrix(int _row, int _col)
	{
		row = _row; col = _col;
		data = new T[row*col];
	}
	void freeMemory()
	{
		row=0; col=0;
		if (data) 
		{
			delete [] data;
			data = NULL;
		}
	}

	void downsample2x(Matrix &dst) const
	{
		dst=Matrix(row/2, col/2);
		rep(i,0,dst.row-1)
			rep(j,0,dst.col-1)
				dst.data[i*dst.col+j]=data[i*2*col+j*2];
	}
};




      
        template<typename T>
        cv::Mat convertToMat(const Matrix<T>& mat) {
          cv::Mat tmp(mat.row,mat.col,CV_8U,mat.data);
          cv::Mat gray_fpt;
          tmp.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
          return gray_fpt;
        }

void calculateBoxBlurSize(double sigma, int n, int *res)
{
	double w=sqrt(12.0*sigma*sigma/n+1);
	int wl=floor(w); if (wl%2==0) wl--;
	int wu=wl+2;
	int m=round(double(12*sigma*sigma-n*wl*wl-4*n*wl-3*n)/(-4*wl-4));
	assert(m>=0 && m<=n);
	//printf("%d %d %d %d\n",m,n,wl,wu);
	double sigmaActual = sqrt((m*wl*wl + (n-m)*wu*wu - n)/12.0);
	//printf("desired sigma = %.6lf\n, actual sigma = %.6lf\n", sigma, sigmaActual);
	rep(i,0,n-1) if (i<m) res[i]=wl; else res[i]=wu;
}

Matrix<uint16_t> tmp(1600,1600);	// !ALERT! need resize...

void ApplyBoxBlur(int row, int col, int boxWidth)
{
	//fill the border area according to openCV rule (REFLECT_101)
	rep(i,0,10) 
		memcpy(&tmp.data[i*tmp.col+11],&tmp.data[(22-i)*tmp.col+11],sizeof(uint16_t)*col);
	rep(i,row+11,row+21) 
		memcpy(&tmp.data[i*tmp.col+11],&tmp.data[(row*2+20-i)*tmp.col+11],sizeof(uint16_t)*col);
	
	rep(i,0,row+21)
	{
		rep(j,0,10)
			tmp.data[i*tmp.col+j]=tmp.data[i*tmp.col+22-j];
		rep(j,col+11,col+21)
			tmp.data[i*tmp.col+j]=tmp.data[i*tmp.col+(col*2+20-j)];
	}
	
	//calculate partial sum
	rep(i,1,col+21) tmp.data[i]+=tmp.data[i-1];
	rep(i,1,row+21) tmp.data[i*tmp.col]+=tmp.data[(i-1)*tmp.col];
	
	rep(i,1,row+21)
		rep(j,1,col+21)
			tmp.data[i*tmp.col+j]+=tmp.data[(i-1)*tmp.col+j]+tmp.data[i*tmp.col+j-1]-tmp.data[(i-1)*tmp.col+j-1];
	
	//calculate result
	repd(i,row+21,boxWidth)
		repd(j,col+21,boxWidth)
		{
			tmp.data[i*tmp.col+j]-=tmp.data[(i-boxWidth)*tmp.col+j]+tmp.data[i*tmp.col+j-boxWidth]-tmp.data[(i-boxWidth)*tmp.col+j-boxWidth];
			tmp.data[i*tmp.col+j]/=boxWidth*boxWidth;
		}
		
	//move to correct position
	rep(i,11,row+10)
		rep(j,11,col+10)
			tmp.data[i*tmp.col+j]=tmp.data[(i+boxWidth/2)*tmp.col+j+boxWidth/2];
}

void myGaussianBlur(const cv::Mat &src, cv::Mat &dst, int *plan)
{
	dst = src;
	int i=0;
	while (plan[i]!=-1) 
	{
		cv::boxFilter(dst, dst, -1, cv::Size(plan[i], plan[i]));
		i++;
	}
}

void gen_pic(const Matrix<uint8_t> &A, std::string filename)
{
	//FILE* out_file = fopen(filename.c_str(), "wb");
	//fprintf(out_file, "P5\n");
	//fprintf(out_file, "%d %d\n255\n", A.col, A.row);
	//fwrite(A.data, sizeof(unsigned char), A.col*A.row, out_file);
	//fclose(out_file);
        imwrite(filename.c_str(), convertToMat(A));
}

void compare_matrix(const cv::Mat &A, const cv::Mat &B)
{
	if (A.rows != B.rows || A.cols != B.cols) 
	{
		printf("!!!ERROR!!! compare_matrix: dimensions differ!\n");
		return;
	}
	double sumdiff=0, sum=0;
	rep(i,0,A.rows-1)
		rep(j,0,A.cols-1)
		{
			sumdiff+=abs((int)A.at<uint8_t>(i,j)-(int)B.at<float>(i,j));
			sum+=(int)B.at<float>(i,j);
		}
	printf("compare_matrix: Relative Error = %.6lf\n",sumdiff/sum);
}

static const int BoxBlurPlan[9][6]={
	{-1,-1,-1,-1,-1,-1},
	{3,3,3,3,3,-1},
	{3,3,3,5,-1,0},
	{3,3,5,5,-1,0},
	{5,5,5,-1,0,0},
	{5,5,7,-1,0,0},
	{7,7,5,-1,0,0},
	{7,7,7,-1,0,0},
	{7,7,7,7,-1,0}
};

struct BoxBlurExecutionPlan_t
{
	int initialized, nodeCount;
	std::pair<int,int> parent[60];
	int target[9];
	BoxBlurExecutionPlan_t() { initialized = 0; }
} BoxBlurExecutionPlan;

void generateBoxBlurExecutionPlan()
{
	if (BoxBlurExecutionPlan.initialized) return;
	BoxBlurExecutionPlan.initialized = 1;
	std::map< std::vector<int>, int > T;
	int all=0;
	T[std::vector<int>()]=0;
	rep(i,1,8)
	{
		std::vector<int> lis;
		int j=0, p=0;
		while (BoxBlurPlan[i][j]!=-1)
		{
			lis.push_back(BoxBlurPlan[i][j]);
			if (!T.count(lis))
			{
				all++; 
				T[lis]=all;
				BoxBlurExecutionPlan.parent[all]=std::make_pair(p,BoxBlurPlan[i][j]);
			}
			p=T[lis];
			j++;
		}
		BoxBlurExecutionPlan.target[i]=p;
	}
	BoxBlurExecutionPlan.nodeCount=all;
}
			
void myBuildGaussianPyramid( const cv::Mat& baseimg, const std::vector<cv::Mat>& pyr_ans, int nOctaves, int nOctaveLayers)
{
	printf("entered\n");
	/*
	double sigma = 1.6;
	static double sig[200];
	sig[0] = sigma;
	double k = std::pow( 2., 1. / nOctaveLayers );
	static int boxBlurSz[20][20];
	rep(i,1,nOctaveLayers+2)
	{
		double sig_prev = std::pow(k, (double)(i-1))*sigma;
		double sig_total = sig_prev*k;
		sig[i] = sig_total;
		//printf("sig[%d] = %.6lf\n",i,sig[i]);
		assert(sig[i]>1.75);
		int len;
		if (sig[i]<2.5 || i==nOctaveLayers+2) len=4; else len=3;
		if (sig[i]<1.8) len=5;
		calculateBoxBlurSize(sig[i], len, boxBlurSz[i]);
		boxBlurSz[i][len]=-1;
		printf("{");
		rep(j,0,5) 
		{
			printf("%d",boxBlurSz[i][j]);
			if (j<5) printf(",");
		}
		printf("},\n");
	}
	*/
	
	assert(nOctaveLayers==6);
	
	generateBoxBlurExecutionPlan();
	
	std::vector< cv::Mat > pyr;
	pyr.resize(nOctaves*(nOctaveLayers + 3));
	
	pyr[0]=cv::Mat(baseimg.rows, baseimg.cols, CV_8U);
	cv::Mat &base = pyr[0]; 
	rep(i,0,baseimg.rows-1)
		rep(j,0,baseimg.cols-1)
			base.at<uint8_t>(i,j)=baseimg.at<float>(i,j);

	static double GBlurTime=0;
	fasttime_t tstart=gettime();
	for( int o = 0; o < nOctaves; o++ )
	{
		if (o)
		{
			cv::Mat &dst = pyr[o*(nOctaveLayers + 3)];
			const cv::Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
			cv::resize(src, dst, Size(src.cols/2, src.rows/2), 0, 0, INTER_NEAREST);
		}
		
		const cv::Mat& src = pyr[o*(nOctaveLayers + 3)];
		
		std::vector<cv::Mat> tmplist;
		tmplist.resize(BoxBlurExecutionPlan.nodeCount+1);
		rep(i,1,BoxBlurExecutionPlan.nodeCount)
		{
			int p=BoxBlurExecutionPlan.parent[i].first;
			int sz=BoxBlurExecutionPlan.parent[i].second;
			if (p==0)
				cv::boxFilter(src, tmplist[i], -1, cv::Size(sz, sz));
			else
				cv::boxFilter(tmplist[p], tmplist[i], -1, cv::Size(sz, sz));
		}
		
		rep(i,1,nOctaveLayers+2)
			cv::swap(pyr[o*(nOctaveLayers + 3) + i], tmplist[BoxBlurExecutionPlan.target[i]]);

	}
	fasttime_t tend=gettime();
	GBlurTime+=tdiff(tstart,tend);
	
	rep(i,0,nOctaves*(nOctaveLayers + 3)-1)
		compare_matrix(pyr[i], pyr_ans[i]);
		
	printf("myGaussianBlur time = %.6lf\n",GBlurTime);
}

		
	
