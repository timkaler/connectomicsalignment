#include "common.h"
#include "fasttime.h"

#include "boxFilter.cpp"

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

void compare_matrix_uchar_uchar(const cv::Mat &A, const cv::Mat &B)
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
			sumdiff+=abs((int)A.at<uint8_t>(i,j)-(int)B.at<uint8_t>(i,j));
			sum+=(int)B.at<uint8_t>(i,j);
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

static std::mutex GaussianPyramidBoxBlurTimer_mutex;

/*
 * A faster implementation of BuildGaussianPyramid
 * It uses BoxBlur to approximate Gaussian, 18% faster than default version.
 * On testdata the highest relative error observed is 7%. For most cases the error is 1% ~ 2%.
 * 
 * !WARNING!:
 * This implementation assumes the following parameter:
 * 	initial sigma = 1.6
 * 	nOctaveLayers = 6
 *
 */
void BuildGaussianPyramid_BoxBlurApproximation( const cv::Mat& baseimg, std::vector< cv::Mat > &pyr, int nOctaves, int nOctaveLayers)
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
	
	//generateBoxBlurExecutionPlan();	//moved to SIFT_initialize() to avoid races
	
	pyr.resize(nOctaves*(nOctaveLayers + 3));
	
	//pyr[0]=cv::Mat(baseimg.rows, baseimg.cols, CV_8U);
	cv::Mat &base = pyr[0]; 
	
	/*
	rep(i,0,baseimg.rows-1)
		rep(j,0,baseimg.cols-1)
			base.at<uint8_t>(i,j)=baseimg.at<float>(i,j);
	*/
	// a better way than above to copy a matrix
	/*
	int nRows = baseimg.rows;
	int nCols = baseimg.cols;
	rep(i,0,nRows-1)
	{
		const float *p1 = baseimg.ptr<float>(i);
		uint8_t *p2 = base.ptr<uint8_t>(i);
		rep(j,0,nCols-1) p2[j] = (uint8_t)p1[j];
	}
	*/
	// !!!WARNING!!! THIS ASSUMES DATA TYPE IS uint8_t !!!
	base = baseimg;
	
	static double GBlurTime=0;
	fasttime_t tstart=gettime();
	for( int o = 0; o < nOctaves; o++ )
	{
		if (o)
		{
			cv::Mat &dst = pyr[o*(nOctaveLayers + 3)];
			const cv::Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
			cv::resize(src, dst, cv::Size(src.cols/2, src.rows/2), 0, 0, cv::INTER_NEAREST);
		}
		
		const cv::Mat& src = pyr[o*(nOctaveLayers + 3)];
		
		std::vector<cv::Mat> tmplist;
		tmplist.resize(BoxBlurExecutionPlan.nodeCount+1);
		rep(i,1,BoxBlurExecutionPlan.nodeCount)
		{
			int p=BoxBlurExecutionPlan.parent[i].first;
			int sz=BoxBlurExecutionPlan.parent[i].second;
			if (p==0)
				cv::boxFilterCV8U(src, tmplist[i], -1, cv::Size(sz, sz));
			else
				cv::boxFilterCV8U(tmplist[p], tmplist[i], -1, cv::Size(sz, sz));
		}
		
		rep(i,1,nOctaveLayers+2)
			cv::swap(pyr[o*(nOctaveLayers + 3) + i], tmplist[BoxBlurExecutionPlan.target[i]]);

	}
	fasttime_t tend=gettime();
	
	GaussianPyramidBoxBlurTimer_mutex.lock();
	GBlurTime+=tdiff(tstart,tend);
	GaussianPyramidBoxBlurTimer_mutex.unlock();
	
	printf("cumulative myGaussianBlur time = %.6lf\n",GBlurTime);
}

