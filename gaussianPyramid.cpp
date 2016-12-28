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

void myGaussianBlur(const Matrix<uint8_t> &src, Matrix<uint8_t> &dst, int *plan)
{
	rep(i,0,src.row-1)
		rep(j,0,src.col-1)
			tmp.data[(i+11)*tmp.col+(j+11)]=src.data[i*src.col+j];
			
	int i=0;
	while (plan[i]!=-1) 
	{
		ApplyBoxBlur(src.row, src.col, plan[i]);
		i++;
	}
	
	dst=Matrix<uint8_t>(src.row, src.col);
	rep(i,0,src.row-1)
		rep(j,0,src.col-1)
			dst.data[i*src.col+j]=tmp.data[(i+11)*tmp.col+(j+11)];
}

void gen_pic(const Matrix<uint8_t> &A, std::string filename)
{
	FILE* out_file = fopen(filename.c_str(), "wb");
	fprintf(out_file, "P5\n");
	fprintf(out_file, "%d %d\n255\n", A.col, A.row);
	fwrite(A.data, sizeof(unsigned char), A.col*A.row, out_file);
	fclose(out_file);
}

void compare_matrix(const Matrix<uint8_t> &A, const cv::Mat &B)
{
	if (A.row != B.rows || A.col != B.cols) 
	{
		printf("!!!ERROR!!! compare_matrix: dimensions differ!\n");
		return;
	}
	double sumdiff=0, sum=0;
	rep(i,0,A.row-1)
		rep(j,0,A.col-1)
		{
			sumdiff+=abs((int)A.data[i*A.col+j]-(int)B.at<unsigned char>(i,j));
			sum+=(int)B.at<unsigned char>(i,j);
		}
	printf("compare_matrix: Relative Error = %.6lf\n",sumdiff/sum);
	Matrix<uint8_t> tmp(A.row,A.col);
	rep(i,0,A.row-1)
		rep(j,0,A.col-1)
			tmp.data[i*A.col+j]=B.at<unsigned char>(i,j);
	
	static int cnt=0;
	cnt++;
	char buf[100];
	sprintf(buf,"%d.pgm",cnt);
	gen_pic(A,buf);
	cnt++;
	sprintf(buf,"%d.pgm",cnt);
	gen_pic(tmp,buf);
}

void myBuildGaussianPyramid( const cv::Mat& baseimg, const std::vector<cv::Mat>& pyr_ans, int nOctaves, int nOctaveLayers)
{
	printf("entered\n");
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
		if (sig[i]>1.75)
		{
			int len;
			if (sig[i]<2.5 || i==nOctaveLayers+2) len=4; else len=3;
			if (sig[i]<1.8) len=5;
			len=15;
			calculateBoxBlurSize(sig[i], len, boxBlurSz[i]);
			boxBlurSz[i][len]=-1;
		}
	}
	
	std::vector<Matrix<uint8_t> > pyr;
	pyr.resize(nOctaves*(nOctaveLayers + 3));
	
	Matrix<uint8_t> &base = pyr[0];
	base = Matrix<uint8_t>(baseimg.rows, baseimg.cols);
	int all=0;
	rep(i,0,baseimg.rows-1)
		rep(j,0,baseimg.cols-1)
		{
			base.data[all]=baseimg.at<unsigned char>(i,j);
			all++;
		}
	
	compare_matrix(base, pyr_ans[0]);
	exit(0);
	
	static double GBlurTime=0;
	for( int o = 0; o < nOctaves; o++ )
	{
		for( int i = 0; i < nOctaveLayers + 3; i++ )
		{
			if ((pyr[0].row>>o)<15 || (pyr[0].col>>o)<15) continue;
			Matrix<uint8_t> &dst = pyr[o*(nOctaveLayers + 3) + i];
			if( o == 0  &&  i == 0 ) continue;
			// base of new octave is halved image from end of previous octave
			if( i == 0 )
			{
				const Matrix<uint8_t> &src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
				src.downsample2x(dst);
				compare_matrix(dst, pyr_ans[o*(nOctaveLayers + 3) + i]);
			}
			else
			{
				const Matrix<uint8_t> &src = pyr[o*(nOctaveLayers + 3)];
				fasttime_t tstart=gettime();
				myGaussianBlur(src, dst, boxBlurSz[i]);
				fasttime_t tend=gettime();
				GBlurTime+=tdiff(tstart,tend);
				compare_matrix(dst, pyr_ans[o*(nOctaveLayers + 3) + i]);
			}
		}
	}
	
	printf("myGaussianBlur time = %.6lf\n",GBlurTime);
	rept(it, pyr) it->freeMemory();
	exit(0);
}

		
	