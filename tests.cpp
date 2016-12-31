void test_boxblur()
{
	int BS=1;
	std::vector<cv::Mat> T1;
	T1.resize(BS);
	rep(i,0,BS-1) T1[i]=cv::Mat(1000,1000,CV_8UC(16));
	rep(i,0,BS-1)
		rep(j,0,999)
		{
			uint8_t *p1 = T1[i].ptr<uint8_t>(j);
			rep(k,0,16000-1)
			{
				p1[k]=rand()%256;
			}
		}

	
	std::vector<cv::Mat> T2;
	T2.resize(BS*16);
	rep(i,0,BS*16-1) T2[i]=cv::Mat(1000,1000,CV_8U);
	rep(i,0,BS*16-1)
		rep(j,0,999)
			rep(k,0,999)
			{
				T2[i].at<uint8_t>(j,k)=rand()%256;
			}
	
	fasttime_t tstart = gettime();
	rep(i,0,BS-1)
		rep(j,0,100)
		{
			cv::Mat tmp;
			cv::boxFilterCV8U(T1[i], tmp, -1, cv::Size(3,3));
			cv::boxFilter(T1[i], T1[i], -1, cv::Size(3,3));
			compare_matrix_uchar_uchar(tmp,T1[i]);
			cv::boxFilterCV8U(T1[i], tmp, -1, cv::Size(5,5));
			cv::boxFilter(T1[i], T1[i], -1, cv::Size(5,5));
			compare_matrix_uchar_uchar(tmp,T1[i]);
			cv::boxFilterCV8U(T1[i], tmp, -1, cv::Size(7,7));
			cv::boxFilter(T1[i], T1[i], -1, cv::Size(7,7));
			compare_matrix_uchar_uchar(tmp,T1[i]);
		}
	
	fasttime_t tend=gettime();
	double t1=tdiff(tstart,tend);
	
	
	tstart = gettime();
	rep(i,0,BS*16-1)
		rep(j,0,100)
		{
			cv::boxFilter(T2[i], T2[i], -1, cv::Size(3,3));
			cv::boxFilter(T2[i], T2[i], -1, cv::Size(5,5));
			cv::boxFilter(T2[i], T2[i], -1, cv::Size(7,7));
		}
		
	tend=gettime();
	double t2=tdiff(tstart,tend);
	
	printf("Time 1 = %.6lf\n, Time 2 = %.6lf\n",t1,t2);
	exit(0);
}

void test_maxfilter()
{
	std::vector<cv::Mat> A,B;
	A.resize(200);
	rep(i,0,199) A[i]=cv::Mat(250,300,CV_16SC(16));
	B.resize(200);
	rep(i,0,199) B[i]=cv::Mat(250,300,CV_16SC(16));
	rep(i,0,199)
		rep(j,0,249)
			rep(k,0,299)
				rep(c,0,15)
				{
					int x=rand()%12345-6000;
					A[i].at<sift_wt>(i,j).chan[c]=x;
					B[i].at<sift_wt>(i,j).chan[c]=x;
				}
				
	maxFilter3x3::applyMaxFilter(B);
		
	rep(i,1,198)
		rep(j,1,248)
			rep(k,1,298)
				rep(c,0,15)
				{
					short v1=-16000;
					rep(d0,-1,1)
						rep(d1,-1,1)
							rep(d2,-1,1)
								v1=std::max(v1,A[i+d0].at<sift_wt>(j+d1,k+d2).chan[c]);
					
					short v2=B[i].at<sift_wt>(j,k).chan[c];
					if (v1!=v2) 
					{
						printf("Failure at %d %d %d %d ans=%d output=%d\n",i,j,k,c,(int)v1,(int)v2);
						exit(0);
					}
				}
	printf("passed\n");
	exit(0);
}


void test_minfilter()
{
	std::vector<cv::Mat> A,B;
	A.resize(200);
	rep(i,0,199) A[i]=cv::Mat(250,300,CV_16SC(16));
	B.resize(200);
	rep(i,0,199) B[i]=cv::Mat(250,300,CV_16SC(16));
	rep(i,0,199)
		rep(j,0,249)
			rep(k,0,299)
				rep(c,0,15)
				{
					int x=rand()%12345-6000;
					A[i].at<sift_wt>(i,j).chan[c]=x;
					B[i].at<sift_wt>(i,j).chan[c]=x;
				}
				
	minFilter3x3::applyMinFilter(B);
		
	rep(i,1,198)
		rep(j,1,248)
			rep(k,1,298)
				rep(c,0,15)
				{
					short v1=16000;
					rep(d0,-1,1)
						rep(d1,-1,1)
							rep(d2,-1,1)
								v1=std::min(v1,A[i+d0].at<sift_wt>(j+d1,k+d2).chan[c]);
					
					short v2=B[i].at<sift_wt>(j,k).chan[c];
					if (v1!=v2) 
					{
						printf("Failure at %d %d %d %d ans=%d output=%d\n",i,j,k,c,(int)v1,(int)v2);
						exit(0);
					}
				}
	printf("passed\n");
	exit(0);
}


