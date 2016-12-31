#include <opencv2/imgproc/imgproc.hpp>
#include "filterengine.hpp"

#include "filterengine.cpp"

#include "rowsum.cpp"
#include "columnsum.cpp"

namespace cv
{

cv::Ptr<cv::BaseRowFilter> getRowSumFilter(int srcType, int sumType, int ksize, int anchor)
{
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(srcType) );

    if( anchor < 0 ) anchor = ksize/2;

    assert(CV_MAT_DEPTH(srcType) == CV_8U && CV_MAT_DEPTH(sumType) == CV_16U);
    return makePtr<RowSum>(ksize, anchor);
}


cv::Ptr<cv::BaseColumnFilter> getColumnSumFilter(int sumType, int dstType, int ksize,
                                                     int anchor, double scale)
{
    CV_Assert( CV_MAT_CN(sumType) == CV_MAT_CN(dstType) );

    if( anchor < 0 )
        anchor = ksize/2;

   
    assert( CV_MAT_DEPTH(dstType) == CV_8U && CV_MAT_DEPTH(sumType) == CV_16U );
    
    return makePtr<ColumnSum >(ksize, anchor, scale);
}


cv::Ptr<cv::FilterEngine> createBoxFilterCV8U( int srcType, int dstType, Size ksize,
                    Point anchor, bool normalize, int borderType )
{
    assert(CV_MAT_DEPTH(srcType) == CV_8U);
    int cn = CV_MAT_CN(srcType), sumType = CV_16U;
    sumType = CV_MAKETYPE( sumType, cn );

    Ptr<BaseRowFilter> rowFilter = cv::getRowSumFilter(srcType, sumType, ksize.width, anchor.x );
    Ptr<BaseColumnFilter> columnFilter = cv::getColumnSumFilter(sumType,
        dstType, ksize.height, anchor.y, normalize ? 1./(ksize.width*ksize.height) : 1);

    return makePtr<FilterEngine>(Ptr<BaseFilter>(), rowFilter, columnFilter,
           srcType, dstType, sumType, borderType );
}

void boxFilterCV8U( InputArray _src, OutputArray _dst, int ddepth,
                Size ksize, Point anchor = Point(-1,-1),
                bool normalize = true, int borderType = BORDER_DEFAULT )
{
    Mat src = _src.getMat();
    int stype = src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if( ddepth < 0 )
        ddepth = sdepth;
    _dst.create( src.size(), CV_MAKETYPE(ddepth, cn) );
    Mat dst = _dst.getMat();
    if( borderType != BORDER_CONSTANT && normalize && (borderType & BORDER_ISOLATED) != 0 )
    {
        if( src.rows == 1 )
            ksize.height = 1;
        if( src.cols == 1 )
            ksize.width = 1;
    }

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType&BORDER_ISOLATED))
        src.locateROI( wsz, ofs );
    borderType = (borderType&~BORDER_ISOLATED);

    Ptr<FilterEngine> f = createBoxFilterCV8U( src.type(), dst.type(),
                        ksize, anchor, normalize, borderType );

    f->apply( src, dst, wsz, ofs );
}

}