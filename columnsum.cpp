namespace cv
{
	
struct ColumnSum :
public BaseColumnFilter
{
    ColumnSum( int _ksize, int _anchor, double _scale ) :
    BaseColumnFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
        scale = _scale;
        sumCount = 0;
        divDelta = 0;
        divScale = 1;
        if( scale != 1 )
        {
            int d = cvRound(1./scale);
            double scalef = ((double)(1 << 16))/d;
            divScale = cvFloor(scalef);
            scalef -= divScale;
            divDelta = d/2;
            if( scalef < 0.5 )
                divDelta++;
            else
                divScale++;
        }
    }

    virtual void reset() { sumCount = 0; }

    virtual void operator()(const uchar** src, uchar* dst, int dststep, int count, int width)
    {
        const int ds = divScale;
        const int dd = divDelta;
        ushort* SUM;
        const bool haveScale = scale != 1;

#if CV_SSE2
        bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#elif CV_NEON
        bool haveNEON = checkHardwareSupport(CV_CPU_NEON);
#endif

        if( width != (int)sum.size() )
        {
            sum.resize(width);
            sumCount = 0;
        }

        SUM = &sum[0];
        if( sumCount == 0 )
        {
            memset((void*)SUM, 0, width*sizeof(SUM[0]));
            for( ; sumCount < ksize - 1; sumCount++, src++ )
            {
                const ushort* Sp = (const ushort*)src[0];
                int i = 0;
#if CV_SSE2
                if(haveSSE2)
                {
                    for( ; i <= width-8; i+=8 )
                    {
                        __m128i _sum = _mm_loadu_si128((const __m128i*)(SUM+i));
                        __m128i _sp = _mm_loadu_si128((const __m128i*)(Sp+i));
                        _mm_storeu_si128((__m128i*)(SUM+i),_mm_add_epi16(_sum, _sp));
                    }
                }
#elif CV_NEON
                if(haveNEON)
                {
                    for( ; i <= width - 8; i+=8 )
                        vst1q_u16(SUM + i, vaddq_u16(vld1q_u16(SUM + i), vld1q_u16(Sp + i)));
                }
#endif
                for( ; i < width; i++ )
                    SUM[i] += Sp[i];
            }
        }
        else
        {
            CV_Assert( sumCount == ksize-1 );
            src += ksize-1;
        }

        for( ; count--; src++ )
        {
            const ushort* Sp = (const ushort*)src[0];
            const ushort* Sm = (const ushort*)src[1-ksize];
            uchar* D = (uchar*)dst;
            if( haveScale )
            {
                int i = 0;
    #if CV_SSE2
                if(haveSSE2)
                {
                    __m128i ds8 = _mm_set1_epi16((short)ds);
                    __m128i dd8 = _mm_set1_epi16((short)dd);

                    for( ; i <= width-16; i+=16 )
                    {
                        __m128i _sm0  = _mm_loadu_si128((const __m128i*)(Sm+i));
                        __m128i _sm1  = _mm_loadu_si128((const __m128i*)(Sm+i+8));

                        __m128i _s0  = _mm_add_epi16(_mm_loadu_si128((const __m128i*)(SUM+i)),
                                                     _mm_loadu_si128((const __m128i*)(Sp+i)));
                        __m128i _s1  = _mm_add_epi16(_mm_loadu_si128((const __m128i*)(SUM+i+8)),
                                                     _mm_loadu_si128((const __m128i*)(Sp+i+8)));
                        __m128i _s2 = _mm_mulhi_epu16(_mm_adds_epu16(_s0, dd8), ds8);
                        __m128i _s3 = _mm_mulhi_epu16(_mm_adds_epu16(_s1, dd8), ds8);
                        _s0 = _mm_sub_epi16(_s0, _sm0);
                        _s1 = _mm_sub_epi16(_s1, _sm1);
                        _mm_storeu_si128((__m128i*)(D+i), _mm_packus_epi16(_s2, _s3));
                        _mm_storeu_si128((__m128i*)(SUM+i), _s0);
                        _mm_storeu_si128((__m128i*)(SUM+i+8), _s1);
                    }
                }
    #endif
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = (uchar)((s0 + dd)*ds >> 16);
                    SUM[i] = (ushort)(s0 - Sm[i]);
                }
            }
            else
            {
                int i = 0;
                for( ; i < width; i++ )
                {
                    int s0 = SUM[i] + Sp[i];
                    D[i] = saturate_cast<uchar>(s0);
                    SUM[i] = (ushort)(s0 - Sm[i]);
                }
            }
            dst += dststep;
        }
    }

    double scale;
    int sumCount;
    int divDelta;
    int divScale;
    std::vector<ushort> sum;
};

} 
