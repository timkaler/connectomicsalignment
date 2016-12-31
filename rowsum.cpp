namespace cv
{


struct RowSum :
        public BaseRowFilter
{
    RowSum( int _ksize, int _anchor ) :
        BaseRowFilter()
    {
        ksize = _ksize;
        anchor = _anchor;
    }

    virtual void operator()(const uchar* src, uchar* dst, int width, int cn)
    {
        const uchar* S = (const uchar*)src;
        ushort* D = (ushort*)dst;
        int i = 0, k, ksz_cn = ksize*cn;

        width = (width - 1)*cn;
        if( ksize == 3 )
        {
            for( i = 0; i < width + cn; i++ )
            {
                D[i] = (ushort)S[i] + (ushort)S[i+cn] + (ushort)S[i+cn*2];
            }
        }
        else if( ksize == 5 )
        {
            for( i = 0; i < width + cn; i++ )
            {
                D[i] = (ushort)S[i] + (ushort)S[i+cn] + (ushort)S[i+cn*2] + (ushort)S[i + cn*3] + (ushort)S[i + cn*4];
            }
        }
        else if( cn == 1 )
        {
            ushort s = 0;
            for( i = 0; i < ksz_cn; i++ )
                s += (ushort)S[i];
            D[0] = s;
            for( i = 0; i < width; i++ )
            {
                s += (ushort)S[i + ksz_cn] - (ushort)S[i];
                D[i+1] = s;
            }
        }
        else if( cn == 3 )
        {
            ushort s0 = 0, s1 = 0, s2 = 0;
            for( i = 0; i < ksz_cn; i += 3 )
            {
                s0 += (ushort)S[i];
                s1 += (ushort)S[i+1];
                s2 += (ushort)S[i+2];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            for( i = 0; i < width; i += 3 )
            {
                s0 += (ushort)S[i + ksz_cn] - (ushort)S[i];
                s1 += (ushort)S[i + ksz_cn + 1] - (ushort)S[i + 1];
                s2 += (ushort)S[i + ksz_cn + 2] - (ushort)S[i + 2];
                D[i+3] = s0;
                D[i+4] = s1;
                D[i+5] = s2;
            }
        }
        else if( cn == 4 )
        {
            ushort s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for( i = 0; i < ksz_cn; i += 4 )
            {
                s0 += (ushort)S[i];
                s1 += (ushort)S[i+1];
                s2 += (ushort)S[i+2];
                s3 += (ushort)S[i+3];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            D[3] = s3;
            for( i = 0; i < width; i += 4 )
            {
                s0 += (ushort)S[i + ksz_cn] - (ushort)S[i];
                s1 += (ushort)S[i + ksz_cn + 1] - (ushort)S[i + 1];
                s2 += (ushort)S[i + ksz_cn + 2] - (ushort)S[i + 2];
                s3 += (ushort)S[i + ksz_cn + 3] - (ushort)S[i + 3];
                D[i+4] = s0;
                D[i+5] = s1;
                D[i+6] = s2;
                D[i+7] = s3;
            }
        }
        else if( cn == 16 )
        {
            ushort s0 = 0, s1 = 0, s2 = 0, s3 = 0;
		ushort s4 = 0, s5 = 0, s6 = 0, s7 = 0;
		ushort s8 = 0, s9 = 0, s10 = 0, s11 = 0;
		ushort s12 = 0, s13 = 0, s14 = 0, s15 = 0;
            for( i = 0; i < ksz_cn; i += 16 )
            {
                s0 += (ushort)S[i];
                s1 += (ushort)S[i+1];
                s2 += (ushort)S[i+2];
                s3 += (ushort)S[i+3];
		    s4 += (ushort)S[i+4];
                s5 += (ushort)S[i+5];
                s6 += (ushort)S[i+6];
                s7 += (ushort)S[i+7];
		    s8 += (ushort)S[i+8];
                s9 += (ushort)S[i+9]; 
                s10 += (ushort)S[i+10];
                s11 += (ushort)S[i+11];
		    s12 += (ushort)S[i+12];
                s13 += (ushort)S[i+13];
                s14 += (ushort)S[i+14];
                s15 += (ushort)S[i+15];
            }
            D[0] = s0;
            D[1] = s1;
            D[2] = s2;
            D[3] = s3;
		D[4] = s4;
            D[5] = s5;
            D[6] = s6;
            D[7] = s7;
		D[8] = s8;
            D[9] = s9;
            D[10] = s10;
            D[11] = s11;
		D[12] = s12;
            D[13] = s13;
            D[14] = s14;
            D[15] = s15;
            for( i = 0; i < width; i += 16 )
            {
                s0 += (ushort)S[i + ksz_cn] - (ushort)S[i];
                s1 += (ushort)S[i + ksz_cn + 1] - (ushort)S[i + 1];
                s2 += (ushort)S[i + ksz_cn + 2] - (ushort)S[i + 2];
                s3 += (ushort)S[i + ksz_cn + 3] - (ushort)S[i + 3];
		    s4 += (ushort)S[i + ksz_cn + 4] - (ushort)S[i + 4];
                s5 += (ushort)S[i + ksz_cn + 5] - (ushort)S[i + 5];
                s6 += (ushort)S[i + ksz_cn + 6] - (ushort)S[i + 6];
                s7 += (ushort)S[i + ksz_cn + 7] - (ushort)S[i + 7];
		    s8 += (ushort)S[i + ksz_cn + 8] - (ushort)S[i + 8];
                s9 += (ushort)S[i + ksz_cn + 9] - (ushort)S[i + 9];
                s10 += (ushort)S[i + ksz_cn + 10] - (ushort)S[i + 10];
                s11 += (ushort)S[i + ksz_cn + 11] - (ushort)S[i + 11];
		    s12 += (ushort)S[i + ksz_cn + 12] - (ushort)S[i + 12];
                s13 += (ushort)S[i + ksz_cn + 13] - (ushort)S[i + 13];
                s14 += (ushort)S[i + ksz_cn + 14] - (ushort)S[i + 14];
                s15 += (ushort)S[i + ksz_cn + 15] - (ushort)S[i + 15];
                D[i+16] = s0;
                D[i+17] = s1;
                D[i+18] = s2;
                D[i+19] = s3;
		    D[i+20] = s4;
                D[i+21] = s5;
                D[i+22] = s6;
                D[i+23] = s7;
		    D[i+24] = s8;
                D[i+25] = s9;
                D[i+26] = s10;
                D[i+27] = s11;
		    D[i+28] = s12;
                D[i+29] = s13;
                D[i+30] = s14;
                D[i+31] = s15;
            }
        }
        else
            for( k = 0; k < cn; k++, S++, D++ )
            {
                ushort s = 0;
                for( i = 0; i < ksz_cn; i += cn )
                    s += (ushort)S[i];
                D[0] = s;
                for( i = 0; i < width; i += cn )
                {
                    s += (ushort)S[i + ksz_cn] - (ushort)S[i];
                    D[i+cn] = s;
                }
            }
    }
};

}
