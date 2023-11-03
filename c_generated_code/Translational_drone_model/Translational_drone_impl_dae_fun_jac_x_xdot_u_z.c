/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Translational_drone_impl_dae_fun_jac_x_xdot_u_z_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s12 CASADI_PREFIX(s12)
#define casadi_s13 CASADI_PREFIX(s13)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_trans CASADI_PREFIX(trans)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

static const casadi_int casadi_s0[8] = {0, 1, 2, 6, 9, 12, 15, 19};
static const casadi_int casadi_s1[6] = {3, 10, 13, 16, 20, 23};
static const casadi_int casadi_s2[6] = {4, 7, 14, 17, 21, 24};
static const casadi_int casadi_s3[5] = {5, 8, 11, 18, 22};
static const casadi_int casadi_s4[38] = {10, 10, 0, 0, 0, 0, 5, 11, 17, 22, 23, 24, 25, 4, 5, 6, 7, 8, 3, 5, 6, 7, 8, 9, 3, 4, 6, 7, 8, 9, 3, 4, 5, 7, 8, 0, 1, 2};
static const casadi_int casadi_s5[38] = {10, 10, 0, 1, 2, 3, 6, 9, 12, 15, 19, 23, 25, 7, 8, 9, 4, 5, 6, 3, 5, 6, 3, 4, 6, 3, 4, 5, 3, 4, 5, 6, 3, 4, 5, 6, 4, 5};
static const casadi_int casadi_s6[23] = {10, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s7[7] = {0, 3, 6, 9, 12, 13, 14};
static const casadi_int casadi_s8[22] = {10, 4, 0, 3, 7, 11, 15, 7, 8, 9, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};
static const casadi_int casadi_s9[28] = {4, 10, 0, 0, 0, 0, 3, 6, 9, 12, 13, 14, 15, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0};
static const casadi_int casadi_s10[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s11[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s12[3] = {0, 0, 0};
static const casadi_int casadi_s13[3] = {10, 0, 0};

/* Translational_drone_impl_dae_fun_jac_x_xdot_u_z:(i0[10],i1[10],i2[4],i3[],i4[])->(o0[10],o1[10x10,25nz],o2[10x10,10nz],o3[10x4,15nz],o4[10x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real *rr, *ss;
  const casadi_int *cii;
  const casadi_real *cs;
  casadi_real w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, *w10=w+10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, *w32=w+41, *w33=w+51, *w34=w+76, *w37=w+83, *w38=w+91, *w41=w+99, *w42=w+105, *w44=w+111, *w45=w+116, *w46=w+121, *w47=w+146, *w48=w+161, *w49=w+163, *w51=w+170, *w52=w+174, *w53=w+178;
  /* #0: @0 = input[1][0] */
  w0 = arg[1] ? arg[1][0] : 0;
  /* #1: @1 = input[1][1] */
  w1 = arg[1] ? arg[1][1] : 0;
  /* #2: @2 = input[1][2] */
  w2 = arg[1] ? arg[1][2] : 0;
  /* #3: @3 = input[1][3] */
  w3 = arg[1] ? arg[1][3] : 0;
  /* #4: @4 = input[1][4] */
  w4 = arg[1] ? arg[1][4] : 0;
  /* #5: @5 = input[1][5] */
  w5 = arg[1] ? arg[1][5] : 0;
  /* #6: @6 = input[1][6] */
  w6 = arg[1] ? arg[1][6] : 0;
  /* #7: @7 = input[1][7] */
  w7 = arg[1] ? arg[1][7] : 0;
  /* #8: @8 = input[1][8] */
  w8 = arg[1] ? arg[1][8] : 0;
  /* #9: @9 = input[1][9] */
  w9 = arg[1] ? arg[1][9] : 0;
  /* #10: @10 = vertcat(@0, @1, @2, @3, @4, @5, @6, @7, @8, @9) */
  rr=w10;
  *rr++ = w0;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #11: @0 = input[0][7] */
  w0 = arg[0] ? arg[0][7] : 0;
  /* #12: @1 = input[0][8] */
  w1 = arg[0] ? arg[0][8] : 0;
  /* #13: @2 = input[0][9] */
  w2 = arg[0] ? arg[0][9] : 0;
  /* #14: @3 = 0.5 */
  w3 = 5.0000000000000000e-01;
  /* #15: @4 = input[2][1] */
  w4 = arg[2] ? arg[2][1] : 0;
  /* #16: @5 = input[0][4] */
  w5 = arg[0] ? arg[0][4] : 0;
  /* #17: @6 = (@4*@5) */
  w6  = (w4*w5);
  /* #18: @6 = (-@6) */
  w6 = (- w6 );
  /* #19: @7 = input[2][2] */
  w7 = arg[2] ? arg[2][2] : 0;
  /* #20: @8 = input[0][5] */
  w8 = arg[0] ? arg[0][5] : 0;
  /* #21: @9 = (@7*@8) */
  w9  = (w7*w8);
  /* #22: @6 = (@6-@9) */
  w6 -= w9;
  /* #23: @9 = input[2][3] */
  w9 = arg[2] ? arg[2][3] : 0;
  /* #24: @11 = input[0][6] */
  w11 = arg[0] ? arg[0][6] : 0;
  /* #25: @12 = (@9*@11) */
  w12  = (w9*w11);
  /* #26: @6 = (@6-@12) */
  w6 -= w12;
  /* #27: @6 = (@3*@6) */
  w6  = (w3*w6);
  /* #28: @12 = 0.5 */
  w12 = 5.0000000000000000e-01;
  /* #29: @13 = input[0][3] */
  w13 = arg[0] ? arg[0][3] : 0;
  /* #30: @14 = (@4*@13) */
  w14  = (w4*w13);
  /* #31: @15 = (@9*@8) */
  w15  = (w9*w8);
  /* #32: @14 = (@14+@15) */
  w14 += w15;
  /* #33: @15 = (@7*@11) */
  w15  = (w7*w11);
  /* #34: @14 = (@14-@15) */
  w14 -= w15;
  /* #35: @14 = (@12*@14) */
  w14  = (w12*w14);
  /* #36: @15 = 0.5 */
  w15 = 5.0000000000000000e-01;
  /* #37: @16 = (@7*@13) */
  w16  = (w7*w13);
  /* #38: @17 = (@9*@5) */
  w17  = (w9*w5);
  /* #39: @16 = (@16-@17) */
  w16 -= w17;
  /* #40: @17 = (@4*@11) */
  w17  = (w4*w11);
  /* #41: @16 = (@16+@17) */
  w16 += w17;
  /* #42: @16 = (@15*@16) */
  w16  = (w15*w16);
  /* #43: @17 = 0.5 */
  w17 = 5.0000000000000000e-01;
  /* #44: @18 = (@9*@13) */
  w18  = (w9*w13);
  /* #45: @19 = (@7*@5) */
  w19  = (w7*w5);
  /* #46: @18 = (@18+@19) */
  w18 += w19;
  /* #47: @19 = (@4*@8) */
  w19  = (w4*w8);
  /* #48: @18 = (@18-@19) */
  w18 -= w19;
  /* #49: @18 = (@17*@18) */
  w18  = (w17*w18);
  /* #50: @19 = 9.81 */
  w19 = 9.8100000000000005e+00;
  /* #51: @20 = (@13*@8) */
  w20  = (w13*w8);
  /* #52: @21 = (@5*@11) */
  w21  = (w5*w11);
  /* #53: @20 = (@20+@21) */
  w20 += w21;
  /* #54: @20 = (2.*@20) */
  w20 = (2.* w20 );
  /* #55: @21 = input[2][0] */
  w21 = arg[2] ? arg[2][0] : 0;
  /* #56: @22 = (@20*@21) */
  w22  = (w20*w21);
  /* #57: @23 = 0.76 */
  w23 = 7.6000000000000001e-01;
  /* #58: @22 = (@22/@23) */
  w22 /= w23;
  /* #59: @22 = (@19*@22) */
  w22  = (w19*w22);
  /* #60: @23 = 9.81 */
  w23 = 9.8100000000000005e+00;
  /* #61: @24 = (@8*@11) */
  w24  = (w8*w11);
  /* #62: @25 = (@13*@5) */
  w25  = (w13*w5);
  /* #63: @24 = (@24-@25) */
  w24 -= w25;
  /* #64: @24 = (2.*@24) */
  w24 = (2.* w24 );
  /* #65: @25 = (@24*@21) */
  w25  = (w24*w21);
  /* #66: @26 = 0.76 */
  w26 = 7.6000000000000001e-01;
  /* #67: @25 = (@25/@26) */
  w25 /= w26;
  /* #68: @25 = (@23*@25) */
  w25  = (w23*w25);
  /* #69: @26 = 9.81 */
  w26 = 9.8100000000000005e+00;
  /* #70: @27 = 1 */
  w27 = 1.;
  /* #71: @28 = (2.*@5) */
  w28 = (2.* w5 );
  /* #72: @29 = (@28*@5) */
  w29  = (w28*w5);
  /* #73: @27 = (@27-@29) */
  w27 -= w29;
  /* #74: @29 = (2.*@8) */
  w29 = (2.* w8 );
  /* #75: @30 = (@29*@8) */
  w30  = (w29*w8);
  /* #76: @27 = (@27-@30) */
  w27 -= w30;
  /* #77: @30 = (@27*@21) */
  w30  = (w27*w21);
  /* #78: @31 = 0.76 */
  w31 = 7.6000000000000001e-01;
  /* #79: @30 = (@30/@31) */
  w30 /= w31;
  /* #80: @30 = (@26*@30) */
  w30  = (w26*w30);
  /* #81: @31 = 9.81 */
  w31 = 9.8100000000000005e+00;
  /* #82: @30 = (@30-@31) */
  w30 -= w31;
  /* #83: @32 = vertcat(@0, @1, @2, @6, @14, @16, @18, @22, @25, @30) */
  rr=w32;
  *rr++ = w0;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w6;
  *rr++ = w14;
  *rr++ = w16;
  *rr++ = w18;
  *rr++ = w22;
  *rr++ = w25;
  *rr++ = w30;
  /* #84: @10 = (@10-@32) */
  for (i=0, rr=w10, cs=w32; i<10; ++i) (*rr++) -= (*cs++);
  /* #85: output[0][0] = @10 */
  casadi_copy(w10, 10, res[0]);
  /* #86: @33 = zeros(10x10,25nz) */
  casadi_clear(w33, 25);
  /* #87: @34 = ones(10x1,7nz) */
  casadi_fill(w34, 7, 1.);
  /* #88: {NULL, NULL, NULL, @0, NULL, NULL, NULL, @1, @2, @6} = vertsplit(@34) */
  w0 = w34[3];
  w1 = w34[4];
  w2 = w34[5];
  w6 = w34[6];
  /* #89: @35 = 00 */
  /* #90: @14 = (@4*@0) */
  w14  = (w4*w0);
  /* #91: @14 = (@12*@14) */
  w14  = (w12*w14);
  /* #92: @16 = (@7*@0) */
  w16  = (w7*w0);
  /* #93: @16 = (@15*@16) */
  w16  = (w15*w16);
  /* #94: @18 = (@9*@0) */
  w18  = (w9*w0);
  /* #95: @18 = (@17*@18) */
  w18  = (w17*w18);
  /* #96: @22 = 1.31579 */
  w22 = 1.3157894736842106e+00;
  /* #97: @25 = (@8*@0) */
  w25  = (w8*w0);
  /* #98: @25 = (2.*@25) */
  w25 = (2.* w25 );
  /* #99: @25 = (@21*@25) */
  w25  = (w21*w25);
  /* #100: @25 = (@22*@25) */
  w25  = (w22*w25);
  /* #101: @25 = (@19*@25) */
  w25  = (w19*w25);
  /* #102: @30 = 1.31579 */
  w30 = 1.3157894736842106e+00;
  /* #103: @0 = (@5*@0) */
  w0  = (w5*w0);
  /* #104: @0 = (-@0) */
  w0 = (- w0 );
  /* #105: @0 = (2.*@0) */
  w0 = (2.* w0 );
  /* #106: @0 = (@21*@0) */
  w0  = (w21*w0);
  /* #107: @0 = (@30*@0) */
  w0  = (w30*w0);
  /* #108: @0 = (@23*@0) */
  w0  = (w23*w0);
  /* #109: @36 = 00 */
  /* #110: @37 = vertcat(@1, @2, @6, @35, @14, @16, @18, @25, @0, @36) */
  rr=w37;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w6;
  *rr++ = w14;
  *rr++ = w16;
  *rr++ = w18;
  *rr++ = w25;
  *rr++ = w0;
  /* #111: @37 = (-@37) */
  for (i=0, rr=w37, cs=w37; i<8; ++i) *rr++ = (- *cs++ );
  /* #112: @38 = @37[:8] */
  for (rr=w38, ss=w37+0; ss!=w37+8; ss+=1) *rr++ = *ss;
  /* #113: (@33[0, 1, 2, 6, 9, 12, 15, 19] = @38) */
  for (cii=casadi_s0, rr=w33, ss=w38; cii!=casadi_s0+8; ++cii, ++ss) rr[*cii] = *ss;
  /* #114: @35 = 00 */
  /* #115: @36 = 00 */
  /* #116: @39 = 00 */
  /* #117: @1 = ones(10x1,1nz) */
  w1 = 1.;
  /* #118: {NULL, NULL, NULL, NULL, @2, NULL, NULL, NULL, NULL, NULL} = vertsplit(@1) */
  w2 = w1;
  /* #119: @1 = (@4*@2) */
  w1  = (w4*w2);
  /* #120: @1 = (@3*@1) */
  w1  = (w3*w1);
  /* #121: @1 = (-@1) */
  w1 = (- w1 );
  /* #122: @40 = 00 */
  /* #123: @6 = (@9*@2) */
  w6  = (w9*w2);
  /* #124: @6 = (@15*@6) */
  w6  = (w15*w6);
  /* #125: @6 = (-@6) */
  w6 = (- w6 );
  /* #126: @14 = (@7*@2) */
  w14  = (w7*w2);
  /* #127: @14 = (@17*@14) */
  w14  = (w17*w14);
  /* #128: @16 = (@11*@2) */
  w16  = (w11*w2);
  /* #129: @16 = (2.*@16) */
  w16 = (2.* w16 );
  /* #130: @16 = (@21*@16) */
  w16  = (w21*w16);
  /* #131: @16 = (@22*@16) */
  w16  = (w22*w16);
  /* #132: @16 = (@19*@16) */
  w16  = (w19*w16);
  /* #133: @18 = (@13*@2) */
  w18  = (w13*w2);
  /* #134: @18 = (-@18) */
  w18 = (- w18 );
  /* #135: @18 = (2.*@18) */
  w18 = (2.* w18 );
  /* #136: @18 = (@21*@18) */
  w18  = (w21*w18);
  /* #137: @18 = (@30*@18) */
  w18  = (w30*w18);
  /* #138: @18 = (@23*@18) */
  w18  = (w23*w18);
  /* #139: @25 = 1.31579 */
  w25 = 1.3157894736842106e+00;
  /* #140: @0 = (2.*@2) */
  w0 = (2.* w2 );
  /* #141: @0 = (@5*@0) */
  w0  = (w5*w0);
  /* #142: @28 = (@28*@2) */
  w28 *= w2;
  /* #143: @0 = (@0+@28) */
  w0 += w28;
  /* #144: @0 = (@21*@0) */
  w0  = (w21*w0);
  /* #145: @0 = (@25*@0) */
  w0  = (w25*w0);
  /* #146: @0 = (@26*@0) */
  w0  = (w26*w0);
  /* #147: @0 = (-@0) */
  w0 = (- w0 );
  /* #148: @41 = vertcat(@35, @36, @39, @1, @40, @6, @14, @16, @18, @0) */
  rr=w41;
  *rr++ = w1;
  *rr++ = w6;
  *rr++ = w14;
  *rr++ = w16;
  *rr++ = w18;
  *rr++ = w0;
  /* #149: @41 = (-@41) */
  for (i=0, rr=w41, cs=w41; i<6; ++i) *rr++ = (- *cs++ );
  /* #150: @42 = @41[:6] */
  for (rr=w42, ss=w41+0; ss!=w41+6; ss+=1) *rr++ = *ss;
  /* #151: (@33[3, 10, 13, 16, 20, 23] = @42) */
  for (cii=casadi_s1, rr=w33, ss=w42; cii!=casadi_s1+6; ++cii, ++ss) rr[*cii] = *ss;
  /* #152: @35 = 00 */
  /* #153: @36 = 00 */
  /* #154: @39 = 00 */
  /* #155: @1 = ones(10x1,1nz) */
  w1 = 1.;
  /* #156: {NULL, NULL, NULL, NULL, NULL, @6, NULL, NULL, NULL, NULL} = vertsplit(@1) */
  w6 = w1;
  /* #157: @1 = (@7*@6) */
  w1  = (w7*w6);
  /* #158: @1 = (@3*@1) */
  w1  = (w3*w1);
  /* #159: @1 = (-@1) */
  w1 = (- w1 );
  /* #160: @14 = (@9*@6) */
  w14  = (w9*w6);
  /* #161: @14 = (@12*@14) */
  w14  = (w12*w14);
  /* #162: @40 = 00 */
  /* #163: @16 = (@4*@6) */
  w16  = (w4*w6);
  /* #164: @16 = (@17*@16) */
  w16  = (w17*w16);
  /* #165: @16 = (-@16) */
  w16 = (- w16 );
  /* #166: @18 = (@13*@6) */
  w18  = (w13*w6);
  /* #167: @18 = (2.*@18) */
  w18 = (2.* w18 );
  /* #168: @18 = (@21*@18) */
  w18  = (w21*w18);
  /* #169: @18 = (@22*@18) */
  w18  = (w22*w18);
  /* #170: @18 = (@19*@18) */
  w18  = (w19*w18);
  /* #171: @0 = (@11*@6) */
  w0  = (w11*w6);
  /* #172: @0 = (2.*@0) */
  w0 = (2.* w0 );
  /* #173: @0 = (@21*@0) */
  w0  = (w21*w0);
  /* #174: @0 = (@30*@0) */
  w0  = (w30*w0);
  /* #175: @0 = (@23*@0) */
  w0  = (w23*w0);
  /* #176: @28 = (2.*@6) */
  w28 = (2.* w6 );
  /* #177: @28 = (@8*@28) */
  w28  = (w8*w28);
  /* #178: @29 = (@29*@6) */
  w29 *= w6;
  /* #179: @28 = (@28+@29) */
  w28 += w29;
  /* #180: @28 = (@21*@28) */
  w28  = (w21*w28);
  /* #181: @25 = (@25*@28) */
  w25 *= w28;
  /* #182: @25 = (@26*@25) */
  w25  = (w26*w25);
  /* #183: @25 = (-@25) */
  w25 = (- w25 );
  /* #184: @42 = vertcat(@35, @36, @39, @1, @14, @40, @16, @18, @0, @25) */
  rr=w42;
  *rr++ = w1;
  *rr++ = w14;
  *rr++ = w16;
  *rr++ = w18;
  *rr++ = w0;
  *rr++ = w25;
  /* #185: @42 = (-@42) */
  for (i=0, rr=w42, cs=w42; i<6; ++i) *rr++ = (- *cs++ );
  /* #186: @41 = @42[:6] */
  for (rr=w41, ss=w42+0; ss!=w42+6; ss+=1) *rr++ = *ss;
  /* #187: (@33[4, 7, 14, 17, 21, 24] = @41) */
  for (cii=casadi_s2, rr=w33, ss=w41; cii!=casadi_s2+6; ++cii, ++ss) rr[*cii] = *ss;
  /* #188: @35 = 00 */
  /* #189: @36 = 00 */
  /* #190: @39 = 00 */
  /* #191: @1 = ones(10x1,1nz) */
  w1 = 1.;
  /* #192: {NULL, NULL, NULL, NULL, NULL, NULL, @14, NULL, NULL, NULL} = vertsplit(@1) */
  w14 = w1;
  /* #193: @9 = (@9*@14) */
  w9 *= w14;
  /* #194: @9 = (@3*@9) */
  w9  = (w3*w9);
  /* #195: @9 = (-@9) */
  w9 = (- w9 );
  /* #196: @7 = (@7*@14) */
  w7 *= w14;
  /* #197: @7 = (@12*@7) */
  w7  = (w12*w7);
  /* #198: @7 = (-@7) */
  w7 = (- w7 );
  /* #199: @4 = (@4*@14) */
  w4 *= w14;
  /* #200: @4 = (@15*@4) */
  w4  = (w15*w4);
  /* #201: @40 = 00 */
  /* #202: @1 = (@5*@14) */
  w1  = (w5*w14);
  /* #203: @1 = (2.*@1) */
  w1 = (2.* w1 );
  /* #204: @1 = (@21*@1) */
  w1  = (w21*w1);
  /* #205: @22 = (@22*@1) */
  w22 *= w1;
  /* #206: @22 = (@19*@22) */
  w22  = (w19*w22);
  /* #207: @14 = (@8*@14) */
  w14  = (w8*w14);
  /* #208: @14 = (2.*@14) */
  w14 = (2.* w14 );
  /* #209: @21 = (@21*@14) */
  w21 *= w14;
  /* #210: @30 = (@30*@21) */
  w30 *= w21;
  /* #211: @30 = (@23*@30) */
  w30  = (w23*w30);
  /* #212: @43 = 00 */
  /* #213: @44 = vertcat(@35, @36, @39, @9, @7, @4, @40, @22, @30, @43) */
  rr=w44;
  *rr++ = w9;
  *rr++ = w7;
  *rr++ = w4;
  *rr++ = w22;
  *rr++ = w30;
  /* #214: @44 = (-@44) */
  for (i=0, rr=w44, cs=w44; i<5; ++i) *rr++ = (- *cs++ );
  /* #215: @45 = @44[:5] */
  for (rr=w45, ss=w44+0; ss!=w44+5; ss+=1) *rr++ = *ss;
  /* #216: (@33[5, 8, 11, 18, 22] = @45) */
  for (cii=casadi_s3, rr=w33, ss=w45; cii!=casadi_s3+5; ++cii, ++ss) rr[*cii] = *ss;
  /* #217: @46 = @33' */
  casadi_trans(w33,casadi_s5, w46, casadi_s4, iw);
  /* #218: output[1][0] = @46 */
  casadi_copy(w46, 25, res[1]);
  /* #219: @10 = zeros(10x10,10nz) */
  casadi_clear(w10, 10);
  /* #220: @32 = ones(10x1) */
  casadi_fill(w32, 10, 1.);
  /* #221: (@10[:10] = @32) */
  for (rr=w10+0, ss=w32; rr!=w10+10; rr+=1) *rr = *ss++;
  /* #222: @32 = @10' */
  casadi_trans(w10,casadi_s6, w32, casadi_s6, iw);
  /* #223: output[2][0] = @32 */
  casadi_copy(w32, 10, res[2]);
  /* #224: @47 = zeros(4x10,15nz) */
  casadi_clear(w47, 15);
  /* #225: @35 = 00 */
  /* #226: @36 = 00 */
  /* #227: @39 = 00 */
  /* #228: @48 = ones(4x1,2nz) */
  casadi_fill(w48, 2, 1.);
  /* #229: {@9, @7, NULL, NULL} = vertsplit(@48) */
  w9 = w48[0];
  w7 = w48[1];
  /* #230: @4 = (@5*@7) */
  w4  = (w5*w7);
  /* #231: @4 = (@3*@4) */
  w4  = (w3*w4);
  /* #232: @4 = (-@4) */
  w4 = (- w4 );
  /* #233: @22 = (@13*@7) */
  w22  = (w13*w7);
  /* #234: @22 = (@12*@22) */
  w22  = (w12*w22);
  /* #235: @30 = (@11*@7) */
  w30  = (w11*w7);
  /* #236: @30 = (@15*@30) */
  w30  = (w15*w30);
  /* #237: @7 = (@8*@7) */
  w7  = (w8*w7);
  /* #238: @7 = (@17*@7) */
  w7  = (w17*w7);
  /* #239: @7 = (-@7) */
  w7 = (- w7 );
  /* #240: @21 = 1.31579 */
  w21 = 1.3157894736842106e+00;
  /* #241: @20 = (@20*@9) */
  w20 *= w9;
  /* #242: @21 = (@21*@20) */
  w21 *= w20;
  /* #243: @19 = (@19*@21) */
  w19 *= w21;
  /* #244: @21 = 1.31579 */
  w21 = 1.3157894736842106e+00;
  /* #245: @24 = (@24*@9) */
  w24 *= w9;
  /* #246: @21 = (@21*@24) */
  w21 *= w24;
  /* #247: @23 = (@23*@21) */
  w23 *= w21;
  /* #248: @21 = 1.31579 */
  w21 = 1.3157894736842106e+00;
  /* #249: @27 = (@27*@9) */
  w27 *= w9;
  /* #250: @21 = (@21*@27) */
  w21 *= w27;
  /* #251: @26 = (@26*@21) */
  w26 *= w21;
  /* #252: @34 = vertcat(@35, @36, @39, @4, @22, @30, @7, @19, @23, @26) */
  rr=w34;
  *rr++ = w4;
  *rr++ = w22;
  *rr++ = w30;
  *rr++ = w7;
  *rr++ = w19;
  *rr++ = w23;
  *rr++ = w26;
  /* #253: @34 = (-@34) */
  for (i=0, rr=w34, cs=w34; i<7; ++i) *rr++ = (- *cs++ );
  /* #254: @49 = @34[:7] */
  for (rr=w49, ss=w34+0; ss!=w34+7; ss+=1) *rr++ = *ss;
  /* #255: (@47[0, 3, 6, 9, 12, 13, 14] = @49) */
  for (cii=casadi_s7, rr=w47, ss=w49; cii!=casadi_s7+7; ++cii, ++ss) rr[*cii] = *ss;
  /* #256: @35 = 00 */
  /* #257: @36 = 00 */
  /* #258: @39 = 00 */
  /* #259: @4 = ones(4x1,1nz) */
  w4 = 1.;
  /* #260: {NULL, NULL, @22, NULL} = vertsplit(@4) */
  w22 = w4;
  /* #261: @4 = (@8*@22) */
  w4  = (w8*w22);
  /* #262: @4 = (@3*@4) */
  w4  = (w3*w4);
  /* #263: @4 = (-@4) */
  w4 = (- w4 );
  /* #264: @30 = (@11*@22) */
  w30  = (w11*w22);
  /* #265: @30 = (@12*@30) */
  w30  = (w12*w30);
  /* #266: @30 = (-@30) */
  w30 = (- w30 );
  /* #267: @7 = (@13*@22) */
  w7  = (w13*w22);
  /* #268: @7 = (@15*@7) */
  w7  = (w15*w7);
  /* #269: @22 = (@5*@22) */
  w22  = (w5*w22);
  /* #270: @22 = (@17*@22) */
  w22  = (w17*w22);
  /* #271: @40 = 00 */
  /* #272: @43 = 00 */
  /* #273: @50 = 00 */
  /* #274: @51 = vertcat(@35, @36, @39, @4, @30, @7, @22, @40, @43, @50) */
  rr=w51;
  *rr++ = w4;
  *rr++ = w30;
  *rr++ = w7;
  *rr++ = w22;
  /* #275: @51 = (-@51) */
  for (i=0, rr=w51, cs=w51; i<4; ++i) *rr++ = (- *cs++ );
  /* #276: @52 = @51[:4] */
  for (rr=w52, ss=w51+0; ss!=w51+4; ss+=1) *rr++ = *ss;
  /* #277: (@47[1:13:3] = @52) */
  for (rr=w47+1, ss=w52; rr!=w47+13; rr+=3) *rr = *ss++;
  /* #278: @35 = 00 */
  /* #279: @36 = 00 */
  /* #280: @39 = 00 */
  /* #281: @4 = ones(4x1,1nz) */
  w4 = 1.;
  /* #282: {NULL, NULL, NULL, @30} = vertsplit(@4) */
  w30 = w4;
  /* #283: @11 = (@11*@30) */
  w11 *= w30;
  /* #284: @3 = (@3*@11) */
  w3 *= w11;
  /* #285: @3 = (-@3) */
  w3 = (- w3 );
  /* #286: @8 = (@8*@30) */
  w8 *= w30;
  /* #287: @12 = (@12*@8) */
  w12 *= w8;
  /* #288: @5 = (@5*@30) */
  w5 *= w30;
  /* #289: @15 = (@15*@5) */
  w15 *= w5;
  /* #290: @15 = (-@15) */
  w15 = (- w15 );
  /* #291: @13 = (@13*@30) */
  w13 *= w30;
  /* #292: @17 = (@17*@13) */
  w17 *= w13;
  /* #293: @40 = 00 */
  /* #294: @43 = 00 */
  /* #295: @50 = 00 */
  /* #296: @52 = vertcat(@35, @36, @39, @3, @12, @15, @17, @40, @43, @50) */
  rr=w52;
  *rr++ = w3;
  *rr++ = w12;
  *rr++ = w15;
  *rr++ = w17;
  /* #297: @52 = (-@52) */
  for (i=0, rr=w52, cs=w52; i<4; ++i) *rr++ = (- *cs++ );
  /* #298: @51 = @52[:4] */
  for (rr=w51, ss=w52+0; ss!=w52+4; ss+=1) *rr++ = *ss;
  /* #299: (@47[2:14:3] = @51) */
  for (rr=w47+2, ss=w51; rr!=w47+14; rr+=3) *rr = *ss++;
  /* #300: @53 = @47' */
  casadi_trans(w47,casadi_s9, w53, casadi_s8, iw);
  /* #301: output[3][0] = @53 */
  casadi_copy(w53, 15, res[3]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Translational_drone_impl_dae_fun_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Translational_drone_impl_dae_fun_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Translational_drone_impl_dae_fun_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Translational_drone_impl_dae_fun_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Translational_drone_impl_dae_fun_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Translational_drone_impl_dae_fun_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Translational_drone_impl_dae_fun_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void Translational_drone_impl_dae_fun_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Translational_drone_impl_dae_fun_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int Translational_drone_impl_dae_fun_jac_x_xdot_u_z_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Translational_drone_impl_dae_fun_jac_x_xdot_u_z_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Translational_drone_impl_dae_fun_jac_x_xdot_u_z_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Translational_drone_impl_dae_fun_jac_x_xdot_u_z_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Translational_drone_impl_dae_fun_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s10;
    case 2: return casadi_s11;
    case 3: return casadi_s12;
    case 4: return casadi_s12;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Translational_drone_impl_dae_fun_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s10;
    case 1: return casadi_s4;
    case 2: return casadi_s6;
    case 3: return casadi_s8;
    case 4: return casadi_s13;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Translational_drone_impl_dae_fun_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 15;
  if (sz_res) *sz_res = 15;
  if (sz_iw) *sz_iw = 11;
  if (sz_w) *sz_w = 193;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
