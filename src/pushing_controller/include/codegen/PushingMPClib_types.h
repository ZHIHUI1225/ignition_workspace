//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// PushingMPClib_types.h
//
// Code generation for function 'pushMove'
//

#ifndef PUSHINGMPCLIB_TYPES_H
#define PUSHINGMPCLIB_TYPES_H

// Include files
#include "rtwtypes.h"

// Type Definitions
struct struct11_T {
  double x_or;
  double y_or;
  double L;
  double R;
  double constraints;
  double J_o;
  double J_r;
  double M_r;
  double M_o;
  double mu_o;
  double mu_i;
  double s_o;
  double gamma_u;
  double gamma_l;
  double q1[2];
  double q2[2];
  double q3[2];
  double q4[2];
  double q5[2];
  double q1ad[2];
  double q2ad[2];
  double q3ad[2];
  double q4ad[2];
  double q5ad[2];
  double x0[2];
  double safe_d;
};

struct struct20_T {
  double x_or;
  double y_or;
  double L;
  double R;
  double W_lim;
  double obstacles;
  double Rlim_m;
  double Rlim_p;
  double q1[2];
  double q2[2];
  double q3[2];
  double q4[2];
  double q5[2];
  double q1ad[2];
  double q2ad[2];
  double q3ad[2];
  double q4ad[2];
  double q5ad[2];
  double p0[2];
  double x0[2];
  double safe_d;
  double constraints;
};

struct struct9_T {
  double E[10];
  double F[35];
  double G[5];
  double S[35];
};

struct struct10_T {
  double A[650];
  double B[1170];
  double C[910];
  double D[1638];
  double X[130];
  double U[234];
  double Y[182];
  double DX[130];
};

struct struct12_T {
  double Uopt[52];
  double Yopt[182];
  double Xopt[130];
  double Topt[26];
  double Slack;
  double Iterations;
  double Cost;
};

struct struct18_T {
  double E[22];
  double F[33];
  double G[11];
  double S[55];
};

struct struct19_T {
  double A[234];
  double B[546];
  double C[234];
  double D[546];
  double X[78];
  double U[182];
  double Y[78];
  double DX[78];
};

struct struct21_T {
  double Uopt[52];
  double Yopt[78];
  double Xopt[78];
  double Topt[26];
  double Slack;
  double Iterations;
  double Cost;
};

struct struct6_T {
  double ym[7];
  double ref[7];
  double md[182];
};

struct struct7_T {
  double y[175];
  double u[50];
  double du[50];
};

struct struct5_T {
  struct6_T signals;
  struct7_T weights;
  struct9_T customconstraints;
  struct10_T model;
};

struct struct4_T {
  double Plant[5];
  double LastMove[2];
  boolean_T iA[248];
};

struct struct16_T {
  double ym[3];
  double ref[3];
  double md[130];
};

struct struct17_T {
  double y[75];
  double u[50];
  double du[50];
};

struct struct15_T {
  struct16_T signals;
  struct17_T weights;
  struct18_T customconstraints;
  struct19_T model;
};

struct struct14_T {
  double Plant[3];
  double LastMove[2];
  boolean_T iA[310];
};

#endif
// End of code generation (PushingMPClib_types.h)
