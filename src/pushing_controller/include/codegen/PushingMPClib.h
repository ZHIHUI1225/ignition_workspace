//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// PushingMPClib.h
//
// Code generation for function 'PushingMPClib'
//

#ifndef PUSHINGMPCLIB_H
#define PUSHINGMPCLIB_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Type Declarations
struct struct4_T;

struct struct5_T;

struct struct11_T;

struct struct12_T;

struct struct14_T;

struct struct15_T;

struct struct20_T;

struct struct21_T;

// Function Declarations
extern void PushingMPClib_initialize();

extern void PushingMPClib_terminate();

extern void pushMove(double xk[5], const double v[182], const double oldu[52],
                     struct4_T *stateData, struct5_T *onlineData,
                     struct11_T *param, double vel_cmd[2], double mv[2],
                     double seq[52], struct4_T *newstateData, struct12_T *info,
                     double *iter, double xopt[130]);

extern void pushMoveObj(const double xk[3], const double v[130],
                        const double oldu[52], struct14_T *stateData,
                        struct15_T *onlineData, struct20_T *param,
                        double vel_cmd[2], double mv[2], double seq[52],
                        struct14_T *newstateData, struct21_T *info,
                        double *iter, double xopt[78]);

#endif
// End of code generation (PushingMPClib.h)
