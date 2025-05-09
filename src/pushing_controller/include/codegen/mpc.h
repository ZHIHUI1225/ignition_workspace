#pragma once
#include "PushingMPClib.h"
#include "PushingMPClib_types.h"
#include "rt_nonfinite.h"
#include <algorithm>

#define P_HOR 25
// // Function Declarations
static void argInit_11x1_real_T(double result[11]);

static void argInit_11x2_real_T(double result[22]);

static void argInit_11x3_real_T(double result[33]);

static void argInit_11x5_real_T(double result[55]);

static void argInit_13x1_real_T(double result[13]);

static void argInit_13x2_real_T(double result[26]);

static void argInit_13x7_real_T(double result[91]);

static void argInit_25x2_real_T(double result[50]);

static void argInit_25x3_real_T(double result[75]);

static void argInit_25x7_real_T(double result[175]);

static void argInit_26x2_real_T(double result[52]);

static void argInit_26x5_real_T(double result[130]);

static void argInit_26x7_real_T(double result[182]);

static void argInit_2x1_real_T(double result[2]);

static void argInit_310x1_boolean_T(boolean_T result[310]);

static void argInit_3x1_real_T(double result[3]);

static void argInit_3x1x26_real_T(double result[78]);

static void argInit_3x3x26_real_T(double result[234]);

static void argInit_3x7x26_real_T(double result[546]);

static void argInit_456x1_boolean_T(boolean_T result[456]);

static void argInit_5x1_real_T(double result[5]);

static void argInit_5x1x26_real_T(double result[130]);

static void argInit_5x5x26_real_T(double result[650]);

static void argInit_5x9x26_real_T(double result[1170]);

static void argInit_7x1_real_T(double result[7]);

static void argInit_7x1x26_real_T(double result[182]);

static void argInit_7x5x26_real_T(double result[910]);

static void argInit_7x9x26_real_T(double result[1638]);

static void argInit_9x1x26_real_T(double result[234]);

static boolean_T argInit_boolean_T();

static double argInit_real_T();

static void argInit_struct4_T(struct4_T *result);

static void argInit_struct5_T(struct5_T *result);

static void argInit_struct6_T(struct6_T *result);

static void argInit_struct7_T(struct7_T *result);

static void argInit_struct9_T(struct9_T *result);

static void argInit_struct10_T(struct10_T *result);

static void argInit_struct11_T(struct11_T *result);

static void argInit_struct14_T(struct14_T *result);

static void argInit_struct15_T(struct15_T *result);

static void argInit_struct16_T(struct16_T *result);

static void argInit_struct17_T(struct17_T *result);

static void argInit_struct18_T(struct18_T *result);

static void argInit_struct19_T(struct19_T *result);

static void argInit_struct20_T(struct20_T *result);

static void argInit_11x1_real_T(double result[11])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 11; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

static void argInit_11x2_real_T(double result[22])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 11; idx0++) {
    for (int idx1{0}; idx1 < 2; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 11 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_11x3_real_T(double result[33])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 11; idx0++) {
    for (int idx1{0}; idx1 < 3; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 11 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_11x5_real_T(double result[55])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 11; idx0++) {
    for (int idx1{0}; idx1 < 5; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 11 * idx1] = argInit_real_T();
    }
  }
}
static void argInit_13x1_real_T(double result[13])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 13; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

static void argInit_13x2_real_T(double result[26])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 13; idx0++) {
    for (int idx1{0}; idx1 < 2; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 13 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_13x7_real_T(double result[91])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 13; idx0++) {
    for (int idx1{0}; idx1 < 7; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 13 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_25x2_real_T(double result[50])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 25; idx0++) {
    for (int idx1{0}; idx1 < 2; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 25 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_25x3_real_T(double result[75])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 25; idx0++) {
    for (int idx1{0}; idx1 < 3; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 25 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_25x7_real_T(double result[175])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 25; idx0++) {
    for (int idx1{0}; idx1 < 7; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 25 * idx1] = argInit_real_T();
    }
  }
}

inline static void argInit_26x2_real_T(double result[52])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 26; idx0++) {
    for (int idx1{0}; idx1 < 2; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 26 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_26x5_real_T(double result[130])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 26; idx0++) {
    for (int idx1{0}; idx1 < 5; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 26 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_26x7_real_T(double result[182])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 26; idx0++) {
    for (int idx1{0}; idx1 < 7; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 26 * idx1] = argInit_real_T();
    }
  }
}

static void argInit_2x1_real_T(double result[2])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 2; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

static void argInit_310x1_boolean_T(boolean_T result[310])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 310; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_boolean_T();
  }
}

static void argInit_3x1_real_T(double result[3])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 3; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

static void argInit_3x1x26_real_T(double result[78])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 3; idx0++) {
    for (int idx2{0}; idx2 < 26; idx2++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 3 * idx2] = argInit_real_T();
    }
  }
}

static void argInit_3x3x26_real_T(double result[234])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 3; idx0++) {
    for (int idx1{0}; idx1 < 3; idx1++) {
      for (int idx2{0}; idx2 < 26; idx2++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[(idx0 + 3 * idx1) + 9 * idx2] = argInit_real_T();
      }
    }
  }
}

static void argInit_3x7x26_real_T(double result[546])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 3; idx0++) {
    for (int idx1{0}; idx1 < 7; idx1++) {
      for (int idx2{0}; idx2 < 26; idx2++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[(idx0 + 3 * idx1) + 21 * idx2] = argInit_real_T();
      }
    }
  }
}

static void argInit_456x1_boolean_T(boolean_T result[456])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 456; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_boolean_T();
  }
}

static void argInit_5x1_real_T(double result[5])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 5; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

static void argInit_5x1x26_real_T(double result[130])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 5; idx0++) {
    for (int idx2{0}; idx2 < 26; idx2++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 5 * idx2] = argInit_real_T();
    }
  }
}

static void argInit_5x5x26_real_T(double result[650])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 5; idx0++) {
    for (int idx1{0}; idx1 < 5; idx1++) {
      for (int idx2{0}; idx2 < 26; idx2++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[(idx0 + 5 * idx1) + 25 * idx2] = argInit_real_T();
      }
    }
  }
}

static void argInit_5x9x26_real_T(double result[1170])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 5; idx0++) {
    for (int idx1{0}; idx1 < 9; idx1++) {
      for (int idx2{0}; idx2 < 26; idx2++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[(idx0 + 5 * idx1) + 45 * idx2] = argInit_real_T();
      }
    }
  }
}

static void argInit_7x1_real_T(double result[7])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 7; idx0++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx0] = argInit_real_T();
  }
}

static void argInit_7x1x26_real_T(double result[182])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 7; idx0++) {
    for (int idx2{0}; idx2 < 26; idx2++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 7 * idx2] = argInit_real_T();
    }
  }
}

static void argInit_7x5x26_real_T(double result[910])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 7; idx0++) {
    for (int idx1{0}; idx1 < 5; idx1++) {
      for (int idx2{0}; idx2 < 26; idx2++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[(idx0 + 7 * idx1) + 35 * idx2] = argInit_real_T();
      }
    }
  }
}

static void argInit_7x9x26_real_T(double result[1638])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 7; idx0++) {
    for (int idx1{0}; idx1 < 9; idx1++) {
      for (int idx2{0}; idx2 < 26; idx2++) {
        // Set the value of the array element.
        // Change this value to the value that the application requires.
        result[(idx0 + 7 * idx1) + 63 * idx2] = argInit_real_T();
      }
    }
  }
}

static void argInit_9x1x26_real_T(double result[234])
{
  // Loop over the array to initialize each element.
  for (int idx0{0}; idx0 < 9; idx0++) {
    for (int idx2{0}; idx2 < 26; idx2++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 9 * idx2] = argInit_real_T();
    }
  }
}

inline static void argInit_struct5_T(struct5_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_struct6_T(&result->signals);
  argInit_struct7_T(&result->weights);
  argInit_struct9_T(&result->customconstraints);
  argInit_struct10_T(&result->model);
}
inline static void argInit_struct4_T(struct4_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_5x1_real_T(result->Plant);
  argInit_2x1_real_T(result->LastMove);
  argInit_456x1_boolean_T(result->iA);
}
static boolean_T argInit_boolean_T()
{
  return false;
}

static double argInit_real_T()
{
  return 0.0;
}

static void argInit_struct10_T(struct10_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_5x1x26_real_T(result->X);
  argInit_5x5x26_real_T(result->A);
  argInit_5x9x26_real_T(result->B);
  argInit_7x5x26_real_T(result->C);
  argInit_7x9x26_real_T(result->D);
  argInit_9x1x26_real_T(result->U);
  argInit_7x1x26_real_T(result->Y);
  std::copy(&result->X[0], &result->X[130], &result->DX[0]);
}

inline static void argInit_struct11_T(struct11_T *result)
{
  double result_tmp;
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  result_tmp = argInit_real_T();
  result->y_or = result_tmp;
  result->L = result_tmp;
  result->R = result_tmp;
  result->constraints = result_tmp;
  result->J_o = result_tmp;
  result->J_r = result_tmp;
  result->M_r = result_tmp;
  result->M_o = result_tmp;
  result->mu_o = result_tmp;
  result->mu_i = result_tmp;
  result->s_o = result_tmp;
  result->gamma_u = result_tmp;
  result->gamma_l = result_tmp;
  argInit_2x1_real_T(result->q1);
  result->safe_d = result_tmp;
  result->x_or = result_tmp;
  result->q2[0] = result->q1[0];
  result->q3[0] = result->q1[0];
  result->q4[0] = result->q1[0];
  result->q5[0] = result->q1[0];
  result->q1ad[0] = result->q1[0];
  result->q2ad[0] = result->q1[0];
  result->q3ad[0] = result->q1[0];
  result->q4ad[0] = result->q1[0];
  result->q5ad[0] = result->q1[0];
  result->x0[0] = result->q1[0];
  result->q2[1] = result->q1[1];
  result->q3[1] = result->q1[1];
  result->q4[1] = result->q1[1];
  result->q5[1] = result->q1[1];
  result->q1ad[1] = result->q1[1];
  result->q2ad[1] = result->q1[1];
  result->q3ad[1] = result->q1[1];
  result->q4ad[1] = result->q1[1];
  result->q5ad[1] = result->q1[1];
  result->x0[1] = result->q1[1];
}

inline static void argInit_struct14_T(struct14_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_3x1_real_T(result->Plant);
  argInit_2x1_real_T(result->LastMove);
  argInit_310x1_boolean_T(result->iA);
}

inline static void argInit_struct15_T(struct15_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_struct16_T(&result->signals);
  argInit_struct17_T(&result->weights);
  argInit_struct18_T(&result->customconstraints);
  argInit_struct19_T(&result->model);
}

static void argInit_struct16_T(struct16_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_3x1_real_T(result->ym);
  argInit_3x1_real_T(result->ref);
  argInit_26x5_real_T(result->md);
}

static void argInit_struct17_T(struct17_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_25x2_real_T(result->u);
  argInit_25x3_real_T(result->y);
  std::copy(&result->u[0], &result->u[50], &result->du[0]);
}

static void argInit_struct18_T(struct18_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_11x2_real_T(result->E);
  argInit_11x3_real_T(result->F);
  argInit_11x1_real_T(result->G);
  argInit_11x5_real_T(result->S);
}

static void argInit_struct19_T(struct19_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_3x3x26_real_T(result->A);
  argInit_3x7x26_real_T(result->B);
  argInit_3x1x26_real_T(result->X);
  argInit_7x1x26_real_T(result->U);
  std::copy(&result->A[0], &result->A[234], &result->C[0]);
  std::copy(&result->B[0], &result->B[546], &result->D[0]);
  for (int i{0}; i < 78; i++) {
    double d;
    d = result->X[i];
    result->Y[i] = d;
    result->DX[i] = d;
  }
}

inline static void argInit_struct20_T(struct20_T *result)
{
  double result_tmp;
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  result_tmp = argInit_real_T();
  result->y_or = result_tmp;
  result->L = result_tmp;
  result->R = result_tmp;
  result->W_lim = result_tmp;
  result->obstacles = result_tmp;
  result->Rlim_m = result_tmp;
  result->Rlim_p = result_tmp;
  argInit_2x1_real_T(result->q1);
  result->safe_d = result_tmp;
  result->constraints = result_tmp;
  result->x_or = result_tmp;
  result->q2[0] = result->q1[0];
  result->q3[0] = result->q1[0];
  result->q4[0] = result->q1[0];
  result->q5[0] = result->q1[0];
  result->q1ad[0] = result->q1[0];
  result->q2ad[0] = result->q1[0];
  result->q3ad[0] = result->q1[0];
  result->q4ad[0] = result->q1[0];
  result->q5ad[0] = result->q1[0];
  result->p0[0] = result->q1[0];
  result->x0[0] = result->q1[0];
  result->q2[1] = result->q1[1];
  result->q3[1] = result->q1[1];
  result->q4[1] = result->q1[1];
  result->q5[1] = result->q1[1];
  result->q1ad[1] = result->q1[1];
  result->q2ad[1] = result->q1[1];
  result->q3ad[1] = result->q1[1];
  result->q4ad[1] = result->q1[1];
  result->q5ad[1] = result->q1[1];
  result->p0[1] = result->q1[1];
  result->x0[1] = result->q1[1];
}
static void argInit_struct6_T(struct6_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_7x1_real_T(result->ym);
  argInit_7x1_real_T(result->ref);
  argInit_26x7_real_T(result->md);
}

static void argInit_struct7_T(struct7_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_25x2_real_T(result->u);
  argInit_25x7_real_T(result->y);
  std::copy(&result->u[0], &result->u[50], &result->du[0]);
}

static void argInit_struct9_T(struct9_T *result)
{
  // Set the value of each structure field.
  // Change this value to the value that the application requires.
  argInit_13x7_real_T(result->F);
  argInit_13x2_real_T(result->E);
  argInit_13x1_real_T(result->G);
  std::copy(&result->F[0], &result->F[35], &result->S[0]);
}
