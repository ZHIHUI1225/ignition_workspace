#pragma once

#include <Eigen/Geometry>
namespace filters {
using namespace Eigen;

typedef Matrix<double, 9, 9> Matrix9d;
typedef Matrix<double, 9, 1> Vector9d;
typedef Matrix<double, 9, 3> Matrix93d;
typedef Matrix<double, 3, 9> Matrix39d;

class KalmanLin {
public:
  explicit KalmanLin(const Vector9d &xin, const Matrix9d &Pin,
                     const Matrix9d &Qin, const Matrix3d &Rin, double dt);
  void predict();
  void update(const Vector3d &y);

private:
  Vector9d x_;
  Matrix9d P_;
  Matrix9d Q_;
  Matrix3d R_;
  Matrix93d K_;
  Matrix9d A_;
  Matrix39d C_;
  Matrix9d I_;
};
} // namespace filters
