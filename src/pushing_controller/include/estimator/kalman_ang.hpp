#pragma once

#include <Eigen/Geometry>

namespace filters {
using namespace Eigen;
typedef Matrix<double, 10, 10> Matrix10d;
typedef Matrix<double, 10, 1> Vector10d;
typedef Matrix<double, 10, 4> Matrix10x4d;
typedef Matrix<double, 4, 10> Matrix4x10d;
class KalmanAng {
public:
  explicit KalmanAng(const Vector10d &xin, const Matrix10d &Pin,
                     const Matrix10d &Qin, const Matrix4d &Rin, double dt);
  void predict();
  void update(const Vector4d &y);

private:
  Vector10d x_;
  Matrix10d P_;
  Matrix10d Q_;
  Matrix4d R_;
  Matrix10x4d K_;
  Matrix10d A_;
  Matrix4x10d C_;
  Matrix10d I_;
  double dt_;
  double dt2_;
};

} // namespace filters
