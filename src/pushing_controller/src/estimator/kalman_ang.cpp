#include <estimator/kalman_ang.hpp>
namespace filters
{
  KalmanAng::KalmanAng(const Vector10d& xin, const Matrix10d& Pin, const Matrix10d& Qin, const Matrix4d& Rin, double dt) :
      x_(xin), P_(Pin), Q_(Qin), R_(Rin), dt_(dt), dt2_(dt * dt)

  {
    K_.setZero();
    A_.setZero();
    A_.block<3, 3>(4, 4).setIdentity();
    A_.block<3, 3>(7, 7).setIdentity();
    A_.block<3, 3>(4, 7) = Matrix3d::Identity() * dt;
    I_.setIdentity();
    C_.setZero();
    C_.block<4, 4>(0, 0) = Matrix4d::Identity();
  }

  void KalmanAng::update(const Vector4d& y)
  {
    x_ = x_ + K_ * (y - C_ * x_);
    x_.block<4, 1>(0, 0).normalize();
    P_ = (I_ - K_ * C_) * P_;
  }
  void KalmanAng::predict()
  {
    static double x0 = x_(0);
    static double x1 = x_(1);
    static double x2 = x_(2);
    static double x3 = x_(3);
    static double x4 = x_(4);
    static double x5 = x_(5);
    static double x6 = x_(6);
    static double x7 = x_(7);
    static double x8 = x_(8);
    static double x9 = x_(9);

    static Matrix4x10d quat_block;
    quat_block << 1 - (dt2_ * (x4 * x4 / 2 + x5 * x5 / 2 + x6 * x6 / 2)) / 4, -(dt_ * x4) / 2 - (dt2_ * x7) / 4,
        -(dt_ * x5) / 2 - (dt2_ * x8) / 4, -(dt_ * x6) / 2 - (dt2_ * x9) / 4, -(dt_ * x1) / 2 - (dt2_ * x0 * x4) / 4,
        -(dt_ * x2) / 2 - (dt2_ * x0 * x5) / 4, -(dt_ * x3) / 2 - (dt2_ * x0 * x6) / 4, -(dt2_ * x1) / 4, -(dt2_ * x2) / 4,
        -(dt2_ * x3) / 4, (dt_ * x4) / 2 + (dt2_ * x7) / 4, 1 - (dt2_ * (x4 * x4 / 2 + x5 * x5 / 2 + x6 * x6 / 2)) / 4,
        -(dt_ * x6) / 2 - (dt2_ * x9) / 4, (dt_ * x5) / 2 + (dt2_ * x8) / 4, (dt_ * x0) / 2 - (dt2_ * x1 * x4) / 4,
        (dt_ * x3) / 2 - (dt2_ * x1 * x5) / 4, -(dt_ * x2) / 2 - (dt2_ * x1 * x6) / 4, (dt2_ * x0) / 4, (dt2_ * x3) / 4, -(dt2_ * x2) / 4,
        (dt_ * x5) / 2 + (dt2_ * x8) / 4, (dt_ * x6) / 2 + (dt2_ * x9) / 4, 1 - (dt2_ * (x4 * x4 / 2 + x5 * x5 / 2 + x6 * x6 / 2)) / 4,
        -(dt_ * x4) / 2 - (dt2_ * x7) / 4, -(dt_ * x3) / 2 - (dt2_ * x2 * x4) / 4, (dt_ * x0) / 2 - (dt2_ * x2 * x5) / 4,
        (dt_ * x1) / 2 - (dt2_ * x2 * x6) / 4, -(dt2_ * x3) / 4, (dt2_ * x0) / 4, (dt2_ * x1) / 4, (dt_ * x6) / 2 + (dt2_ * x9) / 4,
        -(dt_ * x5) / 2 - (dt2_ * x8) / 4, (dt_ * x4) / 2 + (dt2_ * x7) / 4, 1 - (dt2_ * (x4 * x4 / 2 + x5 * x5 / 2 + x6 * x6 / 2)) / 4,
        (dt_ * x2) / 2 - (dt2_ * x3 * x4) / 4, -(dt_ * x1) / 2 - (dt2_ * x3 * x5) / 4, (dt_ * x0) / 2 - (dt2_ * x3 * x6) / 4,
        (dt2_ * x2) / 4, -(dt2_ * x1) / 4, (dt2_ * x0) / 4;

    A_.block<4, 10>(0, 0) = quat_block;

    // x = A*x;

    x_(0, 0) = x0 - (dt_ * (x1 * x4 + x2 * x5 + x3 * x6)) / 2 -
               (dt2_ * (x4 * ((x0 * x4) / 2 - (x2 * x6) / 2 + (x3 * x5) / 2) + x5 * ((x0 * x5) / 2 + (x1 * x6) / 2 - (x3 * x4) / 2) +
                        x6 * ((x0 * x6) / 2 - (x1 * x5) / 2 + (x2 * x4) / 2) + x1 * x7 + x2 * x8 + x3 * x9)) /
                   4;
    x_(1, 0) = x1 + (dt_ * (x0 * x4 - x2 * x6 + x3 * x5)) / 2 +
               (dt2_ * (x5 * ((x0 * x6) / 2 - (x1 * x5) / 2 + (x2 * x4) / 2) - x4 * ((x1 * x4) / 2 + (x2 * x5) / 2 + (x3 * x6) / 2) -
                        x6 * ((x0 * x5) / 2 + (x1 * x6) / 2 - (x3 * x4) / 2) + x0 * x7 - x2 * x9 + x3 * x8)) /
                   4;
    x_(2, 0) = x2 + (dt_ * (x0 * x5 + x1 * x6 - x3 * x4)) / 2 -
               (dt2_ * (x4 * ((x0 * x6) / 2 - (x1 * x5) / 2 + (x2 * x4) / 2) + x5 * ((x1 * x4) / 2 + (x2 * x5) / 2 + (x3 * x6) / 2) -
                        x6 * ((x0 * x4) / 2 - (x2 * x6) / 2 + (x3 * x5) / 2) - x0 * x8 - x1 * x9 + x3 * x7)) /
                   4;
    x_(3, 0) = x3 + (dt_ * (x0 * x6 - x1 * x5 + x2 * x4)) / 2 +
               (dt2_ * (x4 * ((x0 * x5) / 2 + (x1 * x6) / 2 - (x3 * x4) / 2) - x5 * ((x0 * x4) / 2 - (x2 * x6) / 2 + (x3 * x5) / 2) -
                        x6 * ((x1 * x4) / 2 + (x2 * x5) / 2 + (x3 * x6) / 2) + x0 * x9 - x1 * x8 + x2 * x7)) /
                   4;
    x_(4, 0) = x4 + x7 * dt_;
    x_(5, 0) = x5 + x8 * dt_;
    x_(6, 0) = x6 + x9 * dt_;
    x_(7, 0) = x7;
    x_(8, 0) = x8;
    x_(9, 0) = x9;

    x_.block<4, 1>(0, 0).normalize();

    P_ = A_ * P_ * (A_.transpose()) + Q_;
    K_ = P_ * (C_.transpose()) * ((C_ * P_ * (C_.transpose()) + R_).inverse());
  }
} // namespace filters
