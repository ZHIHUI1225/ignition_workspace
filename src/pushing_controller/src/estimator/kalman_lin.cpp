#include <estimator/kalman_lin.hpp>
namespace filters
{

  KalmanLin::KalmanLin(const Vector9d& xin, const Matrix9d& Pin, const Matrix9d& Qin, const Matrix3d& Rin, double dt) :
      x_(xin),P_(Pin), Q_(Qin), R_(Rin)
  {
    I_.setIdentity();
    K_.setZero();
    A_.setZero();
    A_.topRightCorner(6, 6) = MatrixXd::Identity(6, 6) * dt;
    A_.topRightCorner(3, 3) = MatrixXd::Identity(3, 3) * dt * dt * 0.5;
    A_ += I_;

    C_.setZero();
    C_.block<3, 3>(0, 0).setIdentity();
  }

  void KalmanLin::predict()
  {
    x_ = A_ * x_;

    P_ = A_ * P_ * (A_.transpose()) + Q_;
    K_ = P_ * (C_.transpose()) * ((C_ * P_ * (C_.transpose()) + R_).inverse());
  }

  void KalmanLin::update(const Vector3d& y)
  {
    x_ = x_ + K_ * (y - C_ * x_);

    P_ = (I_ - K_ * C_) * P_;
  }
} // namespace filters
