#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // initially set to false, will be set to true after first measurement is processed
  is_initialized_ = false;

  // State dimension
  n_x_ = x_.size();
  // Augmented state dimension
  n_aug_ = n_x_ + 2;
  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;
  // Set the predicted sigma points matrix dimentions
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;
  // Weights of sigma points
  weights_ = VectorXd(n_sig_);
}

UKF::~UKF() {}


/**
 *  Angle normalization to [-Pi, Pi]
 */
void UKF::NormaliseAngle(double *ang) {
    while (*ang > M_PI) *ang -= 2. * M_PI;
    while (*ang < -M_PI) *ang += 2. * M_PI;
}
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
  // Initial covariance matrix
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];
      // Coordinates convertion from polar to cartesian
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);
      float v  = sqrt(vx * vx + vy * vy);
      x_ << px, py, v, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
      // Special case during initialisation
      if (fabs(x_(0)) < 0.0001 and fabs(x_(1)) < 0.0001){
		x_(0) = 0.0001;
		x_(1) = 0.0001;
	  }
    }

    // Initialize weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < weights_.size(); i++) {
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    // store measurement timestamp to use it for calculating delta_t
    time_us_ = meas_package.timestamp_;
    // Done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

   //compute the time elapsed between the current and previous measurements
   double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
   //previous_timestamp_ = measurement_pack.timestamp_;
   time_us_ = meas_package.timestamp_;

   Prediction(dt);
   if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
   UpdateRadar(meas_package);
   }
   if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
   UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Augmented mean vector
  VectorXd x_aug_ = VectorXd(n_aug_);
  // Augmented state covarience matrix
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
  // Sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, n_sig_);

  //create augmented covariance matrix
  MatrixXd Q_ = MatrixXd(2,2);
  Q_ << std_a_*std_a_, 0.0,
        0.0, std_yawdd_*std_yawdd_;

  // Fill the matrices
  //create augmented mean state
  x_aug_.setZero();
  x_aug_.head(n_x_) = x_;

  P_aug_.setZero();
  P_aug_.topLeftCorner(n_x_,n_x_) = P_;
  P_aug_.bottomRightCorner(2, 2)= Q_;

  //create square root matrix of P_aug_
  MatrixXd A_ = P_aug_.llt().matrixL();

  //create augmented sigma points*/
  Xsig_aug_.col(0)= x_aug_;
  double par = sqrt(lambda_+n_aug_);
  for (int i=1;i<n_aug_+1;i++){
      Xsig_aug_.col(i)= x_aug_+par*A_.col(i-1);
      Xsig_aug_.col(i+n_aug_)= x_aug_-par*A_.col(i-1);
  }

  // Predict sigma points
  for (int i = 0; i< n_sig_; i++)
  {
      double p_x = Xsig_aug_(0,i);
      double p_y = Xsig_aug_(1,i);
      double v = Xsig_aug_(2,i);
      double yaw = Xsig_aug_(3,i);
      double yawd = Xsig_aug_(4,i);
      double vu_a = Xsig_aug_(5,i);
      double vu_yawdd = Xsig_aug_(6,i);

      double px_p, py_p;

      if (fabs(yawd)>0.0001){
          px_p = p_x+(v/yawd)*(sin(yaw+delta_t*yawd)-sin(yaw));
          py_p = p_y+(v/yawd)*(-cos(yaw+delta_t*yawd)+cos(yaw));
      }
      else{
          px_p = p_x+v*delta_t*cos(yaw);
          py_p = p_y+v*delta_t*sin(yaw);
      }


      double v_p = v;
      double yaw_p = yaw + yawd*delta_t;
      double yawd_p = yawd;

    // Add noise
      px_p = px_p + 0.5*cos(yaw)*delta_t*delta_t*vu_a;
      py_p = py_p + 0.5*sin(yaw)*delta_t*delta_t*vu_a;

      v_p = v_p + vu_a*delta_t;
      yaw_p = yaw_p + 0.5*vu_yawdd*delta_t*delta_t;
      yawd_p = yawd_p + vu_yawdd*delta_t;

      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yawd_p;
  }

  // Predicted state mean
  x_ = Xsig_pred_ * weights_;
  // Predicted state covariance matrix
  P_.setZero();
  for (int i = 0; i < n_sig_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalisation
    NormaliseAngle(&(x_diff(3)));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // Set measurement dimension
  int n_z_ = 2;

  MatrixXd H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  MatrixXd R_ = MatrixXd(2, 2);
  //R_lidar_ << std_laspx_*std_laspx_,0,
  //            0,std_laspy_*std_laspy_;
  R_ << std_laspx_*std_laspx_,0,
             0,std_laspy_*std_laspy_;

  //create example vector for incoming lidar measurement
  VectorXd z = VectorXd(n_z_);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);
  VectorXd z_pred_ = H_ * x_;
  VectorXd y = z - z_pred_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // Set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd(n_z_, n_sig_);

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);
    Zsig_(1,i) = atan2(p_y,p_x);
    Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / Zsig_(0,i);
  }

  // Mean predicted measurement
  VectorXd z_pred_ = VectorXd(n_z_);
  //calculate mean predicted measurement
  z_pred_  = Zsig_ * weights_;


  //calculate measurement covariance matrix S
  MatrixXd R_ = MatrixXd(n_z_,n_z_);//R_radar_;

  R_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;


  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);
  S_.setZero();
  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    // Angle normalization
    NormaliseAngle(&(z_diff(1)));
    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }
  S_ = S_ + R_;

  // Create matrix for cross correlation Tc
  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);

  // Calculate cross correlation matrix
  Tc_.setZero();
  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    // Angle normalisation
    NormaliseAngle(&(z_diff(1)));

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalisation
    NormaliseAngle(&(x_diff(3)));
    Tc_ = Tc_ + weights_(i) * x_diff * z_diff.transpose();
  }
  // Measurements
  VectorXd z_ = VectorXd(n_z_);
  z_ = meas_package.raw_measurements_;
  //Kalman gain K;
  MatrixXd K_ = Tc_ * S_.inverse();
  VectorXd z_diff = z_ - z_pred_;
  // Angle normalisation
  NormaliseAngle(&(z_diff(1)));

  // Update state mean and covariance matrix
  x_ = x_ + K_ * z_diff;
  P_ = P_ - K_ * S_ * K_.transpose();
}
