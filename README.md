# Cubature Kalman Filter (CKF) for a Nonlinear System

This repository implements a Cubature Kalman Filter (CKF) for a one-dimensional nonlinear dynamic system. The implementation was done for UCL COMP0168.

## Problem Setup

We consider a nonlinear system defined by:

### Process Model
\[
f(x, k) = 0.5x + \frac{25x}{1 + x^2} + 8\cos(1.2k)
\]
where:
- \( x \) is the state variable.
- \( k \) is the discrete time index.

### Measurement Model
\[
h(x) = \frac{x^2}{20}
\]

### Initial Parameters
- Initial state: \( x_0 = 2.0 \)
- Initial covariance: \( P_0 = 1.0 \)
- Process noise covariance: \( Q = 0.1 \)
- Measurement noise covariance: \( R = 0.2 \)

## Algorithm: Cubature Kalman Filter (CKF)
The CKF is employed for nonlinear state estimation, with prediction and measurement update steps as summarized below:

**Time Update (Prediction)**
1. Compute square-root decomposition: \( P_k = SS^T \)
2. Generate cubature points:
\[ \chi_i = x_k \pm \sqrt{n}S_{:,i} \]
3. Propagate cubature points through process model \( f(x, k) \).
4. Calculate predicted state mean and covariance.

**Measurement Update**
1. Compute square-root decomposition: \( P_{k+1}^- = SS^T \)
2. Generate cubature points around predicted state.
3. Propagate cubature points through measurement model \( h(x) \).
4. Compute predicted measurement mean and covariance.
5. Calculate cross-covariance and Kalman gain.
6. Update the state estimate and covariance with new measurement.

## Output
The implementation demonstrates one complete CKF update cycle from \( k = 0 \) to \( k = 1 \), providing updated state estimates and covariances.
