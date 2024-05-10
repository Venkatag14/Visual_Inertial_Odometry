MSCKF (Multi-State Constraint Kalman Filter) is an EKF based tightly-coupled visual-inertial odometry algorithm. S-MSCKF is MSCKF's stereo version. This project is a Python reimplemention of S-MSCKF, the code is directly translated from official C++ implementation KumarRobotics/msckf_vio.

For algorithm details, please refer to:

Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight, Ke Sun et al. (2017)
A Multi-State Constraint Kalman Filterfor Vision-aided Inertial Navigation, Anastasios I. Mourikis et al. (2006)
Requirements
Python 3.6+
numpy
scipy
cv2
pangolin (optional, for trajectory/poses visualization)
Dataset
EuRoC MAV: visual-inertial datasets collected on-board a MAV. The datasets contain stereo images, synchronized IMU measurements, and ground-truth.
This project implements data loader and data publisher for EuRoC MAV dataset.
Run
python vio.py --view --path path/to/your/EuRoC_MAV_dataset/MH_01_easy
or
python vio.py --path path/to/your/EuRoC_MAV_dataset/MH_01_easy (no visualization)

Results
MH_01_easy


License and References
Follow license of msckf_vio. Code is adapted from this implementation.
