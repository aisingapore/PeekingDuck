# Modifications copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright (c) 2019 ZhongdaoWang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Kalman Filter for updating Track motion states.

Modifications include:
- Removed unused distance metric in gating_distance()
- Removed predict() as only multi_predict() is used by JDE
- Removed only_position argument in gating_distance() as only False value is
    used
"""

from typing import Tuple

import numpy as np
import scipy.linalg

# Table for the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
# function and used as Mahalanobis gating threshold.
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    num_dims = 4
    dt = 1.0

    def __init__(self) -> None:
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * self.num_dims, 2 * self.num_dims)
        for i in range(self.num_dims):
            self._motion_mat[i, self.num_dims + i] = self.dt
        self._update_mat = np.eye(self.num_dims, 2 * self.num_dims)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        """Computes gating distance between state distribution and
        measurements using Mahalanobis distance.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean (np.ndarray): Mean vector over the state distribution (8
                dimensional).
            covariance (np.ndarray): Covariance of the state distribution (8x8
                dimensional).
            measurements (np.ndarray): An Nx4 dimensional matrix of N
                measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the
                height.

        Returns:
            (np.ndarray): An array of length N, where the i-th element contains
            the squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)

        distances = measurements - mean
        cholesky_factor = np.linalg.cholesky(covariance)
        maha_distance = scipy.linalg.solve_triangular(
            cholesky_factor,
            distances.T,
            lower=True,
            overwrite_b=True,
            check_finite=False,
        )
        squared_maha = np.sum(maha_distance * maha_distance, axis=0)
        return squared_maha

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates track from unassociated measurement.

        Args:
            measurement (np.ndarray): Bounding box coordinates (x, y, a, h)
                with center position (x, y), aspect ratio a, and height h.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): The mean vector (8 dimensional)
            and covariance matrix (8x8 dimensional) of the new track.
            Unobserved velocities are initialised to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs Kalman filter prediction step (Vectorized version).

        Args:
            mean (np.ndarray): The Nx8 dimensional mean matrix of the object
                states at the previous time step.
            covariance (np.ndarray): The Nx8x8 dimensional covariance matrices
                of the object states at the previous time step.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): The mean vector and covariance
            matrix of the predicted state. Unobserved velocities are
            initialised to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Projects state distribution to measurement space.

        Args:
            mean (np.ndarray): The state's mean vector (8 dimensional array).
            covariance (np.ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (Tuple[np.ndarray, np.ndarray]): The projected mean and covariance
            matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs Kalman filter correction step.

        Args:
            mean (np.ndarray): The predicted state's mean vector (8
                dimensional).
            covariance (np.ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement (np.ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the aspect
                ratio, and h the height of the bounding box.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): The measurement-corrected state
            distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance
