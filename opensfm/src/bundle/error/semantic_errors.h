#pragma once

#include <bundle/bundle_adjuster.h>
#include <bundle/error/error_utils.h>
#include <ceres/ceres.h>
//#include <ceres/sized_cost_function.h>
#include <foundation/types.h>
#include <geometry/functions.h>

#include "foundation/optional.h"
#include "geometry/camera_instances.h"
#include <ceres/internal/logging.h>

namespace bundle {

template <typename T>
inline double GetScalar(const T& x) {
    return x;   // works when T = double
}

template <int N>
inline double GetScalar(const ceres::Jet<double, N>& x) {
    return x.a; // scalar part of Jet
}

class SemanticReprojectionError {
    public: 
    SemanticReprojectionError(const geometry::ProjectionType& type,
                              double std_dev,
                              int observed_label,
                              double confidence,
                              double lambda,
                              const SegmImage& segmentation_image) :
        type_(type),
        observed_label_(observed_label),
        scale_(std::sqrt(lambda * confidence) / std::max(std_dev, 1e-6)),
        segmentation_image_(segmentation_image),
        height_(segmentation_image.rows()),
        width_(segmentation_image.cols()) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const rig_instance,
                    const T* const rig_camera,
                    const T* const point,
                    T* residuals) const {

        // World → rig → camera
        T scale_one = T(1.0);
        T camera_point[3];
        WorldToCameraCoordinatesRig(&scale_one, rig_instance, rig_camera,
                                        point, &camera_point[0]);
        
        // Apply Camera Projection
        T predicted[2];
        geometry::Dispatch<geometry::ProjectFunction>(type_, camera_point, camera,
                                                        predicted);
        
        // Convert to pixel coordinates
        T u = predicted[0];
        T v = predicted[1];

        // Bounds check (use scalar part for Jets)
        double u0 = GetScalar(u);
        double v0 = GetScalar(v);
        if (u0 < 0 || u0 >= width_ || v0 < 0 || v0 >= height_) {
            residuals[0] = T(0);
            return true;
        }

        int iu = static_cast<int>(u0);
        int iv = static_cast<int>(v0);
        //int idx = iv * width_ + iu;

        int predicted_label = segmentation_image_(iv,iu);
        
        // The error is the difference between the predicted semantic label and the observed semantic label
        residuals[0] = T(scale_) * (T(predicted_label) - T(observed_label_));
        
        if constexpr (std::is_same_v<T, double>) {
            LOG(INFO)
                << "[SemanticResidual] "
                << "u: " << u0 << " v: " << v0
                << " | pred: " << predicted_label
                << " | obs: " << observed_label_
                << " | scale: " << scale_
                << " | residual: " << r
                << std::endl;
        }        
        return true;
    }

    protected:
        geometry::ProjectionType type_;
        double scale_;
        int observed_label_;
        const SegmImage& segmentation_image_;
        int height_;
        int width_;

};

} // namespace bundle