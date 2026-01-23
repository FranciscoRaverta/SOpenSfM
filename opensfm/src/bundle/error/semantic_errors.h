#pragma once

#include <bundle/bundle_adjuster.h>
#include <bundle/error/error_utils.h>
#include <ceres/ceres.h>
//#include <ceres/sized_cost_function.h>
#include <foundation/types.h>
#include <geometry/functions.h>
#include <string>

#include "foundation/optional.h"
#include "geometry/camera_instances.h"

namespace bundle {

template <typename T>
inline double GetScalar(const T& x) {
    return x;   // works when T = double
}

template <int N>
inline double GetScalar(const ceres::Jet<double, N>& x) {
    return x.a; // scalar part of Jet
}

inline double BoundaryDistance(
    const SegmImage& seg,
    int u, int v,
    int expected_label,
    int max_radius = 20
) {
    const int h = seg.rows();
    const int w = seg.cols();

    double min_dist2 = std::numeric_limits<double>::infinity();

    for (int r = 1; r <= max_radius; ++r) {
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int x = u + dx;
                int y = v + dy;

                if (x < 0 || x >= w || y < 0 || y >= h) continue;
                if (seg(y, x) != expected_label) continue;

                double d2 = dx * dx + dy * dy;
                if (d2 < min_dist2) {
                    min_dist2 = d2;
                }
            }
        }
        if (std::isfinite(min_dist2)) break; // early exit
    }

    if (!std::isfinite(min_dist2)) {
        return max_radius;  // fallback penalty
    }

    return std::sqrt(min_dist2);
}

class SemanticReprojectionError {
    public: 
    SemanticReprojectionError(const geometry::ProjectionType& type,
                              double std_dev,
                              int observed_label,
                              double confidence,
                              double lambda,
                              const SegmImage& segmentation_image,
                              std::string residual_method) :
        type_(type),
        observed_label_(observed_label),
        scale_(lambda),// / std::max(std_dev, 1e-6)),
        segmentation_image_(segmentation_image),
        height_(segmentation_image.rows()),
        width_(segmentation_image.cols()),
        confidence_(std::srqt(confidence)),
        residual_method_(residual_method) {}

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
        //residuals[0] = T(scale_) * (T(predicted_label) - T(observed_label_)); //This has no meaning, as the difference of labels says nothing
        if (residual_method_ == "negative_log_likelihood") {
            double p = (predicted_label == observed_label_) ? confidence_ : (1.0 - confidence_);
            residuals[0] = T(scale_) * T(std::sqrt(-std::log(std::max(p, 1e-6))));
        } else if (residual_method_ == "binary_residual") {
            double w = scale_ * confidence_;
            residuals[0] = (predicted_label == observed_label_) ? T(0) : T(w);
        } else if (residual_method_ == "boundary_distance_residual") {
            double dist = BoundaryDistance(segmentation_image_, iu, iv, observed_label_, 50);
            double w = scale_ * confidence_;
            residuals[0] = T(w) * T(dist);
        } else {
            return false;
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
        double confidence_;
        std::string residual_method_;

};

} // namespace bundle