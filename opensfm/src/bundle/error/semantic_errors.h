#pragma once

#include <bundle/bundle_adjuster.h>
#include <bundle/error/error_utils.h>
#include <ceres/ceres.h>
//#include <ceres/sized_cost_function.h>
#include <foundation/types.h>
#include <geometry/functions.h>

#include "foundation/optional.h"
#include "geometry/camera_instances.h"

namespace bundle {

class SemanticReprojectionError {
    public: 
    SemanticReprojectionError(const geometry::ProjectionType& type,
                              double std_dev,
                              double observed_label,
                              double confidence,
                              double lambda,
                              std::string segmentation_image_path,
                              std::string confidence_image_path) :
        type_(type),
        observed_label_(observed_label),
        scale_(std::sqrt(lambda * confidence) / std_dev),
        segmentation_image_path_(segmentation_image_path),
        confidence_image_path_(confidence_image_path) {}
        //semantic_map_(semantic_map),
        //width_(width),
        //height_(height) {}

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
        double u0 = ceres::JetOps<T>::GetScalar(u);
        double v0 = ceres::JetOps<T>::GetScalar(v);
        if (u0 < 0 || u0 >= width_ || v0 < 0 || v0 >= height_) {
            residuals[0] = T(0);
            return true;
        }

        int iu = static_cast<int>(u0);
        int iv = static_cast<int>(v0);
        int idx = iv * width_ + iu;

        double predicted_label = semantic_map_[idx];

        // The error is the difference between the predicted semantic label and the observed semantic label
        residuals[0] = T(scale_) * (T(predicted_label) - T(observed_label_));

        return true;
    }

    protected:
        geometry::ProjectionType type_;
        double scale_;
        double observed_label_;
        std::string segmentation_image_path_;
        std::string confidence_image_path_;
        //const std::vector<double>& semantic_map_;
        //int width_;
        //int height_;

};

} // namespace bundle