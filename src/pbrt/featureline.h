#ifndef PBRT_FEATURELINE_H
#define PBRT_FEATURELINE_H


#include <pbrt/pbrt.h>
#include <pbrt/options.h>
#include <pbrt/util/spectrum.h> 

// 前向声明
namespace pbrt {
class Primitive;
class Camera;
class Sampler;
class SurfaceInteraction;
}

namespace feature_line {

// FeatureLineInfo 结构体定义
struct FeatureLineInfo {
    pbrt::Point3f position;
    pbrt::Float depth;
    pbrt::SampledSpectrum color;
};

pstd::optional<FeatureLineInfo> Intersect(
    const pbrt::Ray &edge, // 要测试的光线edge
    const pbrt::SurfaceInteraction &queryInteraction, // 此段edge击中的表面交点
    pbrt::Float path_distance, // edge起点到相机的总距离
    const pbrt::Primitive &aggregate, // 场景对象
    pbrt::Sampler &sampler, // 采样器
    const pbrt::Camera &camera, // 相机对象
    const pbrt::Spectrum &base_feature_spectrum,
    const pbrt::SampledWavelengths &lambda, // 采样波长
    pbrt::Float screenSpaceLineWidth, // 期望的屏幕空间线条宽度
    int numSamples); // 采样数量

}

#endif // PBRT_FEATURELINE_H