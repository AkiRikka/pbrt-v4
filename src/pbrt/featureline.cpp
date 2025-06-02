#include <pbrt/featureline.h>

#include <type_traits>

#include <pbrt/base/camera.h>
#include <pbrt/base/filter.h>
#include <pbrt/film.h>
#include <pbrt/shapes.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/cameras.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/cpu/primitive.h>


namespace feature_line {

namespace {

// 因为报错，所以绕过了dispatch，但需要手动添加getter函数和分支，依赖手动 tag 判断
// 辅助函数：获取表面的Albedo
pbrt::SampledSpectrum getAlbedo(const pbrt::SurfaceInteraction &isect,
                                 const pbrt::SampledWavelengths &lambda) {
    using MatPtr = std::remove_reference_t<decltype(isect.material)>;
    const uint32_t tag = isect.material.Tag();

if (tag == MatPtr::TypeIndex<pbrt::DiffuseMaterial>()) {
        if (auto *mat = isect.material.CastOrNullptr<pbrt::DiffuseMaterial>()) {
            return mat->GetReflectance().Evaluate(isect, lambda);
        }
    } else if (tag == MatPtr::TypeIndex<pbrt::CoatedDiffuseMaterial>()) {
        if (auto *mat = isect.material.CastOrNullptr<pbrt::CoatedDiffuseMaterial>()) {
            return mat->GetReflectance().Evaluate(isect, lambda);
        }
    }
    
    // 可以添加更多类型材质...

    return pbrt::SampledSpectrum(0.f);
}

// 辅助函数：获取相机的FOV
pbrt::Float getCameraFOV(const pbrt::Camera &camera) {
    using CamPtr = std::remove_reference_t<decltype(camera)>;
    const uint32_t tag = camera.Tag();

    if (tag == CamPtr::TypeIndex<pbrt::PerspectiveCamera>()) {
        if (auto *cam = camera.CastOrNullptr<pbrt::PerspectiveCamera>())
            return cam->GetFOV();
    }
    // 可以添加更多相机类型..

    return 45.f; // 默认 FOV
}

// 辅助函数：实现论文中的特征线度量比较
bool satisfiesMetric(const pbrt::SurfaceInteraction& q_isect,
                     const pbrt::SurfaceInteraction& s_isect,
                     const pbrt::Ray& edge,
                     const pbrt::SampledWavelengths &lambda) {
                   
    // 1. MeshID: 如何获取？
    
    
    // 2. Albedo
    const pbrt::Float albedo_threshold = 0.1f;
    pbrt::SampledSpectrum queryAlbedo = getAlbedo(q_isect, lambda);
    pbrt::SampledSpectrum sampleAlbedo = getAlbedo(s_isect, lambda);

    // 转换为 RGB 空间做差值比较
    pbrt::RGB q_rgb = queryAlbedo.ToRGB(lambda, *pbrt::RGBColorSpace::sRGB);
    pbrt::RGB s_rgb = sampleAlbedo.ToRGB(lambda, *pbrt::RGBColorSpace::sRGB);
    if ((std::abs(q_rgb.r - s_rgb.r) + std::abs(q_rgb.g - s_rgb.g) + std::abs(q_rgb.b - s_rgb.b)) / 3.f > albedo_threshold) {
        return true;
    }
    

    // 3. Normal
    const pbrt::Float normal_threshold = 0.08f; //0.08
    if (1.f - pbrt::Dot(q_isect.n, s_isect.n) > normal_threshold) {
        return true;
    }
    
    
    // 4. Depth
    pbrt::Float dq = pbrt::Distance(edge.o, q_isect.p());
    pbrt::Float ds = pbrt::Distance(edge.o, s_isect.p());

    const pbrt::SurfaceInteraction& closer_isect_for_normal = (dq < ds) ? q_isect : s_isect;
    pbrt::Normal3f n_closest = closer_isect_for_normal.n;
    pbrt::Float dist_q_s = pbrt::Distance(q_isect.p(), s_isect.p());

    pbrt::Float abs_dot_edgeDir_nClosest = std::abs(pbrt::Dot(edge.d, n_closest));
    if (abs_dot_edgeDir_nClosest < 1e-6f) {
        abs_dot_edgeDir_nClosest = 1e-6f;
    }

    const pbrt::Float beta = 2.0f; //2.0
    pbrt::Float t_depth = beta * std::min(dq, ds) * dist_q_s / abs_dot_edgeDir_nClosest;

    if (std::abs(ds - dq) > t_depth) {
        return true;
    }
    

    //return true; 
    return false;
}

// 辅助函数：计算线段上距离射线最近的点
// v 是线段 (p_q, p_s) 上的点, edge 是查询射线
pbrt::Point3f ClosestPointOnSegmentToRay(const pbrt::Point3f &p_q,
                                         const pbrt::Point3f &p_s,
                                         const pbrt::Ray &edge) {
    pbrt::Vector3f segmentDir = p_s - p_q;
    pbrt::Vector3f w = p_q - edge.o;

    pbrt::Float c1 = pbrt::Dot(w, edge.d);
    pbrt::Float c2 = pbrt::Dot(segmentDir, edge.d);

    // 处理线段与射线平行的情况
    if (std::abs(1.0f - std::abs(c2)) < 1e-6f) {
        return p_q;
    }
    
    pbrt::Float t = pbrt::Dot(w, segmentDir) - c1 * c2;
    t /= pbrt::Dot(segmentDir, segmentDir) - c2 * c2;
    
    // 将计算出的最近点参数 t 限制在线段 [0, 1] 范围内
    t = pbrt::Clamp(t, 0.f, 1.f);
    
    return p_q + t * segmentDir;
}

} // namespace

// 尝试在边 edge 上找到一条特征线交点（返回最靠近 ray.o 的有效点）
pstd::optional<FeatureLineInfo> Intersect(
    const pbrt::Ray &edge,
    const pbrt::SurfaceInteraction &queryInteraction,
    pbrt::Float path_distance,
    const pbrt::Primitive &aggregate,
    pbrt::Sampler &sampler,
    const pbrt::Camera &camera,
    const pbrt::SampledWavelengths &lambda,
    pbrt::Float screenSpaceLineWidth,
    int numSamples) {

    pstd::optional<FeatureLineInfo> closestLineSoFar;
    pbrt::Float edge_length = pbrt::Distance(edge.o, queryInteraction.p());

    // 计算每像素宽度 & 特征线投影半径范围
    pbrt::Float fov = getCameraFOV(camera);
    pbrt::Float tan_half_fov = tan(pbrt::Radians(fov * 0.5f));
    pbrt::Float p_width = (2.f * tan_half_fov) / camera.GetFilm().FullResolution().y;
    pbrt::Float radius_start = path_distance * p_width * screenSpaceLineWidth;
    pbrt::Float radius_end = (path_distance + edge_length) * p_width * screenSpaceLineWidth;
    pbrt::Frame edgeFrame = pbrt::Frame::FromZ(pbrt::Normalize(edge.d));

    /*
    // Debug
    pbrt::Point3f featureCandidatePoint = pbrt::Point3f(1.0f, 1.0f, 1.0f);
    // 计算候选点沿查询光线方向的深度
    pbrt::Float current_depth = 1.0f;
    closestLineSoFar = FeatureLineInfo{
        featureCandidatePoint,             // 存储特征点位置
        current_depth,                     // 深度值
        pbrt::SampledSpectrum(0.0f)        // 默认颜色
    };
    */
    

    for (int i = 0; i < numSamples; ++i) {
        pbrt::Point2f u_disk = sampler.Get2D();
        pbrt::Point2f pDisk = pbrt::SampleUniformDiskConcentric(u_disk);
        pbrt::Vector3f v_on_start_disk_local = pbrt::Vector3f(pDisk.x, pDisk.y, 0) * radius_start;
        pbrt::Point3f sample_o_world = edge.o + edgeFrame.FromLocal(v_on_start_disk_local);
        pbrt::Vector3f v_on_end_disk_local = pbrt::Vector3f(pDisk.x, pDisk.y, 0) * radius_end;
        pbrt::Point3f sample_target_world = queryInteraction.p() + edgeFrame.FromLocal(v_on_end_disk_local);
        
        // 构造采样射线
        pbrt::Ray sampleRay(sample_o_world, pbrt::Normalize(sample_target_world - sample_o_world));
        sampleRay.o += sampleRay.d * 1e-4f; // 对样本光线起点进行偏移，防止自相交

        pbrt::SurfaceInteraction sampleInteraction;
        pstd::optional<pbrt::ShapeIntersection> shapeSi = aggregate.Intersect(sampleRay);
        
        if (shapeSi.has_value()) {
        sampleInteraction = shapeSi->intr;
        if (satisfiesMetric(queryInteraction, sampleInteraction, edge, lambda)) {
                pbrt::Point3f featureCandidatePoint = ClosestPointOnSegmentToRay(
                    queryInteraction.p(), sampleInteraction.p(), edge);

                // 计算候选点沿查询光线方向的深度
                pbrt::Float t_feature = pbrt::Dot(featureCandidatePoint - edge.o, edge.d);

                pbrt::Float current_depth = t_feature;
                        if (!closestLineSoFar.has_value() || current_depth < closestLineSoFar->depth) {
                            closestLineSoFar = FeatureLineInfo{
                                featureCandidatePoint,             // 存储特征点位置
                                current_depth,                     // 深度值
                                pbrt::SampledSpectrum(0.4f)        // 默认颜色
                            };
                        }
                }
        }
        /*
        if (shapeSi.has_value()) {
            sampleInteraction = shapeSi->intr;
            if (satisfiesMetric(queryInteraction, sampleInteraction, edge, lambda)) {
                pbrt::Point3f featureCandidatePoint = ClosestPointOnSegmentToRay(
                    queryInteraction.p(), sampleInteraction.p(), edge);

                // 计算候选点沿查询光线方向的深度
                pbrt::Float t_feature = pbrt::Dot(featureCandidatePoint - edge.o, edge.d);

                    if (t_feature > 1e-5f && t_feature < edge_length - 1e-5f) {
                        pbrt::Float current_depth = t_feature;
                        if (!closestLineSoFar.has_value() || current_depth < closestLineSoFar->depth) {
                            closestLineSoFar = FeatureLineInfo{
                                featureCandidatePoint,             // 存储特征点位置
                                current_depth,                     // 深度值
                                pbrt::SampledSpectrum(0.0f)        // 默认颜色
                            };
                        }
                    }
                }
            }
        */
    }
    return closestLineSoFar;
}

} // namespace feature_line