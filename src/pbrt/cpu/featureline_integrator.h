#ifndef PBRT_FEATURELINE_INTEGRATOR_H
#define PBRT_FEATURELINE_INTEGRATOR_H

#include <pbrt/cpu/integrators.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/containers.h>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace pbrt {

// Feature Line 数据结构
struct FeatureLine {
    Point3f position;
    Float depth;
    SampledSpectrum color;
    // Vector3f normal;

    FeatureLine() : depth(Infinity) {}
    FeatureLine(Point3f pos, Float d, SampledSpectrum c)
        : position(pos), depth(d), color(c){}
    
    bool IsValid() const { return depth < Infinity; } 
};

struct PathSample
{
    RayDifferential ray;    // 采样光线
    SampledSpectrum L;      // 本次采样得到的辐射亮度
    SampledWavelengths lambda;  // 采样使用的波长
    std::vector<SurfaceInteraction> intersections;  // 表面交点序列

    PathSample(const RayDifferential& r, const SampledSpectrum& radiance,
               const SampledWavelengths& wavelengths)
        : ray(r), L(radiance), lambda(wavelengths) {}
};

class PathCache {
private:
    std::unordered_map<uint64_t, std::vector<PathSample>> cache;    // 使用(x, y)索引来存储像素对应的 Path

    uint64_t PixelToKey(Point2i pixel) const {
        return (uint64_t(pixel.x) << 32) | uint64_t(pixel.y);
    }

public:
    void AddPath(Point2i pixel, const PathSample& path) {
        uint64_t key = PixelToKey(pixel);
        cache[key].push_back(path);
    }

    const std::vector<PathSample>* GetPaths(Point2i pixel) const {
        uint64_t key = PixelToKey(pixel);
        auto it = cache.find(key);
        return (it != cache.end()) ? &it->second : nullptr;
    }

    void Clear() {
        cache.clear();
    }

    size_t Size() const {
        return cache.size();
    }
};

class FeatureLineIntegrator : public RayIntegrator {
public:
    // Constructor
    FeatureLineIntegrator(int maxDepth, Camera camera, Sampler sampler,
                          Primitive aggregate, std::vector<Light> lights,
                          const std::string &lightSampleStrategy = "bvh",
                          bool regularize = false, int testSamples = 16,
                          Float lineThrehold = 0.1f);
    
    void Render() override;

    // 似乎用不到
    // void EvaluatePixelSample(Point2i pPixel, int sampleIndex, Sampler sampler,
    //                         ScratchBuffer &scratchBuffer) override;

    SampledSpectrum Li(RayDifferential ray, SampledWavelengths &lambda, 
                      Sampler sampler, ScratchBuffer &scratchBuffer,
                      VisibleSurface *visibleSurface) const override;
    
    // Feature line detection methods
    FeatureLine IntersectLine(const RayDifferential& ray, Point2i pixel,
                             const SampledWavelengths& lambda) const;

    RayDifferential ModifyPath(const RayDifferential& originalRay, Point2i pixel,
                              const SampledWavelengths& lambda) const;

    // Factory method
    static std::unique_ptr<FeatureLineIntegrator> Create(
        const ParameterDictionary &parameters, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc);

    std::string ToString() const override;


private:
    // 这三个函数都和特征线检测有关，可以改掉
    bool SatisfiesLineMetric(const SurfaceInteraction& intersection1,
                           const SurfaceInteraction& intersection2) const;
    
    Bounds2f ComputeTestRegion(const RayDifferential& ray, Point2i pixel) const;
    
    std::vector<Point2i> SampleRegion(const Bounds2f& region, int numSamples) const;


    // 采样直接光照
    SampledSpectrum SampleLd(const SurfaceInteraction &intr, const BSDF *bsdf,
                           SampledWavelengths &lambda, Sampler sampler) const;

    // Standard path tracing (similar to PathIntegrator)
    SampledSpectrum StandardLi(RayDifferential ray, SampledWavelengths &lambda,
                              Sampler sampler, ScratchBuffer &scratchBuffer,
                              VisibleSurface *visibleSurface) const;

    int maxDepth;
    LightSampler lightSampler;  // 光源采样
    bool regularize;        // 启用平滑
    int testSamples;        // m number of samples for the intersection test
    Float lineThreshold;    // 度量阈值

    mutable PathCache pathCache;
    mutable std::mutex cacheMutex;
};

}

#endif
