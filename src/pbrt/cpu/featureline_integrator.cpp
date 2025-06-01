#include "featureline_integrator.h"

namespace pbrt {

// Constructor
FeatureLineIntegrator::FeatureLineIntegrator(
    int maxDepth, Camera camera, Sampler sampler, Primitive aggregate,
    std::vector<Light> lights, const std::string &lightSampleStrategy,
    bool regularize, int testSamples, Float lineThreshold)
    : RayIntegrator(camera, sampler, aggregate, lights),
      maxDepth(maxDepth),
      lightSampler(LightSampler::Create(lightSampleStrategy, lights, Allocator())),
      regularize(regularize),
      testSamples(testSamples),
      lineThreshold(lineThreshold) {}

void FeatureLineIntegrator::Render() {
    LOG_VERBOSE("Start Feature Line Rendering");
    LOG_VERBOSE("Pass 1: Path Collection");
    pathCache.Clear();

    // Algorithm 1, first loop: collect paths
    Bounds2i pixelBounds = camera.GetFilm().PixelBounds();
    int spp = samplerPrototype.SamplesPerPixel();

    ThreadLocal<ScratchBuffer> scratchBuffers([]() { return ScratchBuffer(); });    
    ThreadLocal<Sampler> samplers([this]() { return samplerPrototype.Clone(); });

    // Collect paths
    ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
        ScratchBuffer &scratchBuffer = scratchBuffers.Get();
        Sampler &sampler = samplers.Get();
        
        for (Point2i pPixel : tileBounds) {
            for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
                sampler.StartPixelSample(pPixel, sampleIndex);
                
                // Generate camera ray
                Float lu = sampler.Get1D();
                SampledWavelengths lambda = camera.GetFilm().SampleWavelengths(lu);
                Filter filter = camera.GetFilm().GetFilter();
                CameraSample cameraSample = GetCameraSample(sampler, pPixel, filter);
                
                pstd::optional<CameraRayDifferential> cameraRay =
                    camera.GenerateRayDifferential(cameraSample, lambda);
                
                if (cameraRay) {
                    // Trace path and store in cache
                    SampledSpectrum L = StandardLi(cameraRay->ray, lambda, sampler, 
                                                 scratchBuffer, nullptr);
                    
                    PathSample pathSample(cameraRay->ray, L, lambda);
                    
                    std::lock_guard<std::mutex> lock(cacheMutex);
                    pathCache.AddPath(pPixel, pathSample);
                }
                
                scratchBuffer.Reset();
            }
        }
    });

    LOG_VERBOSE("Pass 1 complete. Cached paths for %zu pixels", pathCache.Size());


    LOG_VERBOSE("Pass 2: Feature Line Rendering");

    // Algorithm2, second loop: modify path and render
    ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
        ScratchBuffer &scratchBuffer = scratchBuffers.Get();
        Sampler &sampler = samplers.Get();

        for (Point2i pPixel : tileBounds) {
            const std::vector<PathSample>* paths = pathCache.GetPaths(pPixel);
            if (!paths) continue;

            for (const PathSample& pathSample : * paths) {
                // Algorithm2: Modify path
                RayDifferential modifiedRay = ModifyPath(pathSample.ray, pPixel, pathSample.lambda);
                
                // Evaluate modified path
                SampledSpectrum L = StandardLi(modifiedRay, const_cast<SampledWavelengths&>(pathSample.lambda),
                                               sampler, scratchBuffer, nullptr);
                
                // Add to film
                camera.GetFilm().AddSample(pPixel, L, pathSample.lambda, nullptr, 1.0f);
                scratchBuffer.Reset();
            }
        }
    });
    LOG_VERBOSE("Feature Line Rendering complete");

    // 写入
    LOG_VERBOSE("Writing final image");
    ImageMetadata metadata;
    metadata.samplesPerPixel = spp;  // 或者你实际使用的样本数
    camera.InitMetadata(&metadata);
    camera.GetFilm().WriteImage(metadata, 1.0f);  // 写入最终图片
}

SampledSpectrum FeatureLineIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                        Sampler sampler, ScratchBuffer &scratchBuffer,
                                        VisibleSurface *visibleSurface) const {
    return StandardLi(ray, lambda, sampler, scratchBuffer, visibleSurface);
}

// Feature line 相交测试，一个简单的用于测试的，后面应该改掉
FeatureLine FeatureLineIntegrator::IntersectLine(const RayDifferential& ray, Point2i pixel,
                                               const SampledWavelengths& lambda) const {
    // Compute test region around the ray
    Bounds2f region = ComputeTestRegion(ray, pixel);
    
    // Sample nearby pixels in the region
    std::vector<Point2i> samplePixels = SampleRegion(region, testSamples);
    
    FeatureLine closest;
    
    for (Point2i samplePixel : samplePixels) {
        const std::vector<PathSample>* paths = pathCache.GetPaths(samplePixel);
        if (!paths) continue;
        
        for (const PathSample& pathSample : *paths) {
            // Check if this path satisfies the line metric
            pstd::optional<ShapeIntersection> si = Intersect(pathSample.ray);
            if (!si) continue;
            
            pstd::optional<ShapeIntersection> rayIntersection = Intersect(ray);
            if (!rayIntersection) continue;
            
            if (SatisfiesLineMetric(si->intr, rayIntersection->intr)) {
                // Compute line position and depth
                Float t = Distance(ray.o, si->intr.p());
                if (t < closest.depth) {
                    closest = FeatureLine(si->intr.p(), t, pathSample.L);
                }
            }
        }
    }
    
    return closest;
}

// Algorithm 2: path modification
RayDifferential FeatureLineIntegrator::ModifyPath(const RayDifferential& originalRay, 
                                                  Point2i pixel,
                                                  const SampledWavelengths& lambda) const {
    RayDifferential modifiedRay = originalRay;

    FeatureLine line = IntersectLine(originalRay, pixel, lambda);

    if (line.IsValid()) {
        Vector3f direction = Normalize(line.position - originalRay.o);
        Float distance = Distance(originalRay.o, line.position);
        
        modifiedRay.d = direction;

        if (modifiedRay.hasDifferentials) {
            modifiedRay.rxDirection = direction;
            modifiedRay.ryDirection = direction;
            modifiedRay.rxOrigin = originalRay.o;
            modifiedRay.ryOrigin = originalRay.o;
        }
    }

    return modifiedRay;
}

// 应该要改
// Check if two intersections satisfy the line metric
bool FeatureLineIntegrator::SatisfiesLineMetric(const SurfaceInteraction& intersection1,
                                              const SurfaceInteraction& intersection2) const {
    // Simple normal discontinuity test
    Float normalDot = Dot(intersection1.n, intersection2.n);
    return normalDot < (1.0f - lineThreshold);
}

// Compute test region for feature line detection
Bounds2f FeatureLineIntegrator::ComputeTestRegion(const RayDifferential& ray, 
                                                Point2i pixel) const {
    // Simple rectangular region around the pixel
    Float radius = 3.0f; // pixels
    return Bounds2f(Point2f(pixel.x - radius, pixel.y - radius),
                   Point2f(pixel.x + radius, pixel.y + radius));
}

// Sample pixels in the test region
std::vector<Point2i> FeatureLineIntegrator::SampleRegion(const Bounds2f& region, 
                                                       int numSamples) const {
    std::vector<Point2i> samples;
    samples.reserve(numSamples);
    
    Bounds2i pixelBounds = camera.GetFilm().PixelBounds();
    
    for (int i = 0; i < numSamples; ++i) {
        Float u = Float(i) / Float(numSamples - 1);
        Float v = Float(i % 4) / 4.0f; // Simple 2D sampling
        
        Point2f p = Lerp(u, region.pMin, region.pMax);
        Point2i pixel(int(p.x), int(p.y));
        
        // Ensure pixel is within bounds
        if (Inside(pixel, pixelBounds)) {
            samples.push_back(pixel);
        }
    }
    
    return samples;
}

// Standard path tracing implementation (similar to PathIntegrator::Li)
SampledSpectrum FeatureLineIntegrator::StandardLi(RayDifferential ray, 
                                                SampledWavelengths &lambda,
                                                Sampler sampler, 
                                                ScratchBuffer &scratchBuffer,
                                                VisibleSurface *visibleSurf) const {
    SampledSpectrum L(0.f), beta(1.f);
    int depth = 0;
    Float p_b, etaScale = 1;
    bool specularBounce = false, anyNonSpecularBounces = false;
    LightSampleContext prevIntrCtx;

    while (true) {
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        
        if (!si) {
            // Add infinite light contribution
            for (const auto &light : infiniteLights) {
                SampledSpectrum Le = light.Le(ray, lambda);
                if (depth == 0 || specularBounce)
                    L += beta * Le;
                else {
                    Float p_l = lightSampler.PMF(prevIntrCtx, light) *
                               light.PDF_Li(prevIntrCtx, ray.d, true);
                    Float w_b = PowerHeuristic(1, p_b, 1, p_l);
                    L += beta * w_b * Le;
                }
            }
            break;
        }

        // Add emission from surface
        SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
        if (Le) {
            if (depth == 0 || specularBounce)
                L += beta * Le;
            else {
                Light areaLight(si->intr.areaLight);
                Float p_l = lightSampler.PMF(prevIntrCtx, areaLight) *
                           areaLight.PDF_Li(prevIntrCtx, ray.d, true);
                Float w_l = PowerHeuristic(1, p_b, 1, p_l);
                L += beta * w_l * Le;
            }
        }

        SurfaceInteraction &isect = si->intr;
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            specularBounce = true;
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        if (depth == 0 && visibleSurf) {
            // Set up visible surface (similar to PathIntegrator)
            constexpr int nRhoSamples = 16;
            const Float ucRho[nRhoSamples] = {
                0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                0.9674518,  0.2995429,  0.5083201, 0.047338516};
            const Point2f uRho[nRhoSamples] = {
                Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

            SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);
            *visibleSurf = VisibleSurface(isect, albedo, lambda);
        }

        if (regularize && anyNonSpecularBounces) {
            bsdf.Regularize();
        }

        if (depth++ == maxDepth)
            break;

        // Sample direct illumination
        if (IsNonSpecular(bsdf.Flags())) {
            SampledSpectrum Ld = SampleLd(isect, &bsdf, lambda, sampler);
            L += beta * Ld;
        }

        // Sample BSDF for next direction
        Vector3f wo = -ray.d;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
        if (!bs)
            break;

        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        p_b = bs->pdfIsProportional ? bsdf.PDF(wo, bs->wi) : bs->pdf;
        specularBounce = bs->IsSpecular();
        anyNonSpecularBounces |= !bs->IsSpecular();
        if (bs->IsTransmission())
            etaScale *= Sqr(bs->eta);
        prevIntrCtx = si->intr;

        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);

        // Russian roulette
        SampledSpectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < 1 && depth > 1) {
            Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q)
                break;
            beta /= 1 - q;
        }
    }

    return L;
}

// Direct lighting sampling (similar to PathIntegrator::SampleLd)
SampledSpectrum FeatureLineIntegrator::SampleLd(const SurfaceInteraction &intr, 
                                               const BSDF *bsdf,
                                               SampledWavelengths &lambda, 
                                               Sampler sampler) const {
    LightSampleContext ctx(intr);
    BxDFFlags flags = bsdf->Flags();
    if (IsReflective(flags) && !IsTransmissive(flags))
        ctx.pi = intr.OffsetRayOrigin(intr.wo);
    else if (IsTransmissive(flags) && !IsReflective(flags))
        ctx.pi = intr.OffsetRayOrigin(-intr.wo);

    Float u = sampler.Get1D();
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(ctx, u);
    Point2f uLight = sampler.Get2D();
    if (!sampledLight)
        return {};

    Light light = sampledLight->light;
    pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
    if (!ls || !ls->L || ls->pdf == 0)
        return {};

    Vector3f wo = intr.wo, wi = ls->wi;
    SampledSpectrum f = bsdf->f(wo, wi) * AbsDot(wi, intr.shading.n);
    if (!f || !Unoccluded(intr, ls->pLight))
        return {};

    Float p_l = sampledLight->p * ls->pdf;
    if (IsDeltaLight(light.Type()))
        return ls->L * f / p_l;
    else {
        Float p_b = bsdf->PDF(wo, wi);
        Float w_l = PowerHeuristic(1, p_l, 1, p_b);
        return w_l * ls->L * f / p_l;
    }
}

// Factory method
std::unique_ptr<FeatureLineIntegrator> FeatureLineIntegrator::Create(
    const ParameterDictionary &parameters, Camera camera, Sampler sampler,
    Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    std::string lightStrategy = parameters.GetOneString("lightsampler", "bvh");
    bool regularize = parameters.GetOneBool("regularize", false);
    int testSamples = parameters.GetOneInt("testsamples", 16);
    Float lineThreshold = parameters.GetOneFloat("linethreshold", 0.1f);
    
    return std::make_unique<FeatureLineIntegrator>(
        maxDepth, camera, sampler, aggregate, lights, lightStrategy, 
        regularize, testSamples, lineThreshold);
}

std::string FeatureLineIntegrator::ToString() const {
    return StringPrintf("[ FeatureLineIntegrator maxDepth: %d testSamples: %d "
                       "lineThreshold: %f lightSampler: %s regularize: %s ]",
                       maxDepth, testSamples, lineThreshold, 
                       lightSampler, regularize);
}


}