# FAITH (aka "hopes and dreams integrator")

> Well yes, making your own renderer sounds like that your work will completely diverge there. This is a challenge on it's own

> I have a feeling, that we are circling back to the very beginning, and it is the LuxCore renderer. This analysis, that you just wrote, we did somewhere around 2 years ago and the result was precisely LuxCore (back then Isaac Sim did not exist). And from what you wrote it seems to me that it satisfies requirements, but it slow. I understand that in your setting this is a no-go requirement, since it is a timewise nonsense with your hardware. But the reality is, that hardware is cheaper than software and work, that's why corporates have huge servers instead of trying 10x to implement things optimally and accelerated.

> I think that making new custom renderer is not what we want to do... at least not now

> When Fedir will have completed PhD thesis and a free year ahead of him, then \[he can make a renderer\]...

> Which is worse, never making an attempt or failing while trying?

> Man's unfailing capacity to believe what he prefers to be true rather than what the evidence shows to be likely and possible has always astounded me. We long for a caring Universe which will save us from our childish mistakes, and in the face of mountains of evidence to the contrary we will pin all our hopes on the slimmest of doubts. God has not been proven not to exist, therefore he must exist.
> 
> – Academician Prokhor Zakharov, "For I Have Tasted the Fruit" (Sid Meier's Alpha Centauri)

Mitsuba 3 integrator, based on Fiction, Aspirations, Ideals, Trust, and Hope (FAITH).

Okay, but in all seriousness it is a simple SPPM integrator. And I do mean simple. One python file kind of simple.

And thanks to the wonders of drjit, it is also fast-ish. But it's putting the "L" into "LuxCore" for sure.

Comparison of algorithms (64 iters where not specified):

| renderer           | speed | caustics | specular | backend              |
| ------------------ | ----- | -------- | -------- | -------------------- |
| `path`               |  <1s  | trash    | mediocre | drjit (CUDA + OptiX) |
| `ptracer`            |   3s  | amazing  | trash    | drjit (CUDA + OptiX) |
| `rvppt`              |  10s  | amazing  | mediocre | drjit (CUDA + OptiX) |
| LuxCore PT         | 180s  | trash    | okay     | CUDA + OptiX         |
| LuxCore PT + PGI   | 200s  | mediocre | okay     | CUDA + OptiX         |
| LuxCore BDPT       | +600s | great    | great    | CPU                  |
| `faith`              |  12s  | amazing  | great    | drjit (CUDA + OptiX) |
| `faith` (16 iters)   |   3s  | amazing  | okay     | drjit (CUDA + OptiX) |
| `faith` (8 iters)    |   1s  | okay[^1]   | okay     | drjit (CUDA + OptiX) |

[^1]: caustics themselves fully converge and look perfect, except there is some noise on scattered light. It is more like white noise than truly degenerate caustics.

All GPU-accelerated algorithms tested on RTX 5060 Ti 16GB \[Discrete\]. CPU algorithms were tested on Intel(R) Core(TM) i5-14600KF (12+8) @ 5.30 GHz. For DrJIT reported times are warm runs (when CUDA code is already compiled and cached).

The scale is: trash (nonsenset, degenerate or littered with fireflies), mediocre (very grainy, not fully converged), okay (converged, but noticeably noisy), great (fully converged, small noise possible), amazing (almost perfect, no noise).

## 12 seconds too slow for your liking?

Original aim was 10 seconds. But considering that no other algorithm could make renders of desirable quality in this time, `faith` integrator is clearly the best we have, even if it's taking 12 seconds. Moreover, there are potentially many improvements that can be made, as current implementation is correctness-first. It also works really well even with very little iterations. The algorithm converges really quickly, and at >32 iters only sees marginal improvements.

16 iters usually gets great quality at mesely 5 seconds, and 8 iters can be perfectly acceptable if fed through a denoiser.

## Limitations and bugs

### Too many hyperparameters

There are quite a bit of hyperparameters (iterations, photon count, max bounces, initial radius, ...) that all can significantly affect the quality of render. It is not so simple and "plug-n-play" as the `path` integrator. **Beware, that increasing number of iterations does not necessarily increase quality.**

### DrJIT is called DrJIT for a reason

As the name suggests, DrJIT is a Just-In-Time compiler, in this case for CUDA. Hence, it can take some time for compilation, typically in the range of 10 seconds.

That is typically not a problem, because DrJIT caches compilation results, but beware that if you change hyperparameters it will force DrJIT to recompile the entire thing.

### VRAM usage

While `faith` generally doesn't need too much VRAM (which is worth more than gold these days). Still, some settings (especially photon count and max bounces). By reducing those you can lower VRAM usage at the cost of quality.

### Failure modes

The algorithm is resistant to fireflies, but instead prone to another failure: dead pixels.

Primary failure mode of `faith` is when pixels cannot get any visible points deposited, which can happen in complex scenes with shiny transparent surfaces pointing into the void. For example, if you place a glass pane at an angle, the camera rays during the first pass would most likely reflect off the surface into the void, in which case the ray cannot terminate at a diffuse surface, and is discarded. Therefore, at those pixels no light from photon tracing pass can be accumulated, hence they become black. For contrast, a path tracer would be shooting `spp` amount of rays for each pixels, which is set significantly higher than iterations for `faith`, so it's far more probable that at some point rays actually do penetrate the surface. In those spots pixels would appear completely black. This is unfortunately an inherent limitation of SPPM, and no amount of `faith` can fix this. I can only wish for next-gen analytical `facts` integrator...

## Acknowledgement

While this is a super tiny weekend project, I still feel obliged to express the gratitude to the giants on whose shoulders I stand.

Of course, I thank my supervisor and colleagues for their support, and their honesty that pushed me to write this renderer. While I began doing it as a petty act, I learned that renderers are actually super cool on the way, and at the end of the day, it's always good to find yourself writing something cool. Morale of the team had to be improved somehow :).

I thank the creators of Mitsuba 3 and DrJIT for creating such an amazing renderer, and especially DrJIT, which is the allowing technology for `faith`. I mean, my rendering thingy is barely above noise compared to their projects.

> \[Mitsuba 3\] was created by Wenzel Jakob. Significant features and/or improvements to the code were contributed by Sébastien Speierer, Nicolas Roussel, Merlin Nimier-David, Delio Vicini, Tizian Zeltner, Baptiste Nicolet, Miguel Crespo, Vincent Leroy, and Ziyi Zhang.

I also greatly thank Matt Pharr, Wenzel Jakob, and Greg Humphreys for writing, and freely distributing the book *Physically Based Rendering: From Theory To Implementation*, without which I could never possibly fathom how to write an integrator (and admittedly still do not).
