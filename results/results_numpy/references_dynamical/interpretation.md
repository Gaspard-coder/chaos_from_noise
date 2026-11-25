# What worked well
- Non-chaotic signals were 100% rejected (0 false positives): Every single linear, sawtooth, and sinusoid file was rejected.

- Hénon and Rössler estimates are spot-on:

	- Hénon (theory 0.4192): Adaptive selection hit 0.4229, 0.4274, and 0.4160 across r01–r03 (~0.8%–2.0% error), with NMSE_val ~ 1e-11 and Rosenstein trajectory LE matching at 0.386–0.409.

        - Rössler (theory 0.0714 at dt=0.1): Adaptive decimation picked D_k = 2, L* = 7, m* = 5, q* = 10 automatically without any manual dt, yielding 0.0691, 0.0766, and 0.0764 (when divided by dt=0.1), matching Rosenstein (0.0702–0.0732) to within ~5%.

# Why True Chaotic Systems Failed
- Hénon: --parlitz-check ran on a non-invertible 1D/2D map (Gençay & Dechert explicitly warn time-reversal fails on discrete maps), causing G5 to fail and skip G6.
- Rossler: g5_geom_ok passed (0.0153 <= 0.0140), so --parlitz-check alone caused G5_GEOMETRY to fail (on a 1D delay projection or due to `0.5 *
- Logistic: m_min = 2 over-embedded a 1D map (m=1). Unconstrained transverse slopes in $\mathbb{R}^2$ doubled $\lambda_{\text{orb}}$ (~0.82 vs true 0.347 and lambda_traj = 0.335), which G3 & G5 rightly caught.
- Lorenz: Welch PSD picked a slow lobe-switching peak (T_k = 333–714 samples instead of the ~38-sample spiral period), over-decimating by D_k = 16–35 down to 286–625 points.

# Fixes
1. Drop --parlitz-check on 1D scalar projections / discrete maps (and decouple Gate 6 from Gate 5):
	- Look at Rössler: lambda_orb = 0.0153 and lambda_traj = 0.0140 agree to within 9%, yet G5_GEOMETRY failed solely because --parlitz-check was enabled, which in turn skipped G6_IAAFT (p_nmse_iaaft = NaN)!
	- Even better: run Gate 6 (IAAFT) whenever Gates 1–4 pass (do not let Gate 5 block Gate 6), and omit --parlitz-check unless you are testing full-state invertible ODEs.
2. Allow --m-min 1 (and add tiny L2 ridge regularization in LM): Logistic map is 1-dimensional ($n = 1$). Because you forced --m-min 2, the network had 1 transverse direction in $\mathbb{R}^2$ that exploded (lambda_orb = 0.818 while lambda_traj = 0.335 nailed the true value!). Allowing --m-min 1 lets BIC pick $m^* = 1$ for 1D dynamics and eliminates transverse divergence.
3. Fix estimate_dominant_period for broadband / multi-timescale signals (Lorenz):For chaotic flows with slow regime switching (like Lorenz's butterfly wing flips), the lowest-frequency PSD peak is the slow envelope (T_k = 555), not the orbital cycle (T_k ~ 38). Use the spectral centroid or zero-crossing period (--period-method zcr)—or cap D_k so N_eff >= 1500—so Lorenz never gets over-decimated below 500 samples.
4. Note on DET_sawtooth_r01.mat:Notice that sawtooth_r01 passed Gates 1–4 (lambda_orb = 0.0770) and was only caught by Gate 5 (lambda_traj = 0.0070)! That is a huge win for your Gençay & Dechert geometric cross-check—without Gate 5, a sharp corner in a periodic sawtooth wave would have produced a false-positive Jacobian Lyapunov exponent.
