# Why it is not working
The cause is double-undersampling:
1. Pre-Gate downsampled each IMF by $D_k = \lfloor T_k / 20 \rfloor$, leaving ~20 samples per cycle.
2. Step 2 then ran AMI on the already decimated signal and picked $L_{\text{est}} \approx 5$ decimated samples (a quarter-cycle, $T_k / 4$).
3. Step 3 then trained the neural network to predict $L^ \approx 5$ decimated steps ahead* ($x_{t + 5 D_k}$, a full quarter-cycle jump) on a signal whose envelope and phase fluctuate stochastically.

Over a full quarter-cycle horizon, random amplitude/frequency fluctuations make point prediction fail (NMSE_val > 0.10). When a neural network cannot predict $x_{t+L}$ accurately, least-squares regression shrinks the mapping slope toward the mean ($\vert{}\partial F / \partial X\vert{} < 1$), which forces $\lambda = \ln \vert{}\partial F / \partial X\vert{} < 0$

# How to fix it
Do not stack adaptive decimation $D_k$ on top of a large lag $L_{\text{est}}$. Either:Option A (recommended): Keep adaptive decimation $D_k = \lfloor T_k / 20 \rfloor$ (so consecutive decimated samples are already spaced by $T_k / 20$), and fix --lags 1 (or --lags 1 2) so the network predicts one decimated step ahead, orOption B: Keep D_k = 1 (--target-spc 100000) and let L_est (AMI) set the single spacing between delay coordinates.

#  Why the Complexity–Entropy Plane Calls IMF_1–IMF_4 "Chaotic" While Surrogates Reject Them
Even after fixing the double-undersampling bug so G1_NMSE passes on IMF_3–IMF_6, do not be surprised if IMF_1–IMF_4 still fail Gate 6 (IAAFT surrogates):
- Overlap in the $H \times C$ plane: In the Bandt–Pompe Complexity–Entropy causality plane (Rosso et al., 2007), correlated stochastic processes—specifically $1/f^\alpha$ colored noise, fractional Brownian/Gaussian motion (frac_alea), and narrowband EMD IMFs of atmospheric noise—occupy the exact same central $(H, C)$ region as deterministic chaotic systems. The $H \times C$ plane alone cannot distinguish narrowband correlated noise from low-dimensional chaos.
- What an atmospheric IMF actually is: EMD acts as a dyadic filter bank. Each high-frequency mode (IMF_1–IMF_4) is a narrowband stochastic oscillation $x_k(t) = a_k(t)\cos\phi_k(t)$ driven by continuous random atmospheric forcing:
	- Over short windows (3–5 samples, as measured by permutation entropy), random envelope and phase modulation creates ordinal diversity and apparent local divergence.
	- Globally, there is no deterministic map $X_{t+1} = F(X_t)$ governing future innovations—so an IAAFT surrogate with the exact same power spectrum and amplitude distribution has the same NMSE and $\lambda$ as the real IMF (p_IAAFT > 0.05).
