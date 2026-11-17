(lp_neural_network_py310) gvibert-24@gpu1:~/lp_neural_network_py310$ CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python ~/lp_neural_network_py310/chaostest_GPU.py
Starting 2-stage GPU optimization across 36 combinations (32 parallel starts) on cuda:0...
[  1/ 36] (L=1, m=1, q=1) | SSE = 9.265385 | Elapsed: 1.8s
  └─> New Best SSE: 9.265385 at triplet (L=1, m=1, q=1)
[  2/ 36] (L=1, m=1, q=2) | SSE = 1.157533 | Elapsed: 3.6s
  └─> New Best SSE: 1.157533 at triplet (L=1, m=1, q=2)
[  3/ 36] (L=1, m=1, q=3) | SSE = 1.146045 | Elapsed: 5.2s
  └─> New Best SSE: 1.146045 at triplet (L=1, m=1, q=3)
[  4/ 36] (L=1, m=1, q=4) | SSE = 1.110742 | Elapsed: 6.9s
  └─> New Best SSE: 1.110742 at triplet (L=1, m=1, q=4)
[  5/ 36] (L=1, m=1, q=5) | SSE = 1.045434 | Elapsed: 8.6s
  └─> New Best SSE: 1.045434 at triplet (L=1, m=1, q=5)
[  6/ 36] (L=1, m=1, q=6) | SSE = 1.095960 | Elapsed: 10.3s
[  7/ 36] (L=1, m=2, q=1) | SSE = 9.095709 | Elapsed: 11.9s
[  8/ 36] (L=1, m=2, q=2) | SSE = 1.133552 | Elapsed: 13.9s
[  9/ 36] (L=1, m=2, q=3) | SSE = 1.128774 | Elapsed: 16.0s
[ 10/ 36] (L=1, m=2, q=4) | SSE = 1.469465 | Elapsed: 18.2s
[ 11/ 36] (L=1, m=2, q=5) | SSE = 1.799439 | Elapsed: 20.4s
[ 12/ 36] (L=1, m=2, q=6) | SSE = 1.336152 | Elapsed: 22.6s
[ 13/ 36] (L=1, m=3, q=1) | SSE = 10.980810 | Elapsed: 24.2s
[ 14/ 36] (L=1, m=3, q=2) | SSE = 2.004274 | Elapsed: 26.2s
[ 15/ 36] (L=1, m=3, q=3) | SSE = 2.555413 | Elapsed: 28.2s
[ 16/ 36] (L=1, m=3, q=4) | SSE = 4.561511 | Elapsed: 30.2s
[ 17/ 36] (L=1, m=3, q=5) | SSE = 1.613966 | Elapsed: 32.3s
[ 18/ 36] (L=1, m=3, q=6) | SSE = 1.529965 | Elapsed: 34.4s
[ 19/ 36] (L=2, m=1, q=1) | SSE = 50.876434 | Elapsed: 35.8s
[ 20/ 36] (L=2, m=1, q=2) | SSE = 13.038057 | Elapsed: 37.4s
[ 21/ 36] (L=2, m=1, q=3) | SSE = 50.875813 | Elapsed: 38.9s
[ 22/ 36] (L=2, m=1, q=4) | SSE = 24.621677 | Elapsed: 40.5s
[ 23/ 36] (L=2, m=1, q=5) | SSE = 6.827574 | Elapsed: 42.1s
[ 24/ 36] (L=2, m=1, q=6) | SSE = 6.743707 | Elapsed: 43.7s
[ 25/ 36] (L=2, m=2, q=1) | SSE = 71.485878 | Elapsed: 45.0s
[ 26/ 36] (L=2, m=2, q=2) | SSE = 32.145172 | Elapsed: 47.0s
[ 27/ 36] (L=2, m=2, q=3) | SSE = 64.564056 | Elapsed: 48.9s
[ 28/ 36] (L=2, m=2, q=4) | SSE = 60.296486 | Elapsed: 51.0s
[ 29/ 36] (L=2, m=2, q=5) | SSE = 23.181992 | Elapsed: 53.0s
[ 30/ 36] (L=2, m=2, q=6) | SSE = 24.213696 | Elapsed: 55.1s
[ 31/ 36] (L=2, m=3, q=1) | SSE = 67.705353 | Elapsed: 56.7s
[ 32/ 36] (L=2, m=3, q=2) | SSE = 55.311882 | Elapsed: 58.7s
[ 33/ 36] (L=2, m=3, q=3) | SSE = 31.157841 | Elapsed: 60.7s
[ 34/ 36] (L=2, m=3, q=4) | SSE = 52.961189 | Elapsed: 62.7s
[ 35/ 36] (L=2, m=3, q=5) | SSE = 25.212608 | Elapsed: 64.7s
[ 36/ 36] (L=2, m=3, q=6) | SSE = 21.172302 | Elapsed: 66.7s

Done in 66.7s!
PyTorch Best Orders (L, m, q): (1, 1, 5) with SSE = 1.045434
