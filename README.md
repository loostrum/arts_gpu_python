# ARTS GPU Python
Some tools to test GPU-based analysis tools and compare these to the CPU-based versions, mostly for the Apertif Radio Transient System.


## beamformer.py
Simulation of the ARTS tied-array beamformer.


## dedisp.py
Dedisperse a frequency-time-intensity data cube, given sampling time, frequency channels, and range of dispersion measures.
Output is a dm-time-intensity cube.
Performance results (CPU = AMD Ryzen 1700X 8C / 16T, GPU = NVIDIA Geforce GTX 1080 Ti) for 320 DM steps, 1536 frequency channels, 1000 time steps:
* CPU: 12340 ms
* CPU parallelized: 2087 ms
* GPU: 235 ms
* GPU without host <-> device copies: 32 ms
