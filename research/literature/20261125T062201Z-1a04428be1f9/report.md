# 文献检索报告

"trend" AND "seasonality" AND "interpretable forecasting"

实验交接 [cycle-e6bd782009166d68](../../memory/cycles/cycle-e6bd782009166d68.json)

本轮未解问题 · Which open-source controlled identification benchmarks can distinguish denoising or local velocity trend estimation from additional predictive-state memory, and separate command encoding effects from actuator dynamics w…

运行 `20261125T062201Z-1a04428be1f9` · 2026-11-25T06:22:01.427355+00:00 · 状态 **READY_FOR_SCREENING**

覆盖验收 **PARTIAL**
- q1 · "trend" AND "seasonality" AND "interpretable forecasting" 实取 9/67，检查单路上限或调整检索式
- q2 · "seasonal-trend decomposition" AND "forecasting" 实取 9/795，检查单路上限或调整检索式
- q5 · "harmonic" AND "trend" AND "forecasting" AND "neural network" 实取 9/29，检查单路上限或调整检索式
- q7 · How can an additive forecast model distinguish periodic motion from slow trend and attribute forecast gains with matched ablations? 实取 9/50，检查单路上限或调整检索式
- q12 · W4396917366：HTTP 403，请检查 API 访问权限或额度
- q13 · W4396917366：HTTP 403，请检查 API 访问权限或额度
- q14 · W7167581812：HTTP 403，请检查 API 访问权限或额度
- q15 · W7167581812：HTTP 403，请检查 API 访问权限或额度
- q16 · W4409158573：HTTP 403，请检查 API 访问权限或额度
- q17 · W4409158573：HTTP 403，请检查 API 访问权限或额度
- q18 · W7160600450：HTTP 403，请检查 API 访问权限或额度
- q19 · W7160600450：HTTP 403，请检查 API 访问权限或额度
- q20 · W7160705014：HTTP 403，请检查 API 访问权限或额度
- q21 · W7160705014：HTTP 403，请检查 API 访问权限或额度
- q22 · W4402640700：HTTP 403，请检查 API 访问权限或额度

## 确定性初筛 Top-3（实际 3 篇）

### 1. [Characterizing Seasonality and Trend From In Situ Time‐Series Observations Using Explainable Deep Learning for Ground Deformation Forecasting](<https://doi.org/10.1029/2024jh000122>) · 2024 · DOI `10.1029/2024jh000122`

优先级 65/100。发现路径 q1。

- 主题匹配词：forecasting、interpretable、seasonality、trend
- 可核查的机制线索：ground deformation、long-term trends、explainable deep learning
- 比较价值：由主题与关键词发现，可作主题对照
- 公开实现线索（query_failed）：查询失败，覆盖不完整
- 复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知

### 2. [An Adaptive Prophet-based Hybrid Framework for Volatility-Aware Trend and Seasonality Modeling in Stock Market Price Forecasting](<https://doi.org/10.1109/iccnct68477.2026.11590099>) · 2026 · DOI `10.1109/iccnct68477.2026.11590099`

优先级 64/100。发现路径 q1。

- 主题匹配词：forecasting、interpretable、seasonality、trend
- 可核查的机制线索：stock price prediction、Facebook Prophet、hybrid forecasting model
- 比较价值：由主题与关键词发现，可作主题对照
- 公开实现线索（query_failed）：查询失败，覆盖不完整
- 复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知

### 3. [D-Tracker: Modeling Interest Diffusion in Social Activity Tensor Data Streams](<https://doi.org/10.1145/3690624.3709192>) · 2025 · DOI `10.1145/3690624.3709192`

优先级 64/100。发现路径 q1。

- 主题匹配词：forecasting、interpretable、seasonality、trend
- 可核查的机制线索：tensor decomposition、partial differential equations、interpretable forecasting
- 比较价值：由主题与关键词发现，可作主题对照
- 公开实现线索（unverified）：https://github.com/Higashiguchi-Shingo/D-Tracker
- 复现成本：有仓库线索，需核对归属与实现范围，数据获取条件待核实，算力预算未知

下一步用 review-packet 初筛，提交入围名单与暂定排序。

## 全部候选（按价值排序）

1. [Characterizing Seasonality and Trend From In Situ Time‐Series Observations Using Explainable Deep Learning for Ground Deformation Forecasting](<https://doi.org/10.1029/2024jh000122>) · 2024 · DOI `10.1029/2024jh000122` · 65 分 · q1
   主题匹配词：forecasting、interpretable、seasonality、trend。可核查的机制线索：ground deformation、long-term trends、explainable deep learning。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（query_failed）：查询失败，覆盖不完整。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
2. [An Adaptive Prophet-based Hybrid Framework for Volatility-Aware Trend and Seasonality Modeling in Stock Market Price Forecasting](<https://doi.org/10.1109/iccnct68477.2026.11590099>) · 2026 · DOI `10.1109/iccnct68477.2026.11590099` · 64 分 · q1
   主题匹配词：forecasting、interpretable、seasonality、trend。可核查的机制线索：stock price prediction、Facebook Prophet、hybrid forecasting model。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（query_failed）：查询失败，覆盖不完整。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
3. [D-Tracker: Modeling Interest Diffusion in Social Activity Tensor Data Streams](<https://doi.org/10.1145/3690624.3709192>) · 2025 · DOI `10.1145/3690624.3709192` · 64 分 · q1
   主题匹配词：forecasting、interpretable、seasonality、trend。可核查的机制线索：tensor decomposition、partial differential equations、interpretable forecasting。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（unverified）：https://github.com/Higashiguchi-Shingo/D-Tracker。复现成本：有仓库线索，需核对归属与实现范围，数据获取条件待核实，算力预算未知。
4. [Forecasting hospital bed occupancy: a time series approach with prophet](<https://doi.org/10.1186/s12911-026-03542-w>) · 2026 · DOI `10.1186/s12911-026-03542-w` · 60 分 · q1
   主题匹配词：forecasting、interpretable、seasonality、trend。可核查的机制线索：hospital bed occupancy、prophets、interpretable forecasting。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（query_failed）：查询失败，覆盖不完整。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
5. [Forecasting hospital bed occupancy: a time series approach with prophet](<https://doi.org/10.6084/m9.figshare.c.8467281>) · 2026 · DOI `10.6084/m9.figshare.c.8467281` · 60 分 · q1
   主题匹配词：forecasting、interpretable、seasonality、trend。可核查的机制线索：hospital bed occupancy、prophets、interpretable forecasting。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（query_failed）：查询失败，覆盖不完整。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
6. [Time Series Decomposition of Land Surface Temperature for Long-Term Trend Forecasting and Impact on Nesting Sea Turtle Habitats in the Arabian Gulf](<https://doi.org/10.1175/jamc-d-24-0092.1>) · 2024 · DOI `10.1175/jamc-d-24-0092.1` · 57 分 · q5
   主题匹配词：forecasting、seasonality、trend。可核查的机制线索：land surface temperature、time series decomposition、sea turtle nesting habitat；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（query_failed）：查询失败，覆盖不完整。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
7. [Interpretable wind power forecasting combining seasonal-trend representations learning with temporal fusion transformers architecture](<https://doi.org/10.1016/j.energy.2024.132482>) · 2024 · DOI `10.1016/j.energy.2024.132482` · 55 分 · q1
   主题匹配词：forecasting、interpretable、trend。可核查的机制线索：wind power forecasting、interpretable forecasting、Temporal Fusion Transformer。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
8. [Interpretable Forecasting of Container Throughput for Integrated Port Operations: A Data-Driven Analysis from a Single-Port System](<https://doi.org/10.56578/ijtdi100205>) · 2026 · DOI `10.56578/ijtdi100205` · 52 分 · q1
   主题匹配词：forecasting、interpretable、trend。可核查的机制线索：container throughput forecasting、interpretable forecasting、seasonal patterns。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
9. [RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series](<https://doi.org/10.1609/aaai.v33i01.33015409>) · 2019 · DOI `10.1609/aaai.v33i01.33015409` · 50 分 · q2
   主题匹配词：forecasting、seasonality、trend。可核查的机制线索：seasonal-trend decomposition、least absolute deviation、sparse regularization。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
10. [CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](<https://doi.org/10.48550/arxiv.2202.01575>) · 2022 · DOI `10.48550/arxiv.2202.01575` · 50 分 · q9
   主题匹配词：forecasting、trend。可核查的机制线索：time series forecasting、contrastive learning、disentangled representations。比较价值：由已知论文回查发现，可作主题对照。公开实现线索（unverified）：https://github.com/salesforce/CoST。复现成本：有仓库线索，需核对归属与实现范围，数据获取条件待核实，算力预算未知。
11. [A Hybrid Framework for Air Pollution Forecasting using DSES, DLA, Self-Attention Transformer, and BiGRU with Bayesian Optimization](<https://doi.org/10.1038/s41598-026-61520-8>) · 2026 · DOI `10.1038/s41598-026-61520-8` · 49 分 · q1
   主题匹配词：forecasting、interpretable、trend。可核查的机制线索：hybrid forecasting model、transformer self-attention、bidirectional gated recurrent unit。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
12. [prophet: Automatic Forecasting Procedure](<https://doi.org/10.32614/cran.package.prophet>) · 2017 · DOI `10.32614/cran.package.prophet` · 49 分 · q7
   主题匹配词：forecasting、seasonality、trend。可核查的机制线索：time series forecasting、prophets、additive models。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
13. [Periodic Modeling Prediction Model Based on CS-Mamba](<https://doi.org/10.1109/icmlca66850.2025.11336258>) · 2025 · DOI `10.1109/icmlca66850.2025.11336258` · 49 分 · q4
   主题匹配词：forecasting、seasonality、trend。可核查的机制线索：Mamba、selective state space models、long-term forecasting。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，摘要提及 public datasets，算力预算未知。
14. [OneShotSTL: One-Shot Seasonal-Trend Decomposition For Online Time Series Anomaly Detection And Forecasting](<https://doi.org/10.14778/3583140.3583155>) · 2023 · DOI `10.14778/3583140.3583155` · 45 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：seasonal-trend decomposition、time series anomaly detection、time series forecasting。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，摘要提及 benchmark datasets，算力预算未知。
15. [A Systematic Comparative Examination of Time Series Forecasting Models for Financial Market Trend Prediction](<https://doi.org/10.1007/978-3-032-30957-0_22>) · 2026 · DOI `10.1007/978-3-032-30957-0_22` · 44 分 · q1
   主题匹配词：forecasting、interpretable、trend。可核查的机制线索：financial time series forecasting。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
16. [SSA-based hybrid forecasting models and applications](<https://doi.org/10.11591/eei.v9i5.1950>) · 2020 · DOI `10.11591/eei.v9i5.1950` · 44 分 · q5
   主题匹配词：forecasting、trend。可核查的机制线索：singular spectrum analysis、hybrid forecasting model、time series forecasting；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
17. [Aggregating Prophet and Seasonal Trend Decomposition for Time Series Forecasting of Italian Electricity Spot Prices](<https://doi.org/10.3390/en16031371>) · 2023 · DOI `10.3390/en16031371` · 43 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：electricity spot prices、time series forecasting、prophets。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
18. [Effective LSTMs with seasonal-trend decomposition and adaptive learning and niching-based backtracking search algorithm for time series forecasting](<https://doi.org/10.1016/j.eswa.2023.121202>) · 2023 · DOI `10.1016/j.eswa.2023.121202` · 43 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：time series forecasting、seasonal-trend decomposition、backtracking。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
19. [Interpretable PM2.5 Forecasting for Urban Air Quality: A Comparative Study of Adaptive Time-Series Models](<https://doi.org/10.48550/arxiv.2603.25495>) · 2026 · DOI `10.48550/arxiv.2603.25495` · 43 分 · q3
   主题匹配词：forecasting、interpretable。可核查的机制线索：PM2.5 forecasting、short-term air quality prediction、interpretable forecasting。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
20. [xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition](<https://doi.org/10.1609/aaai.v39i19.34270>) · 2025 · DOI `10.1609/aaai.v39i19.34270` · 43 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：time series forecasting、dual-stream architecture、exponential smoothing。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
21. [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](<https://doi.org/10.48550/arxiv.1905.10437>) · 2019 · DOI `10.48550/arxiv.1905.10437` · 41 分 · q8
   主题匹配词：forecasting、interpretable。可核查的机制线索：N-BEATS、residual blocks、deep learning architectures。比较价值：由已知论文回查发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
22. [Beyond the Hype: Comparing Lightweight and Deep Learning Models for Air Quality Forecasting](<https://doi.org/10.48550/arxiv.2512.09076>) · 2025 · DOI `10.48550/arxiv.2512.09076` · 40 分 · q3
   主题匹配词：forecasting、interpretable。可核查的机制线索：air quality forecasting、particulate matter、Facebook Prophet。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
23. [CycleDegrade: An Explicit Periodicity Modeling Approach for Long-Term Fuel Cell Voltage Degradation Forecasting](<https://doi.org/10.1109/tii.2026.3731609>) · 2026 · DOI `10.1109/tii.2026.3731609` · 40 分 · q4
   主题匹配词：forecasting、trend。可核查的机制线索：degradation prediction、periodicity modeling、long-term degradation。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
24. [Forecasting the pulse: How deviations from regular patterns in online data can identify offline phenomena](<https://doi.org/10.1108/intr-06-2012-0115>) · 2013 · DOI `10.1108/intr-06-2012-0115` · 40 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：online data、Twitter、hashtags。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
25. [MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns](<https://doi.org/10.1504/ijor.2025.143957>) · 2022 · DOI `10.1504/ijor.2025.143957` · 40 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：seasonal-trend decomposition、high-frequency time series、time series decomposition。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
26. [STR: A Seasonal-Trend Decomposition Procedure Based on Regression](<https://doi.org/10.26180/21521289>) · 2022 · DOI `10.26180/21521289` · 40 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：seasonal-trend decomposition、STR、time series decomposition。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
27. [STR: Seasonal-Trend Decomposition Using Regression](<https://doi.org/10.32614/cran.package.str>) · 2016 · DOI `10.32614/cran.package.str` · 40 分 · q2
   主题匹配词：forecasting、trend。可核查的机制线索：seasonal-trend decomposition、time series decomposition、seasonal component。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
28. [A Comparative Evaluation of Harmonic Analysis and Neural Networks for Sea Level Prediction in the Northern South China Sea](<https://doi.org/10.3390/su17136081>) · 2025 · DOI `10.3390/su17136081` · 40 分 · q5
   主题匹配词：forecasting、trend。可核查的机制线索：sea level prediction、harmonic analysis、artificial neural networks；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
29. [APPLICABILITY OF ARTIFICIAL NEURAL NETWORK MODEL FOR SIMULATION OF MONTHLY RUNOFF IN COMPARISON WITH SOM OTHER TRADITIONAL MODELS](<https://doi.org/10.32508/stdj.v12i4.2237>) · 2009 · DOI `10.32508/stdj.v12i4.2237` · 40 分 · q5
   主题匹配词：forecasting、trend。可核查的机制线索：artificial neural networks、back-propagation algorithm、M5 model tree；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
30. [Application of Neural Networks to Predict Power Consumption of a Megapolis](<https://doi.org/10.1109/icieam57311.2023.10139236>) · 2023 · DOI `10.1109/icieam57311.2023.10139236` · 40 分 · q5
   主题匹配词：forecasting、trend。可核查的机制线索：Moscow、fuzzy neural network、meteorological factors；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
31. [Harmonics and Neurons: a Fourier-Neural approach to energy pattern analysis](<https://doi.org/10.1007/s42452-025-06540-1>) · 2025 · DOI `10.1007/s42452-025-06540-1` · 40 分 · q5
   主题匹配词：forecasting、trend。可核查的机制线索：time series forecasting、per capita energy consumption、periodic trends；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
32. [System Analysis and Forecast of Yield Time Series Based on Neural Network Technologies](<https://doi.org/10.18280/ijdne.180224>) · 2023 · DOI `10.18280/ijdne.180224` · 40 分 · q5
   主题匹配词：forecasting、trend。可核查的机制线索：high-order Markov chains、transition probability matrices、long-term memory；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
33. [TIME SERIES AND NEURAL NETWORK ALGORITHMS IN REAL ESTATE VALUATION](<https://doi.org/10.26102/2310-6018/2018.22.3.007>) · 2018 · DOI `10.26102/2310-6018/2018.22.3.007` · 40 分 · q5
   主题匹配词：forecasting、trend。可核查的机制线索：housing prices、price forecasting、time series analysis；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
34. [CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns](<https://doi.org/10.52202/079017-3373>) · 2024 · DOI `10.52202/079017-3373` · 39 分 · q11, q4
   主题匹配词：forecasting。可核查的机制线索：periodic patterns、weather forecasting、energy forecasting。比较价值：由已知论文回查、主题与关键词发现，可作主题对照。公开实现线索（unverified）：https://github.com/ACAT-SCUT/CycleNet。复现成本：有仓库线索，需核对归属与实现范围，数据获取条件待核实，算力预算未知。
35. [Communications to the Editor—Exponential Forecasting: Some New Variations](<https://doi.org/10.1287/mnsc.15.5.311>) · 1969 · DOI `10.1287/mnsc.15.5.311` · 38 分 · q7
   主题匹配词：forecasting、trend。可核查的机制线索：exponential smoothing、trends、cyclical effects。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
36. [NeuralProphet: Explainable Forecasting at Scale](<https://doi.org/10.48550/arxiv.2111.15397>) · 2021 · DOI `10.48550/arxiv.2111.15397` · 38 分 · q10
   主题匹配词：forecasting、interpretable。可核查的机制线索：NeuralProphet、time series forecasting、explainable forecasting。比较价值：由已知论文回查发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
37. [The Importance of Taking into Account the Results of Additive Modeling During Dynamics Process Analyzing](<https://doi.org/10.31521/modecon.v38(2023)-01>) · 2023 · DOI `10.31521/modecon.v38(2023)-01` · 36 分 · q7
   主题匹配词：forecasting、trend。可核查的机制线索：gross domestic product、GDP forecasting、seasonal component。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
38. [Wind power forecasting based on singular spectrum analysis and a new hybrid Laguerre neural network](<https://doi.org/10.1016/j.apenergy.2019.114139>) · 2019 · DOI `10.1016/j.apenergy.2019.114139` · 32 分 · q5
   主题匹配词：forecasting。可核查的机制线索：wind power forecasting、singular spectrum analysis、hybrid forecasting model；可与该检索路线对照："harmonic" AND "trend" AND "forecasting" AND "neural network"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
39. [Forecasting crude oil futures using an ensemble model including investor sentiment and attention](<https://doi.org/10.1108/k-03-2023-0364>) · 2023 · DOI `10.1108/k-03-2023-0364` · 30 分 · q3
   主题匹配词：forecasting。可核查的机制线索：crude oil futures、futures price forecasting、investor sentiment。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
40. [A comparison of focus forecasting with averaging and exponential smoothing](<https://openalex.org/W2288789415>) · 1986 · DOI `未收录` · 28 分 · q7
   主题匹配词：forecasting。可核查的机制线索：averaging、exponential smoothing、forecasting method comparison。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
41. [A note on aggregation, disaggregation and forecasting performance](<https://doi.org/10.1002/1099-131x(200009)19:5<457::aid-for761>3.0.co;2-6>) · 2000 · DOI `10.1002/1099-131x(200009)19:5<457::aid-for761>3.0.co;2-6` · 28 分 · q7
   主题匹配词：forecasting。可核查的机制线索：aggregation、disaggregation、forecasting performance。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
42. [Additive models with trend filtering](<https://doi.org/10.1214/19-aos1833>) · 2019 · DOI `10.1214/19-aos1833` · 28 分 · q7
   主题匹配词：trend。可核查的机制线索：additive models、backfitting、linear smoothers。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
43. [FM-LLM: A frequency-enhanced mixture-of-experts framework for adapting LLMs to time series forecasting](<https://doi.org/10.1016/j.knosys.2026.115776>) · 2026 · DOI `10.1016/j.knosys.2026.115776` · 28 分 · q6
   主题匹配词：forecasting。可核查的机制线索：time series forecasting、mixture of experts；可与该检索路线对照："time series forecasting" AND "periodic residual"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
44. [Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models](<https://doi.org/10.52202/075280-0834>) · 2023 · DOI `10.52202/075280-0834` · 27 分 · q3
   主题匹配词：forecasting。可核查的机制线索：generalized additive models、neural additive models、NeuralProphet。比较价值：由主题与关键词发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
45. [Multi-Scale Periodic Residual State Space Model for Time Series Forecasting](<https://doi.org/10.1007/978-981-95-0006-2_36>) · 2025 · DOI `10.1007/978-981-95-0006-2_36` · 25 分 · q6
   主题匹配词：forecasting。可核查的机制线索：time series forecasting；可与该检索路线对照："time series forecasting" AND "periodic residual"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
46. [A note on a comparison of exponential smoothing methods for forecasting seasonal series](<https://doi.org/10.1016/0169-2070(89)90068-x>) · 1989 · DOI `10.1016/0169-2070(89)90068-x` · 24 分 · q7
   主题匹配词：forecasting。可核查的机制线索：exponential smoothing、seasonal forecasting。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
47. [Monitoring and adjusting forecasts in the presence of additive outliers](<https://doi.org/10.1002/for.3980030208>) · 1984 · DOI `10.1002/for.3980030208` · 22 分 · q7
   主题匹配词：forecasting。可核查的机制线索：additive outliers、forecast accuracy。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
48. [CDSFNet: cycle-anchored dual-spectrum forecasting with confidence-adaptive fusion for long-term multivariate time series](<https://doi.org/10.1007/s11227-026-08925-4>) · 2026 · DOI `10.1007/s11227-026-08925-4` · 20 分 · q6
   主题匹配词：forecasting。可核查的机制线索：phase consistency；可与该检索路线对照："time series forecasting" AND "periodic residual"。比较价值：由替代方法路线发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。
49. [Level-adjusted exponential smoothing for modeling planned discontinuities](<https://doi.org/10.1016/s0169-2070(98)00083-1>) · 1999 · DOI `10.1016/s0169-2070(98)00083-1` · 2 分 · q7
   主题匹配词：无直接词项命中。可核查的机制线索：level shifts、forecast updating、forecast accuracy。比较价值：由研究问题语义检索发现，可作主题对照。公开实现线索（not_checked）：尚未查仓。复现成本：实现工作量待核实，数据获取条件待核实，算力预算未知。

## 种子与检索覆盖

- q1 主题与关键词 · "trend" AND "seasonality" AND "interpretable forecasting" · 取回 9/67（覆盖有限）
- q2 主题与关键词 · "seasonal-trend decomposition" AND "forecasting" · 取回 9/795（覆盖有限）
- q3 主题与关键词 · "additive" AND "neuralprophet" · 取回 4/4
- q4 主题与关键词 · "periodic patterns" AND "residual cycle forecasting" · 取回 3/3
- q5 替代方法路线 · "harmonic" AND "trend" AND "forecasting" AND "neural network" · 取回 9/29（覆盖有限）
- q6 替代方法路线 · "time series forecasting" AND "periodic residual" · 取回 3/3
- q7 研究问题语义检索 · How can an additive forecast model distinguish periodic motion from slow trend and attribute forecast gains with matched ablations? · 取回 9/50（覆盖有限）
- q8 已知论文回查 · N-BEATS: Neural basis expansion analysis for interpretable time series forecasting · 取回 1/1
- q9 已知论文回查 · CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting · 取回 1/1
- q10 已知论文回查 · NeuralProphet: Explainable Forecasting at Scale · 取回 1/1
- q11 已知论文回查 · CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns · 取回 1/1
- q12 公开实现查询 · W4396917366 · HTTP 403，请检查 API 访问权限或额度
- q13 公开实现查询 · W4396917366 · HTTP 403，请检查 API 访问权限或额度
- q14 公开实现查询 · W7167581812 · HTTP 403，请检查 API 访问权限或额度
- q15 公开实现查询 · W7167581812 · HTTP 403，请检查 API 访问权限或额度
- q16 公开实现查询 · W4409158573 · HTTP 403，请检查 API 访问权限或额度
- q17 README 归属核查 · W4409158573 · HTTP 403，请检查 API 访问权限或额度
- q18 公开实现查询 · W7160600450 · HTTP 403，请检查 API 访问权限或额度
- q19 公开实现查询 · W7160600450 · HTTP 403，请检查 API 访问权限或额度
- q20 公开实现查询 · W7160705014 · HTTP 403，请检查 API 访问权限或额度
- q21 公开实现查询 · W7160705014 · HTTP 403，请检查 API 访问权限或额度
- q22 公开实现查询 · W4402640700 · HTTP 403，请检查 API 访问权限或额度

### 覆盖限制

- 主题、语义及引用扩展来自 OpenAlex。已知条目可经 DataCite 精确补录。
- 初排前六额外核查 GitHub 仓库和 README；归属见各候选的证据等级，可运行性仍待验证。
- 分数衡量元数据支持的阅读优先级。数据准备、训练成本和论文结果需在原文阶段验证。
- q12 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q13 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q14 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q15 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q16 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q17 README 核查失败（HTTP 403，请检查 API 访问权限或额度）
- q18 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q19 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q20 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q21 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- q22 公开实现查询失败（HTTP 403，请检查 API 访问权限或额度），此路径覆盖未知。
- 未提供种子，施引、参考文献和同作者路径未执行。

## 补检对照

父运行 20261125T062113Z-dfd8a4a333a6 · PARTIAL → PARTIAL
新增 8 篇，本轮未取回 9 篇。
