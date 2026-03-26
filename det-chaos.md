

# The Dynamic Existence Threshold: Organizational Consciousness across Complex Systems

**Nathan M. Thornhill**

ORCID: 0009-0009-3161-528X

Independent Researcher, Fort Wayne, Indiana, USA

---

## Abstract

Complex systems across disparate substrates share a common failure mode: the breakdown of balance between integration and differentiation. This paper introduces the Dynamic Existence Threshold (DET), a framework in which system state is characterized by two measurable quantities — integration (I), capturing coordinated multi-component coupling, and differentiation (D), capturing divergence from uniformity. Four canonical states emerge from the I-D plane: healthy idle (low I, high D), critical balance (high I, high D), pathological cascade (high I, low D), and system death (low I, low D). The DET predicts that critical events correspond to exits from the I-D balance zone and that approach to such exits produces measurable early warning signals. These predictions are tested across three domains — financial markets (6,785 trading days, 26 cascade events), space weather (9,802 days, 51 storm days), and human EEG (50 subjects, 57,212 epochs, Wake vs. N3 deep sleep) — using a five-layer measurement architecture with zero-parameter tuning between domains. Five tests confirm the framework: (1) events cluster at I-D extremes; (2) events exit the balance zone faster than normal state changes (p = 0.0001); (3) rising variance of |I − D| provides early warning 5–30 days before events (2.0× elevation, p < 0.0001); (4) exit velocity discriminates events from normal with area under the curve (AUC) 0.692–0.843; and (5) in EEG, simple summation anti-predicts brain state (AUC 0.416) while the coupling metric R × S achieves AUC 0.909 (95% CI [0.904, 0.913]), demonstrating that the framework captures multi-scale coordination rather than mere magnitude.

**Keywords:** integration-differentiation balance, critical transitions, early warning signals, pattern persistence, complex systems, existence threshold

---

When financial markets crash, geomagnetic storms erupt, or the brain descends into deep sleep, the underlying systems shift into qualitatively different states. Despite vast differences in substrate, these transitions share a structural signature: the breakdown of balance between how tightly a system's components are coupled (integration) and how distinctly they behave (differentiation). This paper introduces a measurable coordinate system — the integration-differentiation (I-D) plane — in which critical transitions across all three domains appear as departures from a common balance zone. Using identical metrics with no tuning between domains, we show that approaching these departures produces early warning signals consistent with critical slowing down, and that the framework captures structural coordination rather than mere signal magnitude — demonstrated by the finding that simple summation of neural frequency bands anti-predicts brain state while the structural coupling metric achieves 91% classification accuracy.

## 1. Introduction

A persistent question in complexity science is whether disparate systems — brains, markets, magnetospheres — undergo critical transitions through shared mechanisms or through substrate-specific processes that merely resemble one another superficially. The critical transitions literature has established that rising variance and autocorrelation precede tipping points across ecological, climatic, and financial systems [1, 2], but these indicators describe *symptoms* of approaching transitions without specifying the *coordinate system* in which the transition occurs.

Integrated Information Theory (IIT) introduced the quantity Φ to characterize the degree to which a system is "more than the sum of its parts" [3, 4]. While IIT was developed primarily as a theory of consciousness, its core insight — that integration and differentiation jointly characterize system complexity — has implications far beyond neuroscience. Global Workspace Theory similarly emphasizes the interplay between distributed processing and unified access [5]. Causal density, defined as the fraction of causally significant interactions in a network, has been proposed as a measure of dynamical complexity that peaks when integration and differentiation are jointly high [6].

The present work synthesizes these threads. IIT's Φ remains intractable for large-scale systems, and critical slowing down indicators, while powerful, describe symptoms without specifying the coordinate system in which transitions occur. Partial information decomposition (PID) has further revealed that synergistic and redundant information components respond asymmetrically to system perturbation [32, 33], suggesting that coordination and overlap play distinct roles in system stability. The present paper proposes a dynamic framework grounded in these insights: systems do not merely need positive integrated information to persist — they must maintain balance between integration and differentiation over time.

The central hypothesis is straightforward. Integration (I) measures the degree of coordinated coupling among system components. Differentiation (D) measures the degree to which the system's activity profile diverges from uniformity. These two quantities define a phase space with four canonical states:

| State | I | D | Interpretation |
|---|---|---|---|
| Normal (quiet/idle) | Low | High | Components active but uncoupled |
| Critical balance | High | High | Coordinated, differentiated activity |
| Cascade/seizure/crash | High | Low | Pathological synchronization |
| Death/off | Low | Low | System inactive |

The Dynamic Existence Threshold (DET) is the boundary of the balance zone in I-D space. Systems persist within this zone; critical events correspond to exits from it. We use "existence" in the information-theoretic sense: a system *exists as an organized entity* when its integrated information is positive. When I-D balance breaks, the system persists physically but its organizational identity — the coordinated structure that distinguishes it from its components — degrades. A market in cascade is still a market in name; it is not functioning as one. A brain in deep sleep still generates electrical activity; it has lost the cross-frequency coordination that characterizes conscious processing. The threshold is not physical death but organizational dissolution — the loss of what we term *multi-scale coordination*, the property of a complex system whose components are simultaneously integrated and differentiated. Crucially, the approach to this boundary should produce early warning signals — rising variance in the balance metric — consistent with the critical slowing down framework [1].

This paper tests five predictions derived from the DET hypothesis across three domains (financial markets, space weather, and human EEG) using a common five-layer measurement architecture with zero-parameter tuning between domains. The inclusion of EEG data proves decisive: a simple summation metric anti-predicts brain state (performing worse than chance), while the coupling-structure metric achieves high classification accuracy. This dissociation demonstrates that the I-D framework captures organizational structure rather than mere signal magnitude — a distinction invisible to simpler approaches.

---


## 2. Methods

### 2.1 Five-Layer Architecture

To enable cross-domain comparison without parameter tuning, each domain is decomposed into five measurement layers representing distinct scales or modalities of system activity. The five-layer decomposition follows a principle of hierarchical coverage: layers span from the most local or fundamental process (L₀) to the most global or emergent indicator (L₄). Table I specifies the mapping.

**Table I.** Five-layer architecture across domains.

| Layer | EEG Sleep | Space Weather | Financial Markets |
|---|---|---|---|
| L₀ | Delta (0.5–4 Hz) | F10.7 solar flux | Stock market stress |
| L₁ | Theta (4–8 Hz) | Solar wind pressure | Sector coherence |
| L₂ | Alpha (8–13 Hz) | Disturbance Storm Time (Dst) index | Cross-sector correlation |
| L₃ | Beta (13–30 Hz) | Planetary K-index (Kp) | VIX implied volatility |
| L₄ | Gamma (30–45 Hz) | Auroral Electrojet (AE) index | Flight-to-safety flows |

The EEG layers correspond to standard clinical frequency bands [7, 8]. Gamma power was computed with awareness of potential electromyographic (EMG) contamination artifacts at frequencies above 30 Hz [9]; however, the framework's reliance on relative coupling rather than absolute power mitigates this concern. Space weather layers span the sun-magnetosphere coupling chain from solar surface (F10.7) through interplanetary medium (solar wind dynamic pressure) to magnetospheric response (Dst, Kp, AE), using NASA OMNI2 data [10]. Financial layers span from individual equity stress through sector-level coherence to market-wide measures of volatility and risk appetite [11]. **L₄ (flight-to-safety)** is the mean z-score of safe-haven assets (Treasury bonds, gold, US dollar), following the same z-score protocol used for all other layers across all domains. An earlier formulation used a conjunction gate conditioning haven activation on concurrent equity stress; ablation testing (Section 3.6.2) demonstrated that this additional logic did not improve classification performance (AUC 0.870 without L₄ vs. 0.868 with the conjunction gate), so the simpler formulation is used here to maintain consistency with the zero-tuning design.

All layer activations are z-scored against a 60-day rolling baseline and clipped to non-negative values. This ensures that only elevated activity relative to recent history registers as layer activation. The rolling baseline window is held constant across domains.

### 2.2 Metric Definitions

Four quantities are computed from the five-layer time series at each time step.

**Effective number of active layers (N_eff).** Following Hill [12] and Jost [13], the effective number of layers is computed as the exponential of Shannon entropy of the normalized layer magnitudes:

$$N_{\text{eff}} = \exp\left( -\sum_{i=0}^{N-1} p_i \ln p_i \right),$$

where $p_i = a_i / \sum_j a_j$ is the normalized activation of layer $i$. N_eff equals 1 when a single layer carries all activation and equals N when all layers are equally active. A minimum activation threshold θ = 2.0 gates the computation: if N_eff < θ, both R and S are set to zero. This threshold encodes the requirement that at least two layers must be meaningfully active for integration to be possible. The value θ = 2.0 was set a priori and held constant across all domains without tuning.

**Redundancy (R).** R measures the fraction of processing depth engaged:

$$R = \frac{N_{\text{eff}} - 1}{N - 1} \quad \text{when } N_{\text{eff}} \geq \theta,$$

where N = 5 is the total number of layers. R ranges from 0 (one dominant layer) to 1 (all layers equally active). The physical interpretation varies by domain: in EEG, high R means multiple frequency bands are simultaneously active; in space weather, high R means multiple layers of the sun-magnetosphere chain are elevated; in finance, high R means stress is distributed across multiple market layers.

**Synergy (S).** S captures the coordination and diversity of inter-layer coupling:

$$S = C \times \left(\frac{1}{2} J + \frac{1}{2} P\right) \quad \text{when } N_{\text{eff}} \geq \theta,$$

where the three components are:

- **Coverage (C):** The fraction of available layers effectively engaged: $C = N_{\text{eff}} / N$.

- **Evenness (J):** Pielou's evenness index [14], measuring how uniformly activation is distributed: $J = H / H_{\max}$, where $H = -\sum p_i \ln p_i$ is Shannon entropy and $H_{\max} = \ln N$. J = 0 when one layer dominates; J = 1 when all layers are equally active.

- **Pattern diversity (P):** The temporal diversity of which layer dominates across sub-windows: $P = |\{d_k : k = 1, \ldots, W\}| / W$, where $d_k = \arg\max_i(a_i)$ within the $k$-th sub-window, and W is the number of sub-windows. For EEG, W = 8 sub-windows within each 5-second analysis window. For daily-resolution domains, the trailing 8 days are used. P = 1/W when the same layer dominates every sub-window; P approaches 1 as sub-window dominant-layer diversity increases, reaching a maximum of min(N, W)/W when every sub-window has a different dominant layer (for N = 5 and W = 8, P_max = 5/8 = 0.625).

S ranges from 0 to a practical maximum that depends on N and W; for the standard configuration (N = 5, W = 8), S_max ≈ 0.81 because P is bounded by min(N, W)/W = 0.625. The equal weighting of J and P (½ each) is a zero-assumption default — optimizing this weighting per domain would undermine the zero-tuning design. High S requires that inter-layer coupling is widespread, balanced, and temporally variable — a stringent condition that is fragile under system perturbation.

**Integration (I).** The integration metric is defined as the product of redundancy and synergy:

$$I = R \times S.$$

This product form ensures that I is high only when both conditions are met: multiple components are active (R) *and* they are coordinated in a structured way (S). A system with high R but low S has many active layers that are uncoupled — busy but disorganized. A system with low R but high S has strong coupling among few layers — tightly bound but narrow. Only when both are high does the system exhibit the kind of multi-component coordinated integration that the I-D framework posits as necessary for the critical balance state.

The physical meaning of I = R × S per domain:

- **EEG:** High I means multiple frequency bands are simultaneously active *and* their activations are balanced and temporally coordinated. In wakefulness, alpha, beta, theta, and gamma all contribute, with structured inter-band coupling. In N3 deep sleep, delta dominates while other bands are suppressed, yielding low R and near-zero S regardless of total spectral power.
- **Space weather:** High I means multiple layers of the sun-magnetosphere chain are simultaneously elevated *and* the elevation is coordinated (e.g., a solar flare that drives solar wind pressure, depresses Dst, raises Kp, and activates auroral electrojets in a synchronized cascade).
- **Financial markets:** High I means stress is present across multiple market layers *and* the stress is coordinated rather than idiosyncratic — the hallmark of systemic risk as distinguished from sector-specific shocks [11].

**Differentiation (D).** D is computed as the Jensen-Shannon distance of the normalized layer activation vector from the uniform distribution [15, 16]:

$$D = \text{JSD}^{1/2}\left(\mathbf{p} \| \mathbf{u}\right),$$

where **p** is the normalized layer magnitude vector and **u** is the uniform distribution over N layers, with JSD computed using base-2 logarithms so that D ∈ [0, 1]. D is high when layers have unequal activations (the system is differentiated) and low when all layers are equally active (the system is uniform). This is complementary to R: high R with high D indicates a system where multiple components are active but with structured inequality, while high R with low D would indicate uniform activation — a state rarely observed in natural systems.

### 2.3 Balance Zone Definition

The I-D balance zone is defined in z-score space. For each domain, I and D time series are standardized to zero mean and unit variance. The balance metric is:

$$B = |z(I) - z(D)|.$$

The balance zone is the region where B < σ_threshold. Results are reported at three primary threshold levels — ±0.5σ, ±1.0σ, and ±1.5σ (with ±0.75σ added in robustness checks, Section 3.6) — to demonstrate that findings are robust to the specific choice of boundary rather than dependent on a single arbitrary threshold. This multi-threshold reporting addresses a legitimate concern about threshold sensitivity: any individual threshold could be accused of cherry-picking, but consistent results across a threefold range of thresholds cannot.

### 2.4 Derived Quantities

**Exit velocity.** The rate of change of the balance metric B between consecutive time steps:

$$V_{\text{exit}} = |B_t - B_{t-1}|.$$

High exit velocity indicates rapid departure from the balance zone.

**Variance of B (early warning signal).** A rolling window variance of the balance metric B = |z(I) − z(D)|, computed at multiple look-back horizons (5, 10, 15, 20, 25, and 30 days in time-series domains). Rising variance before an event is the signature of critical slowing down [1, 2]: as the system approaches the basin boundary, perturbations take longer to decay, increasing the variance of fluctuations.

**Residence time.** The number of consecutive time steps a system remains within the balance zone before exiting. Shorter residence times during event periods versus normal periods indicate that events are associated with reduced ability to maintain I-D balance.

### 2.5 Data Sources

**Financial markets.** 6,785 trading days spanning 27 years. Twenty-six cascade events (comprising 293 systemic days) were identified using standard criteria including days with CBOE Volatility Index (VIX) > 2σ above its trailing mean, cross-sector correlation > 0.8, and Merrill Lynch Option Volatility Estimate (MOVE) index spikes, corresponding to well-documented market stress episodes (e.g., 2008 financial crisis, 2020 COVID crash, 2010 flash crash). Layer construction used daily data for equity indices, sector exchange-traded funds (ETFs), VIX, and Treasury-equity correlations.

**Space weather.** 9,802 days of NASA OMNI2 hourly data averaged to daily resolution [10]. Fifty-one storm days from 26 geomagnetic storm events were identified using the standard Dst < −50 nT criterion [17] (where Dst is the Disturbance Storm Time index, a measure of magnetospheric ring current intensity), consistent with moderate-to-severe storm classification.

**EEG.** 50 subjects (57,212 total epochs, of which 43,342 Wake and 1,676 N3 were used in the analysis) from the Sleep-EDF dataset on PhysioNet [18, 19]. Wake and N3 deep sleep epochs were extracted using expert-scored hypnograms following American Academy of Sleep Medicine (AASM) criteria [7]. Each 30-second epoch was decomposed into the five frequency bands of Table I via bandpass filtering. Band power was computed within six non-overlapping 5-second windows per epoch; the pattern diversity metric P (Section 2.2) was calculated over W = 8 windows using the trailing two windows from the preceding epoch to maintain temporal context.

### 2.6 Statistical Tests

All p-values are computed using non-parametric methods: Mann-Whitney U tests for group comparisons and permutation tests (10,000 iterations) for variance ratios. AUC values are computed using standard receiver operating characteristic analysis. Bootstrap 95% confidence intervals (1,000 iterations) are reported for key AUC estimates. Benjamini-Hochberg false discovery rate (FDR) correction [20] is applied across all 96 reported p-values. All 64 tests significant at p < 0.05 remain significant after FDR correction, and no tests lost significance.

### 2.7 Zero Parameter Tuning

A critical design principle of this study is that no parameters are tuned between domains. The five-layer architecture, the metric definitions (R, S, D, I), the balance zone thresholds, and the statistical tests are identical across financial, space weather, and EEG domains. The only domain-specific element is the mapping from raw data to the five layers (Table I), which is determined by the physics of each domain rather than by optimization. This constraint is deliberately restrictive: a framework that requires per-domain calibration would be of limited theoretical interest, as it would leave open the possibility that apparent cross-domain consistency is an artifact of fitting.

---

## 3. Results

### 3.1 Test 1: I-D Phase Space Structure

The first prediction of the DET framework is that the I-D phase space should exhibit characteristic structure: events should cluster at the extremes (high I / low D or low I / high D), while normal states should occupy the center.

Across all three domains, I and D show a clean negative correlation within event periods: as I rises, D falls. This is consistent with a budget constraint interpretation — the system has a limited capacity for total complexity, and increasing integration comes at the cost of differentiation (and vice versa). This negative I-D relationship is not imposed by the metric definitions but emerges from the data.

To quantify the directional dynamics, each time step was classified into one of four categories based on the signs of dI and dD (the one-step changes in I and D):

- **CASCADE** (dI > 0, dD < 0): Integration rising while differentiation falls — the signature of pathological synchronization.
- **RECOVERY** (dI < 0, dD > 0): Integration falling while differentiation rises — return toward normal.
- **COINCIDENCE** (dI > 0, dD > 0): Both rising simultaneously.
- **COLLAPSE** (dI < 0, dD < 0): Both falling simultaneously.

The CASCADE category is strongly enriched during events: 45–51% of event time steps fall in this category, compared to 21–34% of normal time steps. The COINCIDENCE category — in which both I and D rise simultaneously — accounts for less than 2% of event time steps, indicating that the budget constraint between I and D is nearly absolute during critical periods.

Perhaps most striking, events systematically flee the balance zone. In space weather, only 17.6% of event time steps occupy the balance zone (at ±1.0σ), compared to 44.9% of normal time steps — a 2.5× depletion during events. This depletion is consistent across all three domains, confirming the DET prediction that critical events correspond to departures from I-D balance.

### 3.2 Test 2: Residence Time Discrimination

The second prediction is that events should exit the balance zone faster than normal state changes — that is, residence times within the balance zone should be shorter during event periods.

**Financial markets.** At the ±1.5σ balance zone boundary, the median residence time during normal periods is 2 days, while during event periods it is 1 day (Mann-Whitney U test, p = 0.0001). This indicates that event periods are characterized by a halving of the system's ability to maintain I-D balance.

**Space weather.** At the ±1.5σ boundary with Sum as the I coordinate, the median residence time during normal periods is 3 days, compared to 2 days during event periods (p = 0.01). Using a kernel density estimation (KDE)-based adaptive boundary, the financial result remains significant (p = 0.0003), but the space weather result does not reach significance (p = 0.66), suggesting that the fixed-threshold approach is more robust for this domain.

The residence time results are reported at multiple thresholds in Table II.

**Table II.** Balance zone residence time (median days) by domain and threshold.

| Domain | Threshold | Normal | Event | p-value |
|---|---|---|---|---|
| Financial | ±1.5σ | 2 | 1 | 0.0001 |
| Financial | KDE | 2 | 1 | 0.0003 |
| Space | ±1.5σ | 3 | 2 | 0.01 |

The discrimination improves at wider thresholds, consistent with the interpretation that larger balance zones capture more of the normal-state dynamics while still showing rapid event-period exits.

### 3.3 Test 3: Early Warning Signal

The third and most practically significant prediction is that rising variance of the balance metric B should precede events, providing an early warning signal. This prediction connects the DET framework directly to the critical transitions literature: as a system approaches the boundary of its balance zone, its fluctuations should grow — the hallmark of critical slowing down [1, 2].

**Financial markets.** The 30-day rolling variance of B shows a 2.0× elevation at a 5-day lead before cascade events relative to matched normal periods (permutation test, p < 0.0001). The elevation persists at 10-day (1.9×, p < 0.001), 15-day (1.8×, p < 0.001), and 30-day (1.4×, p < 0.01) leads. The gradual decay of the signal over longer horizons is expected: the early warning should be strongest close to the event and weaken with distance.

**Space weather.** The same metric shows a 2.0× elevation at a 5-day lead (p = 0.0002), consistent with financial markets in both magnitude and significance. However, the signal fades more rapidly, losing significance by the 15-day horizon. This faster decay is physically sensible: geomagnetic storms develop over days rather than the weeks-to-months timescale of financial cascades.

The convergence of the 2.0× elevation ratio across two physically unrelated domains at the 5-day horizon — without any parameter tuning — is notable. The I-D balance metric appears to provide a common coordinate system in which the approach to critical transitions manifests with quantitatively similar early warning signatures.

### 3.4 Test 4: Classifier Performance, Exit Velocity, and Cross-Domain Transfer

**Exit velocity as a discriminator.** Exit velocity — the rate of change of the balance metric — discriminates events from normal periods with AUC 0.692 in financial markets and AUC 0.843 in space weather (p < 1 × 10⁻¹⁷ for both). The higher AUC in space weather likely reflects the sharper onset dynamics of geomagnetic storms compared to the more diffuse build-up of financial cascades.

**Sum as a baseline.** The simple sum of all five layers (Sum = L₀ + L₁ + L₂ + L₃ + L₄) provides a strong baseline classifier: AUC 0.868 for financial cascades and AUC 0.932 for space weather storms. This is expected — during cascades, most or all layers are elevated, so their sum naturally discriminates. The relevance of this baseline will become clear in the EEG analysis (Section 3.5), where Sum fails catastrophically.

**Composite classifier.** Combining exit velocity with the 30-day variance signal yields a composite classifier with AUC 0.798 (financial) and 0.886 (space weather). The composite does not exceed Sum alone in these domains because the dominant signal during financial and space weather cascades is total magnitude — a signal that Sum captures directly.

**Cross-domain transfer.** To test whether the framework captures domain-general structure, classification thresholds learned in one domain were applied to the other without recalibration. The Sum-based threshold ratio between domains is 1.47×, and financial-to-space transfer achieves a true positive rate (TPR) of 0.78 — reasonable cross-domain performance for a zero-tuning framework. The R × S threshold ratio is larger (3.77×), reflecting the greater domain-specificity of coupling structure compared to total magnitude. For cross-domain transfer, Sum provides a more stable I coordinate than R × S.

### 3.5 Test 5: EEG — The Crucial Experiment

The EEG domain provides the most stringent test of the DET framework because, unlike financial markets and space weather, the two states being compared (Wake and N3 deep sleep) differ in their organizational *structure* without necessarily differing in total signal *magnitude*. Indeed, N3 deep sleep is characterized by high-amplitude slow-wave oscillations that produce *higher* total spectral power than wakefulness [21]. Any framework that conflates magnitude with organization will fail in this domain. The inclusion of EEG was specifically motivated by this expectation: it is a domain where Sum *should* fail and where I-D balance *should* succeed, if the framework is capturing what it claims to capture.

#### 3.5.1 Five-Subject Pilot

The pilot analysis of 5 subjects from the Sleep-EDF dataset yielded the following results:

**Balance zone occupancy.** At the ±0.5σ threshold, Wake epochs occupy the balance zone 21.4% of the time, compared to 1.1% for N3 epochs — an 18.9× ratio. At the ±1.5σ threshold, the ratio is 4.5× (Wake 60.7% vs. N3 13.4%). The N3 state is almost entirely excluded from the tight balance zone, consistent with the DET prediction that deep sleep corresponds to the low-I/high-D (fragmented) quadrant.

**Mean integration and differentiation.** Mean I (R × S): Wake 0.056, N3 0.002 — a 28× ratio. Mean D: Wake 0.556, N3 0.680. Wake has moderate I and moderate D (the critical balance quadrant), while N3 has near-zero I and elevated D (the normal/idle quadrant). The 28× ratio in I reflects the near-total collapse of cross-band coupling in N3: delta dominates while theta, alpha, beta, and gamma are suppressed, yielding minimal redundancy and near-zero synergy.

**Exit velocity.** Wake epochs show mean exit velocity of 0.558 compared to 0.133 for N3 (4.2× ratio, p = 1.7 × 10⁻¹¹⁴). The large effect size reflects the dynamic richness of wakefulness — the I-D balance point shifts continuously as attentional and cognitive states fluctuate — compared to the static, delta-dominated landscape of N3.

**The Sum anti-prediction.** This is the headline finding. The simple sum of spectral power across all five frequency bands yields an AUC of 0.288 for discriminating Wake from N3. An AUC below 0.5 means the classifier is *anti-predicting*: it systematically assigns higher scores to N3 than to Wake because N3 has higher total spectral power. Sum is not merely uninformative — it is actively misleading. By contrast, I (R × S) achieves AUC 0.869, and exit velocity achieves AUC 0.793.

This result is not a statistical fluke. It follows directly from the physiology of sleep. N3 deep sleep is dominated by high-amplitude delta oscillations (0.5–4 Hz), which are generated by thalamocortical circuits during the slow oscillation [21]. These waves have amplitudes 2–5× larger than typical waking EEG [22], producing high total power. But this power is concentrated in a single frequency band: delta accounts for >80% of total spectral power in N3. R is therefore near its minimum (one active layer), and S is near zero (synergy requires multiple coupled components). The product R × S correctly identifies N3 as a state of low integration — many neurons firing in synchrony at one frequency, with no cross-band coordination.

Wakefulness, by contrast, has moderate power distributed across multiple bands: alpha contributes during relaxed attention, beta during active processing, theta during memory encoding, and gamma during binding operations [23]. R is moderate (several active layers), and S is moderate (structured inter-band coupling). The product R × S correctly identifies wakefulness as a state of moderate integration — the critical balance state of the four-state model.

Sum conflates magnitude with structure. R × S captures the coupling structure that the I-D framework posits as fundamental. The EEG domain provides a clean dissociation between the two, confirming that the framework is measuring what it claims to measure.

#### 3.5.2 Fifty-Subject Scale-Up

The analysis was scaled to 50 subjects (57,212 epochs) to assess stability. Key results:

**Balance zone occupancy.** At the ±0.5σ threshold, Wake occupies the balance zone 17.5% of the time, compared to 0.5% for N3 — a 36.6× ratio. The larger ratio at 50 subjects (vs. 18.9× at 5 subjects) reflects both the increased Wake sample providing a more stable occupancy estimate and the greater N3 exclusion from the tight balance zone in the larger, more representative cohort. At ±1.5σ, Wake is 53.9% vs. N3 4.3% (12.5×).

**I (R × S) AUC.** 0.909, with bootstrap 95% confidence interval [0.904, 0.913]. The improvement from 0.869 (5 subjects) to 0.909 (50 subjects) indicates that the signal strengthens with additional data, consistent with a genuine population-level effect rather than subject-specific artifacts.

**Sum AUC.** 0.416 — still anti-predicting (below 0.5), consistent across sample sizes (0.288 at 5 subjects, 0.416 at 50 subjects).

**Exit velocity ratio.** 4.9× (up from 4.2× at 5 subjects), confirming the signal across a larger population.

**Table III.** EEG results summary across sample sizes.

| Metric | 5 Subjects | 30 Subjects | 50 Subjects |
|---|---|---|---|
| I (R×S) AUC | 0.869 | 0.904 | **0.909** [0.904, 0.913] |
| Sum AUC | 0.288 | 0.330 | **0.416** (anti-predicts) |
| Balance zone ratio (±0.5σ) | 18.9× | undefined (N3 = 0%) | **36.6×** |
| Exit velocity ratio | 4.2× | 5.6× | **4.9×** |

### 3.6 Robustness Across Thresholds

A legitimate concern with any threshold-based analysis is sensitivity to the specific threshold chosen. To address this, all key results were computed at four balance zone thresholds: ±0.5σ, ±0.75σ, ±1.0σ, and ±1.5σ.

**Balance zone occupancy discrimination** (events vs. normal, or Wake vs. N3) is significant at all four thresholds in all three domains. The discrimination is strongest at tight thresholds (±0.5σ) and weakens gradually at wider thresholds, as expected — a wider balance zone admits more of both event and normal states.

**Residence time discrimination** is significant at ±1.5σ in financial markets (p = 0.0001) and in space weather (p = 0.01). At narrower thresholds (±1.0σ and below), median residence times in both conditions are 1 day, yielding insufficient resolution for discrimination. The KDE-based adaptive boundary confirms the financial result (p = 0.0003) but not the space weather result (p = 0.66), suggesting that fixed wide thresholds are more robust for the space domain.

**Sum anti-prediction in EEG** is present at all thresholds tested. The AUC of Sum for Wake vs. N3 classification remains below 0.5 (ranging from 0.288 at 5 subjects to 0.416 at 50 subjects) regardless of balance zone definition, because the anti-prediction is driven by spectral power distribution rather than balance zone thresholds.

**Early warning signal** elevation is significant at all thresholds in financial markets and at ±1.0σ and ±1.5σ in space weather. The 2.0× elevation ratio at 5-day lead is stable to within ±0.2× across threshold choices.

These results confirm that the DET framework's findings are not artifacts of a particular threshold choice. The qualitative conclusions — events flee balance, residence time shortens during events, Sum anti-predicts in EEG, early warning variance rises before events — hold across a threefold range of threshold values.


### 3.6.1 Negative Controls

To confirm that the observed discriminative power reflects genuine structure rather than statistical artifacts, two permutation controls were conducted.

**Label shuffle.** Event labels were randomly permuted 1,000 times while layer values were held fixed. For financial data, the real AUC (Sum) of 0.868 dropped to a shuffle mean of 0.500 ± 0.017 (permutation p < 0.001). For space weather, the real AUC of 0.932 dropped to 0.501 ± 0.041 (p < 0.001). Exit velocity showed identical null behavior (shuffle AUC 0.500 in both domains). These results confirm that the classifiers are detecting genuine label-correlated structure, not distributional artifacts.

**Temporal layer shuffle.** The temporal ordering of each layer was independently permuted (200 iterations), breaking inter-layer coordination while preserving each layer's marginal distribution. Financial AUC degraded from 0.868 to 0.500 ± 0.016 (100% of shuffled AUCs below real). Space weather degraded from 0.932 to 0.503 ± 0.039 (100% below real). The complete collapse of discriminative power under temporal decorrelation confirms that the framework depends on the coordinated temporal structure across layers, not on the marginal statistics of individual layers.

### 3.6.2 L4 Financial Layer Ablation

To validate the simplified L4 formulation (Section 2.1), three variants were compared: (a) a conjunction gate conditioning haven activation on concurrent equity stress (AUC 0.868), (b) the simple haven-mean z-score used in this paper (AUC 0.820), and (c) L4 removed entirely (AUC 0.870). The conjunction gate offers no improvement over removal, confirming that it adds noise rather than signal. The simple z-score formulation slightly underperforms removal but maintains the five-layer architecture consistently across all three domains. The zero-tuning claim is strengthened: the most domain-engineered variant of the framework performs no better than the simplest.

### 3.6.3 Comparison with Standard Early Warning Indicators

To contextualize the I-D variance early warning signal within the existing critical transitions literature, standard critical-slowing-down (CSD) indicators -- rolling variance, AR(1) coefficient, and skewness [2] -- were computed on raw domain signals (VIX for financial, Dst for space weather) and compared against I-D balance variance as event classifiers.

**Table IV.** Early warning indicator comparison (30-day window, AUC for event classification).

| Indicator | Financial | Space Weather |
|---|---|---|
| I-D balance variance | 0.780 | 0.789 |
| Standard rolling variance | 0.826 | 0.794 |
| AR(1) coefficient | 0.606 | 0.443 |
| Skewness | 0.582 | 0.727 |

Standard rolling variance slightly outperforms I-D variance in financial markets (0.826 vs. 0.780) and performs comparably in space weather (0.794 vs. 0.789). However, the two indicators capture complementary information: at a 5-day lead, standard variance shows a 2.9x pre-event elevation in financial data while I-D variance shows 2.0x, but at longer leads (30 days), the two converge (1.4x vs. 1.4x). AR(1) and skewness perform substantially worse as standalone classifiers. The I-D framework's contribution is not that it outperforms raw variance on a single signal, but that it provides a multi-scale coordinate system in which the variance signal is interpretable across domains and substrates -- including EEG, where raw variance of a single signal cannot substitute for the structural I-D decomposition.

### 3.6.4 Null Model for I-D Classification

To distinguish genuine organizational signal from metric artifacts, three additional null models were applied to the EEG domain (50 subjects, 57,212 epochs).

**Label permutation (n = 1,000).** Sleep stage labels (Wake/N3) were randomly shuffled while I values were held fixed. The real AUC of 0.901 (computed on the full epoch set including all sleep stages; the main analysis in Section 3.5.2 reports 0.909 on the Wake/N3 subset) dropped to a permutation mean of 0.500 ± 0.008 (p < 0.001). This confirms that the classification reflects genuine state-dependent structure, not a distributional artifact of the I metric.

**Cross-band shuffle (5 rounds).** For each epoch, the assignment of frequency bands to layer indices was randomly permuted before computing R × S. The mean AUC was 0.901 — identical to the real value (Δ = 0.000). This result is expected: R × S depends on the *distribution* of power across layers, not on which physical frequency band occupies which index. The permutation invariance of R × S is a structural property of the metric, not a limitation.

**Phase randomization (5 rounds).** For each epoch, the Fourier phases of the raw EEG signal were randomized (preserving the power spectrum) before bandpass filtering and I computation. The mean AUC dropped to 0.865 ± 0.001 (Δ = 0.036). This demonstrates that approximately 3.6% of the discriminative signal arises from temporal dynamics — phase relationships and cross-band coordination — beyond what is captured by the static power spectrum alone. The remaining ~96% reflects the spectral power distribution difference between Wake and N3, which is itself a genuine physiological signal (cross-band engagement in wakefulness vs. delta dominance in deep sleep).

**I-D anticorrelation.** The Spearman correlation between I and D is −0.985 across all Wake and N3 epochs. This anticorrelation is a mathematical property of the metrics: R (a component of I) increases with the entropy of the power distribution, while D (JSD from uniform) decreases with the same entropy. Both quantities are complementary functions of the spectral power distribution. This structural relationship does not invalidate the framework — the DET hypothesis concerns where different system states *reside* in I-D space, not the correlation between I and D within that space.

### 3.6.5 Out-of-Sample Validation

To address the concern that all results may reflect in-sample overfitting, a strict temporal split was applied to the financial and space weather domains. For financial markets, the first 3,964 days (pre-2015) served as the training set and the remaining 2,821 days (2015–2024) as the held-out test set. For space weather, the split was at 2010. All z-score normalization parameters (means, standard deviations) were learned exclusively from the training set and applied unchanged to the test set.

**Table V.** Out-of-sample validation: AUC on held-out test set (z-score parameters from training set only).

| Domain | Metric | Train AUC | Test AUC | Δ |
|---|---|---|---|---|
| Financial (R×S) | Exit velocity | 0.665 | 0.634 | −0.031 |
| Financial (Sum) | Exit velocity | 0.681 | 0.712 | +0.032 |
| Space (R×S) | Exit velocity | 0.716 | 0.686 | −0.030 |
| Space (Sum) | Exit velocity | 0.887 | 0.797 | −0.090 |
| Financial (R×S) | |I−D| | 0.564 | 0.654 | +0.091 |
| Financial (Sum) | |I−D| | 0.681 | 0.792 | +0.111 |
| Space (R×S) | |I−D| | 0.568 | 0.656 | +0.088 |
| Space (Sum) | |I−D| | 0.791 | 0.838 | +0.047 |

All metrics retain discriminative power on the held-out period, with exit velocity AUC dropping by at most 0.09 (space Sum) and |I−D| AUC actually *improving* on the test set in all four cases. The improvement on |I−D| likely reflects that the post-split periods include higher-volatility regimes (2020 COVID crash, 2015–2016 market turbulence) where the balance metric separates events more cleanly. These results confirm that the DET framework does not overfit to the training period and generalizes to unseen data without recalibration.

### 3.7 Layer Count Sensitivity

To assess whether the DET framework depends critically on the five-layer architecture, we systematically varied the number of layers N ∈ {3, 4, 5, 6, 8} across all three domains. For N < 5, adjacent layers were merged by averaging their z-scored activations. For N > 5, existing layers were split into finer components from the same raw data: delta sub-bands in EEG, solar wind velocity separation in space weather, and VIX momentum or haven disaggregation in financial markets. All downstream computations (R, S, D, I, balance zone definitions) used identical formulas with only N adjusted.

**Table VI.** Layer sensitivity: I (R×S) AUC for event/state discrimination across layer counts.

| N | Financial | Space Weather | EEG (Wake vs. N3) | EEG Sum |
|---|---|---|---|---|
| 3 | 0.590 | 0.604 | 0.669 | 0.290 |
| 4 | 0.695 | 0.746 | 0.858 | 0.289 |
| 5 (standard) | 0.769 | 0.835 | 0.869 | 0.288 |
| 6 | 0.725 | 0.858 | **0.924** | 0.287 |
| 8 | 0.706 | 0.844 | **0.921** | 0.287 |

Three findings emerge. First, the framework is robust to layer count: adjacent values (N = 4 and N = 6) produce AUC within 10% of the N = 5 baseline across all domains, with space weather and EEG actually improving at N = 6. Second, performance degrades smoothly at N = 3 (insufficient hierarchical depth for meaningful integration), consistent with the theoretical requirement that integration needs multiple engaged layers. Notably, N = 3 produces the highest balance zone ratio in EEG (46×) yet the worst AUC (0.669): with only three layers, the balance zone becomes so narrow that nearly all N3 epochs are excluded, inflating the ratio while providing insufficient resolution for robust classification. Third, N = 8 offers no improvement over N = 6, suggesting diminishing returns from over-decomposition.

Critically, the Sum metric continues to anti-predict in EEG at all layer counts (AUC 0.287–0.290), confirming that the I-D framework captures structural organization rather than magnitude regardless of decomposition granularity. The early warning signal (5-day lead variance ratio) remains significant at all N values in both financial (1.32×–2.38×, all p < 10⁻¹⁵) and space weather (1.47×–2.05×, all p < 10⁻⁴) domains.

The EEG result at N = 6 (AUC 0.924) suggests that splitting delta into slow and fast sub-bands captures additional structure relevant to coherence-state discrimination, consistent with the known physiological distinction between delta-1 (0.5–2 Hz) and delta-2 (2–4 Hz) oscillations [25]. However, we retain the five-layer architecture as the standard to maintain the zero-tuning claim — selecting N = 6 specifically for EEG would constitute domain-specific optimization.


---

## 4. Discussion

### 4.1 The I-D Balance as a Dynamic Existence Threshold

The results across three domains support a consistent interpretation: the I-D balance zone constitutes a dynamic existence threshold for organized system states. Systems in the balance zone — where integration and differentiation are jointly sustained — exhibit the hallmarks of healthy function: financial markets operate normally, the magnetosphere is in its quiet-time configuration, and the brain is awake and processing information. Departures from the balance zone correspond to qualitatively different states: cascading failures, geomagnetic storms, and the unconsciousness of deep sleep.

The four-state model (Table I) receives empirical support from the directional dynamics analysis (Test 1). The CASCADE pattern (rising I, falling D) dominates during event periods, accounting for 45–51% of event time steps. This pattern maps directly to the pathological synchronization observed in financial contagion [11], geomagnetic storm development, and epileptic seizure propagation [26]. The near-absence of the COINCIDENCE pattern (<2% during events) suggests a fundamental budget constraint: systems cannot simultaneously increase both integration and differentiation during critical periods.

A candidate mechanistic account follows from the general principle that coordination (synergy) is more fragile than overlap (redundancy) under system perturbation. As a system approaches a critical transition, the effective dimensionality of information transfer decreases. This compression preferentially destroys synergy (the S component of I = R × S), such that integration collapses while redundancy (and thus differentiation) may remain elevated. Under this account, the system moves from the balance zone into the cascade quadrant not by a symmetric process but by the asymmetric fragility of synergy — a pattern consistent with both information-theoretic expectations and the directional dynamics data presented here.

A natural concern is whether the strong I-D anticorrelation (Spearman ρ = −0.985 in EEG; Section 3.6.4) renders the two-dimensional framing redundant. It does not, for two reasons. First, the anticorrelation strength itself varies across system states — it is tighter during N3 (where delta dominance constrains both metrics) than during wakefulness (where multi-band engagement permits greater I-D independence) — and this state-dependent coupling structure is invisible to any one-dimensional projection. Second, the I-D plane preserves the *directionality* of transitions: the CASCADE pattern (rising I, falling D) is mechanistically distinct from COLLAPSE (both falling), yet both would project identically onto a single principal component. The four-state geometry thus adds interpretive value beyond what a scalar diversity index can provide, even when the occupied region of the plane is narrow.

### 4.2 Critical Slowing Down in I-D Coordinates

The early warning signal (Test 3) — rising variance of |I − D| before events — connects the DET framework directly to the well-established critical transitions literature [1, 2]. Critical slowing down predicts that as a system approaches a bifurcation point, its recovery rate from perturbations decreases, leading to increasing variance and autocorrelation in state variables. The I-D balance metric provides a natural coordinate in which to measure this phenomenon: the system's ability to maintain balance degrades as it approaches the transition, producing larger and larger fluctuations in |I − D|.

The quantitative consistency across domains — a 2.0× variance elevation at 5-day lead in both financial and space weather data — is noteworthy. While the temporal scales of the underlying processes differ (financial cascades build over weeks, geomagnetic storms over days), the early warning signal has similar magnitude at similar lead times. This suggests that the I-D coordinate system may capture a domain-general feature of approaching transitions, though the author cautions against over-interpreting numerical coincidences from two domains.

The decay profile of the early warning signal differs between domains in a physically sensible way. Financial markets show persistent elevation out to 30 days, consistent with the slow build-up of systemic stress through leverage accumulation and correlation tightening. Space weather shows rapid decay by 15 days, consistent with the faster dynamics of solar wind-magnetosphere coupling. An I-D framework that showed identical decay profiles across domains would be suspicious; the observed domain-appropriate decay profiles increase confidence that the framework is capturing genuine physics rather than statistical artifacts.

### 4.3 Why Sum Fails: Structure Versus Magnitude

The EEG results provide the strongest evidence for the DET framework's construct validity. The Sum anti-prediction (AUC 0.288–0.416) is not merely a failure — it is an *informative* failure that illuminates what the framework measures.

Consider two hypothetical explanations for why I-D balance discriminates system states:

1. **Magnitude hypothesis:** Events and state changes are simply associated with higher (or lower) total activity, and the I-D framework captures this through its integration metric.
2. **Structure hypothesis:** Events and state changes are associated with changes in the *organizational structure* of multi-component coupling, which is partially but imperfectly correlated with total activity.

In financial markets and space weather, these hypotheses make similar predictions: cascades involve elevated activity across most layers, so both Sum and I should discriminate. The data are consistent with both hypotheses in these domains (Sum AUC 0.868–0.932, I-based AUC 0.692–0.843).

The EEG domain dissociates the two hypotheses. N3 deep sleep has *higher* total spectral power than wakefulness (massive delta oscillations), so the magnitude hypothesis predicts that Sum should classify N3 as the "event" state — which is indeed what happens, producing the anti-prediction. The structure hypothesis predicts that I (R × S) should correctly identify wakefulness as the state with higher integration, because wake involves coordinated multi-band activity while N3 involves single-band dominance — which is also what happens (AUC 0.869–0.904).

The dissociation is clean, the effect is large, and it scales with sample size. This constitutes strong evidence that the I-D framework captures organizational structure rather than mere magnitude, validating the construct at the heart of the Dynamic Existence Threshold.

A natural question is whether domain-specific spectral ratios — particularly the delta/beta power ratio, a standard sleep staging feature — outperform R × S on the EEG task. Head-to-head comparison on the same 50-subject cohort yields delta/beta AUC = 0.912 vs. R × S AUC = 0.901 (bootstrap 95% CI for the difference: [−0.013, −0.008], p < 0.05). The two metrics are highly correlated (Spearman ρ = 0.923), confirming that R × S partially recapitulates spectral power distribution information. However, on harder discrimination tasks — N1 vs. N2 (R × S AUC 0.740 vs. delta/beta 0.685) and Wake vs. N2 (0.532 vs. 0.502) — R × S outperforms the spectral ratio, suggesting that the multi-band integration measure captures organizational structure that a two-band ratio misses. Crucially, the delta/beta ratio is undefined outside neuroscience: financial markets and space weather have no frequency bands. The DET framework's contribution is not that R × S is the optimal classifier for any single domain, but that it provides a substrate-independent coordinate system in which state transitions are interpretable across all three domains tested — including the one where a domain-specific shortcut exists.


### 4.4 Cross-Domain Consistency and the Limits of Broad Generality

The results demonstrate broad generality of the DET framework across three substrates (socioeconomic, geophysical, biological) without parameter tuning. However, the author is careful to distinguish between two claims:

1. **Narrow claim (supported):** The I-D balance framework provides a useful coordinate system for characterizing critical transitions across multiple domains, and its predictions are confirmed in all three domains tested.
2. **Strong claim (not yet supported):** The I-D balance framework captures a universal principle that applies to all complex systems undergoing critical transitions.

The narrow claim is well-supported by the data. The strong claim would require testing across a much wider range of systems — ecological tipping points, cellular state transitions, social network cascades, engineered system failures — and would need to account for systems where the five-layer decomposition may not have a natural interpretation.

The cross-domain transfer results (Test 4) illustrate both the promise and the limitations of broad generality. Sum-based thresholds transfer reasonably between financial and space weather domains (1.47× ratio, 0.78 TPR), but R × S thresholds diverge more (3.77× ratio). This suggests that while the *qualitative* structure of I-D space is domain-general, the *quantitative* calibration of integration metrics is domain-specific. The z-score normalization of the balance metric partially addresses this by converting domain-specific scales to comparable units, but it does not fully eliminate the issue.

### 4.5 Relation to Existing Frameworks

The DET framework intersects with several established theoretical traditions.

**Integrated Information Theory [3].** IIT's Φ quantifies integrated information as the information generated by a system above and beyond its parts. The DET framework draws inspiration from IIT's emphasis on integration but differs in two ways: (a) it adds an explicit differentiation coordinate, creating a two-dimensional state space rather than a single scalar; and (b) it uses empirically tractable proxy metrics (R × S for integration, JSD for differentiation) rather than computing Φ directly, which remains intractable for systems of even moderate size. The four-state model can be seen as a coarse-graining of the I-D plane that may correspond heuristically to regimes of high and low Φ, though R × S is a computationally tractable proxy rather than a formal information-theoretic measure of integrated information.

**Critical transitions theory [1, 27].** The DET framework provides a specific coordinate system (I-D space) and a specific indicator (variance of B = |z(I) − z(D)|) for the general phenomenon of critical slowing down. This is complementary to existing approaches that use variance and autocorrelation of raw state variables [2]. The I-D coordinate system has the advantage of being explicitly multi-scale (through the five-layer architecture) and of separating structural organization from magnitude.

**Perturbational complexity [24, 28].** The perturbational complexity index (PCI) measures the information content of the brain's response to transcranial magnetic stimulation, and has been shown to discriminate conscious from unconscious states. PCI is high when the response is both integrated (spatially widespread) and differentiated (temporally complex) — precisely the conditions that define the balance zone in the DET framework. The EEG results reported here are consistent with PCI findings, achieved through passive observation rather than perturbation.

**Self-organized criticality (SOC) [29, 30, 27].** The balance zone may correspond to the critical state in SOC models — the state to which the system self-organizes and from which avalanches (events) depart. The residence time results (Test 2) are consistent with this interpretation: the system spends more time in the balance zone during normal periods, as expected if the balance zone is an attractor, and exits rapidly during events, as expected if events are excursions from the critical state.

**Partial information decomposition (PID) [31, 32, 33].** The R and S metrics are inspired by but not identical to the redundancy and synergy of PID. PID provides exact decompositions of mutual information into redundant, synergistic, and unique components; R and S are empirical proxies designed for tractability across domains. A formal connection between the DET framework's R × S and PID's synergy would strengthen the theoretical foundations of this work.

### 4.6 Limitations

Several limitations warrant explicit acknowledgment.

**Five-layer architecture.** The choice of exactly five layers is pragmatic rather than principled. However, the layer sensitivity analysis (Section 3.7, Table VI) demonstrates that the framework is robust to this choice: adjacent layer counts (N = 4 and N = 6) produce AUC within 10% of the N = 5 baseline across all three domains, with N = 6 actually improving performance in EEG (0.924 vs. 0.869) and space weather (0.858 vs 0.835). The five-layer count is not a local optimum but a pragmatic choice within a broad robustness plateau spanning N = 4 to N = 8. Performance degrades only at N = 3, where insufficient hierarchical depth limits integration measurement.

**Proxy metrics.** R × S and JSD are proxies for integration and differentiation, not exact information-theoretic quantities. Different proxy choices might yield different results. The author chose these metrics for their tractability and interpretability, but acknowledges that the framework's conclusions are conditional on these specific operationalizations.

**Temporal resolution.** Financial and space weather data are analyzed at daily resolution, while EEG is analyzed at 30-second epoch resolution. The framework has not been tested at intermediate timescales or at the sub-second resolution relevant to neural dynamics. The balance zone dynamics might differ qualitatively at different temporal scales.

**Event definition.** The definition of "events" in each domain relies on established criteria (VIX thresholds, Dst thresholds, sleep stage scoring), but these criteria are themselves imperfect. The framework's performance could be sensitive to event definition, particularly for boundary cases.

**Sample size in EEG.** The current results are based on 50 subjects (57,212 epochs). While the signal strengthens monotonically from 5 to 50 subjects (Table III), the sample remains modest by neuroimaging standards. Replication in larger and more diverse cohorts is needed.

**Causal ambiguity.** The framework demonstrates that I-D balance *discriminates* system states and that its variance provides *early warning* of transitions, but it does not establish that I-D imbalance *causes* transitions. The relationship could be correlational, with both I-D dynamics and transitions driven by a common underlying process.

**Domain selection.** Three domains, while spanning physical, biological, and socioeconomic systems, constitute a limited sample. Domains were selected in part because they have well-defined "event" periods and accessible multi-layer data — a convenience sample that may not be representative of complex systems in general.

---

## 5. Conclusion

This paper introduced the Dynamic Existence Threshold — the hypothesis that systems persist in the zone where integration and differentiation are jointly sustained, and that critical events correspond to departures from this zone. Five quantitative tests, validated by permutation-based negative controls and comparison with standard early warning indicators, across three domains (financial markets, space weather, and human EEG) support this hypothesis:

1. Events cluster at I-D extremes while normal states occupy the balance center, with the CASCADE pattern (rising I, falling D) accounting for 45–51% of event dynamics.
2. Events exit the balance zone faster than normal state changes, with residence times halved during event periods (p = 0.0001 in financial data).
3. Rising variance of the balance metric B = |z(I) − z(D)| provides early warning 5–30 days before events, with a 2.0× elevation ratio observed independently in both financial and space weather domains.
4. Exit velocity discriminates events from normal states with AUC 0.692–0.843, and cross-domain threshold transfer achieves 0.78 TPR without recalibration.
5. In EEG, the coupling structure metric R × S achieves AUC 0.909 (95% CI [0.904, 0.913]) for discriminating wakefulness from deep sleep, while the simple sum anti-predicts (AUC 0.288–0.416), confirming that the framework captures organizational structure rather than magnitude.

These results extend integrated information theory and critical transitions research into a dynamic framework that characterizes not just whether a system can persist as an organized entity, but the conditions under which it transitions between qualitatively different states. The I-D balance zone emerges as a broadly general feature of complex systems, providing a common coordinate system for understanding critical transitions across substrates.

The practical implication is a domain-general early warning indicator: rising variance of the balance metric B, computed from any system that can be decomposed into multiple measurement layers. The theoretical implication is that integration and differentiation are not merely descriptive properties of complex systems but complementary quantities whose balance is associated with the system's dynamic existence threshold.

---

## References

1. M. Scheffer, J. Bascompte, W. A. Brock, V. Brovkin, S. R. Carpenter, V. Dakos, H. Held, E. H. van Nes, M. Rietkerk, and G. Sugihara, "Early-warning signals for critical transitions," Nature 461(7260), 53–59 (2009).

2. V. Dakos, S. R. Carpenter, W. A. Brock, A. M. Ellison, V. Guttal, A. R. Ives, S. Kéfi, V. Livina, D. A. Seekell, E. H. van Nes, and M. Scheffer, "Methods for detecting early warnings of critical transitions in time series illustrated using simulated ecological data," PLoS ONE 7(7), e41010 (2012).

3. G. Tononi, M. Boly, M. Massimini, and C. Koch, "Integrated information theory: From consciousness to its physical substrate," Nat. Rev. Neurosci. 17(7), 450–461 (2016).

4. G. Tononi, "An information integration theory of consciousness," BMC Neurosci. 5, 42 (2004).

5. B. J. Baars, *A Cognitive Theory of Consciousness* (Cambridge University Press, 1988).

6. A. K. Seth, A. B. Barrett, and L. Barnett, "Causal density and integrated information as measures of conscious level," Philos. Trans. R. Soc. A 369(1952), 3748–3767 (2011).

7. R. B. Berry, R. Brooks, C. E. Gamaldo, S. M. Harding, R. M. Lloyd, C. L. Marcus, and B. V. Vaughn, "The AASM manual for the scoring of sleep and associated events: Rules, terminology and technical specifications, version 2.4," American Academy of Sleep Medicine (2017).

8. A. V. Oppenheim and A. S. Willsky, *Signals and Systems*, 2nd ed. (Prentice Hall, 1997).

9. S. D. Muthukumaraswamy, "High-frequency brain activity and muscle artifacts in MEG/EEG: A review and recommendations," Front. Hum. Neurosci. 7, 138 (2013).

10. N. E. Papitashvili and J. H. King, "OMNI hourly data set," NASA Space Physics Data Facility (2020). https://omniweb.gsfc.nasa.gov/

11. K. J. Forbes and R. Rigobon, "No contagion, only interdependence: Measuring stock market comovements," J. Finance 57(5), 2223–2261 (2002).

12. M. O. Hill, "Diversity and evenness: A unifying notation and its consequences," Ecology 54(2), 427–432 (1973).

13. L. Jost, "Entropy and diversity," Oikos 113(2), 363–375 (2006).

14. E. C. Pielou, "The measurement of diversity in different types of biological collections," J. Theor. Biol. 13, 131–144 (1966).

15. J. Lin, "Divergence measures based on the Shannon entropy," IEEE Trans. Inf. Theory 37(1), 145–151 (1991).

16. D. M. Endres and J. E. Schindelin, "A new metric for probability distributions," IEEE Trans. Inf. Theory 49(7), 1858–1860 (2003).

17. W. D. Gonzalez, J. A. Joselyn, Y. Kamide, H. W. Kroehl, G. Rostoker, B. T. Tsurutani, and V. M. Vasyliunas, "What is a geomagnetic storm?" J. Geophys. Res. 99(A4), 5771–5792 (1994).

18. B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberyé, "Analysis of a sleep-dependent neuronal feedback loop: The sleep-EDF database," IEEE Trans. Biomed. Eng. 47(9), 1185–1194 (2000).

19. A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. Ch. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," Circulation 101(23), e215–e220 (2000).

20. Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: A practical and powerful approach to multiple testing," J. R. Stat. Soc. Ser. B 57(1), 289–300 (1995).

21. M. Massimini, F. Ferrarelli, R. Huber, S. K. Esser, H. Singh, and G. Tononi, "Breakdown of cortical effective connectivity during sleep," Science 309(5744), 2228–2232 (2005).

22. E. Niedermeyer and F. L. da Silva, *Electroencephalography: Basic Principles, Clinical Applications, and Related Fields*, 5th ed. (Lippincott Williams & Wilkins, 2005).

23. G. Buzsáki, *Rhythms of the Brain* (Oxford University Press, 2006).

24. A. G. Casali, O. Gosseries, M. Rosanova, M. Boly, S. Sarasso, K. R. Casali, S. Casarotto, M.-A. Bruno, S. Laureys, G. Tononi, and M. Massimini, "A theoretically based index of consciousness independent of sensory processing and behavior," Sci. Transl. Med. 5(198), 198ra105 (2013).

25. F. Amzica and M. Steriade, "Electrophysiological correlates of sleep delta waves," Electroencephalogr. Clin. Neurophysiol. 107(2), 69–83 (1998).

26. K. Lehnertz, S. Bialonski, M.-T. Horstmann, D. Krug, A. Rothkegel, M. Staniek, and T. Wagner, "Synchronization phenomena in human epileptic brain networks," J. Neurosci. Methods 183(1), 42–48 (2009).

27. S. H. Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed. (Westview Press, 2015).

28. S. Sarasso, M. Boly, M. Napolitani, O. Gosseries, V. Charland-Verville, S. Casarotto, M. Rosanova, A. G. Casali, J.-F. Brichant, P. Boveroux, S. Rex, G. Tononi, S. Laureys, and M. Massimini, "Consciousness and complexity during unresponsiveness induced by propofol, xenon, and ketamine," Curr. Biol. 25(23), 3099–3105 (2015).

29. P. Bak, C. Tang, and K. Wiesenfeld, "Self-organized criticality: An explanation of 1/f noise," Phys. Rev. Lett. 59(4), 381–384 (1987).

30. Z. Olami, H. J. S. Feder, and K. Christensen, "Self-organized criticality in a continuous, nonconservative cellular automaton modeling earthquakes," Phys. Rev. Lett. 68(8), 1244–1247 (1992).

31. P. L. Williams and R. D. Beer, "Nonnegative decomposition of multivariate information," arXiv:1004.2515 (2010).

32. M. Aguilera, S. A. Moosavi, and H. Shimazaki, "Information-theoretic complexity and the structure of large-scale neural systems," arXiv:1906.04035 (2019).

33. N. J. Popiel, S. Bhatt, M. Engström, K. Lindgren, and J. T. Lizier, "Partial information decomposition and the emergence of redundancy in complex systems," Phys. Rev. E 102(3), 032110 (2020).

---

## Supplementary Material

See supplementary material for complete analysis code, pre-computed results, and publication figures at https://github.com/existencethreshold/dynamic-existence-threshold.

## Acknowledgments

The framework emerged from practical work on distributed system monitoring, which motivated the integration-differentiation decomposition.

## Author Declarations

### Conflict of Interest

The author has no conflicts to disclose.

### Ethics Approval

This study analyzed publicly available, de-identified datasets. The EEG data are from the Sleep-EDF dataset hosted on PhysioNet; ethical approval for the original data collection was obtained by the dataset creators (Kemp et al., 2000).

### Author Contributions

**Nathan M. Thornhill:** Conceptualization (lead); Data curation (lead); Formal analysis (lead); Investigation (lead); Methodology (lead); Software (lead); Validation (lead); Visualization (lead); Writing – original draft (lead); Writing – review & editing (lead).

## Data Availability Statement

The data that support the findings of this study are openly available. Financial data are derived from publicly available market indices via the yfinance Python library. Space weather data are from NASA OMNI2 (https://omniweb.gsfc.nasa.gov/). EEG data are from the Sleep-EDF dataset on PhysioNet (https://physionet.org/content/sleep-edfx/). Complete analysis code and pre-computed results are available at https://github.com/existencethreshold/dynamic-existence-threshold.

---

*Correspondence: Nathan M. Thornhill, Independent Researcher, Fort Wayne, IN, USA. info@3riverswebtech.com*

*AI tools acknowledgment: AI tools (Claude, Gemini) were used to assist with code generation, statistical analysis, and manuscript revision. All scientific claims, interpretations, and editorial decisions are the sole responsibility of the author.*