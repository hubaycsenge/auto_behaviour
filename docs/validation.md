# Validating before you trust it

ABC produces suggestions. This page is about finding out how good they are on
*your* footage, because the published numbers are not about your footage and
are less encouraging than they first look.

## What the literature actually reports

- Zero-shot video-LLMs scoring animal behaviour reach **fair** agreement with an
  expert human coder: Cohen's κ ≈ 0.43, against human–human κ ≈ 0.69–0.83. Only
  the largest models were usable at all; small ones were "practically useless".
- Temporal localisation is coarse. Even large models score ~50 mIoU on
  Charades-STA, and independent reproductions of 7B-class results come in far
  below the reported figures. **Treat every predicted timestamp as approximate.**
- Video models hallucinate actions, mis-order events and confuse scene
  transitions — this has its own benchmarks (VIDHALLUC, VidHal).
- Multimodal LLMs scoring affect show systematic demographic bias, and field
  performance lags lab benchmarks badly: r = 0.456 against human ratings on real
  recordings from models that looked near-human in the lab.
- Off-the-shelf VLMs asked about dog emotion rely on "superficial correlations,
  such as background context and breed morphology", dropping to near chance on
  cropped faces.
- No published work applies zero-shot video-LLM ethogram coding to animal–robot
  or human–robot interaction. There is no prior estimate of how well this works
  in your setting — which is a reason to measure it, and an opportunity.

## The measurement

**1. Build a gold set.** Have ≥ 2 people code a representative subset by hand in
BORIS — varying dogs, lighting and camera angles, not just the clean sessions.
Report inter-rater reliability (Cohen's or Fleiss' κ; ICC for continuous
scores). This is your ceiling: ABC cannot be expected to beat human–human
agreement.

**2. Score ABC against human consensus**, not against one coder.

- *Point events*: a hit if it falls within a tolerance window (±1 s is a
  reasonable start) of a human-coded event of the same behaviour and subject.
  Report precision, recall and F1 per behaviour.
- *State events*: a hit at temporal IoU ≥ 0.5. Report mIoU for boundaries
  separately from detection F1 — a system can find every state and still place
  its edges badly, and those two failures need different fixes.
- Report **per behaviour**, never only in aggregate. A mean F1 hides the fact
  that the common behaviours carry it while the rare ones — usually the
  interesting ones — are missed entirely.

**3. Check for bias.** Break the error down by dog, breed, coat colour, session
and camera position. The affect-scoring literature found consistent
demographic bias in models that looked unbiased in aggregate; the equivalent
here is a system that codes dark-coated dogs or one camera angle badly.

**4. Decide from the numbers.**

| result | what to do |
|---|---|
| κ ≥ 0.7 against your coding | reasonable as a first pass for a human to correct |
| F1 ≥ 0.8 and κ ≥ 0.6 | trust for batch coding *with spot checks* |
| below that | keep it as an assistant; lean on the pose and audio engines, which measure rather than interpret, and on few-shot examples in `extra_instructions` |

## What ABC gives you to work with

**Confidence on every event.** Sweep the *Minimum confidence* setting and plot
precision against recall on your gold set; that curve tells you where to set it
for your study. Do not assume the default is right for you.

**Provenance in the BORIS comment.** Every event carries `[engine confidence]`
and a one-line justification, so a coder reviewing in BORIS can see which method
claimed it and why — and you can score engines separately from the same project.

**Self-consistency.** Set `self_consistency_samples` to 3 on the vLLM engine:
it samples repeatedly and keeps only events a majority of samples agree on,
with the confidence tempered by how often the event actually reproduced. It
costs 3× the compute and is the cheapest defence against hallucination that
does not need a second model.

**A second model.** Run `vlm_vllm` on the A100 and `vlm_llamacpp` with a
different model on the Turing nodes, leave the behaviours unowned, and compare.
Events both find are considerably more trustworthy than events either finds
alone.

**Engines that cannot hallucinate.** The pose engine's output is geometry. If a
behaviour can be defined geometrically — and the reference ethogram defines four
of them that way, in so many words — prefer it, and use the VLM to check it
rather than the other way round.

## Reporting

For a paper, report: the ethogram; the models and their exact versions; frame
sampling rate, window length and resolution; the prompt; decoding settings
(temperature, guided decoding); human–human κ on the gold set; ABC-vs-human κ,
precision, recall and F1 per behaviour at a stated IoU threshold; mIoU for
state boundaries; and per-subgroup error.

The ABC version, job ID, engine list and every engine option are written into
the BORIS project description and into `job.json`, so the run is reproducible
from the artefacts themselves.
