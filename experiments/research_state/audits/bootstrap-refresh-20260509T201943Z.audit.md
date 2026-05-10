# CounterFact v4 Baseline Audit: bootstrap-refresh-20260509T201943Z

## Scope

This audit covers the current provisional bootstrap champion on the prompt-disjoint
96-case CounterFact v4 standard substrate.

- Dataset source: `azhx/counterfact`
- Train generator: `cl_seq_train_v4_counterfact_standard`
- Visible pack: `cl_dev_visible_v4_counterfact_standard`
- Locked pack: `cl_confirm_locked_v4_counterfact_standard`
- Method: `baseline_seq_lora_ft_v4_rank16_final_consolidation`

Prompt text is intentionally omitted from the structured audit outputs so locked
confirmation surfaces are not copied into repo artifacts.

## Aggregate Failure Shape

Visible-dev:

- target exact: 0.5000
- anchor exact: 0.3507
- joint exact: 0.1944

Locked confirmation:

- target exact: 0.4271
- anchor exact: 0.5382
- joint exact: 0.2604

The run is not simply failing to learn every edit. It learns a substantial subset
of target edits, but the target/anchor objective is not jointly stable. The
strongest symptom is answer-mode bleed: repeated edited answers appear as common
outputs on unrelated same-relation prompts.

## Case-Level Observations

Locked cases where both target and anchor exact are zero include:

- `1000` / `P264` / `Purple Hearts`: target `Coral`, old `Sunshine`
- `108` / `P17` / `Bastille`: target `Canada`, old `France`
- `117` / `P495` / `RSM-56 Bulava`: target `Japan`, old `Russia`
- `141` / `P407` / `Du gamla, du fria`: target `Finnish`, old `Swedish`
- `23` / `P20` / `Charles Alfred Pillsbury`: target `Berlin`, old `Minneapolis`
- `28` / `P30` / `Pidgeon Island`: target `Asia`, old `Antarctica`
- `31` / `P495` / `controller.controller`: target `Singapore`, old `Canada`
- `33` / `P159` / `Majorette`: target `London`, old `Lyon`
- `46` / `P20` / `Maso da San Friano`: target `Vienna`, old `Florence`
- `47` / `P1412` / `Jean-Baptiste Marchand`: target `German`, old `French`
- `50` / `P740` / `Webley & Scott`: target `Wales`, old `Birmingham`
- `59` / `P19` / `Eli Maor`: target `Portsmouth`, old `Israel`
- `60` / `P27` / `Carol Zhao`: target `Japan`, old `Canada`
- `61` / `P19` / `Henry Mackenzie`: target `Honolulu`, old `Edinburgh`
- `8` / `P190` / `Wellington`: target `Sheffield`, old `Sydney`
- `91` / `P937` / `LeRoy Collins`: target `Moscow`, old `Florida`

Locked cases where both target and anchor exact are perfect include:

- `100` / `P140` / `Jahangir`: target `Judaism`, old `Islam`
- `101` / `P136` / `Frank Mantooth`: target `trance`, old `jazz`
- `102` / `P176` / `Renault 8`: target `Fiat`, old `Renault`
- `104` / `P17` / `Hohenlohe-Langenburg`: target `Italy`, old `Germany`
- `167` / `P108` / `William Rees-Mogg`: target `CBS`, old `BBC`
- `26` / `P106` / `Billy Roche`: target `architect`, old `actor`
- `32` / `P136` / `Sylvano Bussotti`: target `jazz`, old `opera`
- `34` / `P937` / `Laurent Cars`: target `Philadelphia`, old `Paris`
- `53` / `P17` / `Pochepsky District`: target `India`, old `Russia`
- `65` / `P937` / `Mayer Carl von Rothschild`: target `London`, old `Frankfurt`
- `73` / `P101` / `Lee Alvin DuBridge`: target `diplomat`, old `physics`
- `76` / `P937` / `Nancy Astor, Viscountess Astor`: target `Paris`, old `London`
- `81` / `P449` / `Running Mates`: target `CBS`, old `TNT`
- `93` / `P17` / `Wanne-Eickel Central Station`: target `Switzerland`, old `Germany`
- `97` / `P159` / `Ipsos MORI`: target `Oslo`, old `London`

This split matters: the baseline is not uniformly under-capable. It has enough
capacity to memorize and generalize some edits while preserving related anchors,
but the sequential training schedule produces uneven relation-level interference.

## Relation-Level Weaknesses

Worst locked relation slices by combined target/anchor behavior:

- `P264`: target 0.0000, anchor 0.0000
- `P407`: target 0.0000, anchor 0.0000
- `P495`: target 0.0000, anchor 0.1111
- `P740`: target 0.0000, anchor 0.3333
- `P19`: target 0.2500, anchor 0.1667
- `P30`: target 0.0000, anchor 0.4167
- `P190`: target 0.0000, anchor 0.4444
- `P27`: target 0.3333, anchor 0.2222
- `P20`: target 0.5000, anchor 0.0833

The failures concentrate in geography/location/nationality-style relations and a
few sparse relation IDs. These are exactly the relations where edited answers are
often reusable short labels, making answer-mode bleed more likely.

## Prediction-Mode Evidence

Top locked target predictions:

- `paris`: 63
- `cbs`: 18
- `london`: 15
- `switzerland`: 12
- `russian`: 12
- `fiat`: 12
- `germany`: 9
- `architect`: 9
- `italian`: 9
- `milan`: 9

Top locked anchor predictions:

- `london`: 25
- `switzerland`: 21
- `paris`: 18
- `french`: 12
- `spain`: 9
- `warsaw`: 9
- `istanbul`: 9
- `islam`: 8
- `frankfurt`: 8
- `actor`: 7

Confusions:

- target predicted the old fact on 6 locked examples
- anchor predicted the edit target on 13 locked examples

The second number is the important one. The anchor side is often not reverting
to the pre-edit base model; it is being overwritten by the same edited answer
space the method is trying to store.

## Diagnosis

The current v4 method is a useful but weak champion because it uses a
recency-ordered sequence of single-example optimizer updates. The final
consolidation pass fixed one side at a time: anchor-heavy ordering rescued
anchors but collapsed targets, while target-last ordering rescued target quality
but left relation-level anchor bleed.

The disappointing metric is therefore not evidence that CounterFact is an
unsuitable substrate. It is evidence that the baseline objective is optimized in
a way that lets update order dominate joint retention.

## Non-Cheating Next Baseline

The v5 branch keeps the same selected stack and the same prompt-disjoint
CounterFact substrate. It changes only the optimizer schedule:

- rank-32 LoRA on the same top4 selected surface
- no retrieval
- no answer-set postprocessor
- no visible-dev or locked-prompt rehearsal
- mixed target/anchor gradient accumulation during episode replay
- mixed final consolidation over active training records only

Expected signature if the diagnosis is right:

- target exact should improve without ending on target-only recency
- anchor exact should stop collapsing to recently edited answers
- joint exact should rise more than either marginal metric alone

