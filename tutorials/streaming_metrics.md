# Streaming metrics: what they mean and where they mislead

Three numbers describe a streaming TTS deployment: how much audio the GPU
produces, how long a user waits before hearing anything, and whether what they
hear plays without gaps. All three come from one recording -- the arrival time
and byte size of every chunk of every request -- and all three are computed in
`thetalker/metrics.py`.

## Time to first audio, twice

**First byte** is the interval from the request being sent to the first audio
byte arriving. It is the figure vendors publish, and it is a property of the
server: how quickly the pipeline emits its first chunk.

**Audible** adds the leading silence the model itself generates. Many TTS models
open an utterance with a short pause; a stream that delivers its first byte in
80 ms and then plays 300 ms of silence does not start speaking in 80 ms. The
audible number is what a listener experiences.

The detector is deliberately conservative. Leading silence is the offset to the
first 5 ms window whose RMS reaches 5% of that file's own peak and stays there for
20 ms. It can only ever move the reported latency later, never earlier: an empty
buffer, a buffer that opens with sound, and a buffer that never crosses the
threshold all read zero. The three constants are this repository's definition of
onset, not a standard, and a published "audible TTFA" from another stack is not
known to share them.

Connect cost is measured separately (`t_connect_ms`) and excluded from both
numbers, so a run over a fresh connection stays comparable to one over a warm
pool.

## Continuity as a real-time player

Throughput and latency together still do not say whether audio played. A stream
can deliver its first chunk in 80 ms, then go quiet for a second while the next
chunk is produced. The user hears 80 ms of speech, a gap, and then the rest.

Continuity models exactly that listener. A player is fed from the first chunk
with no pre-roll: the playback clock starts when chunk 1 arrives, and from then
on the player consumes audio in real time. At every later chunk arrival, compare
the audio delivered so far against the time elapsed; if the player has consumed
more than it has been given, it ran dry. The worst such deficit in a request is
its `buffer_deficit_s`, and the request is continuous when that stays within the
budget, 0.1 s by default.

This is a verbatim port of vllm-omni's own
`vllm_omni/benchmarks/audio_continuity.py::compute_continuity_stats`, so a
continuity percentage here means the same thing as one published upstream. Each
record also carries the verdict at 0.05 s and 0.2 s, because the choice of budget
is a judgment call and the result should be visible at more than one of them.

**The consequence for chunk sizing.** The first chunk is the player's entire
buffer until the second one arrives. If the first chunk is one codec frame, 80 ms
of audio, and the next chunk takes about a second to produce, the stream stalls
no matter how fast the GPU is. A 16-frame first chunk carries 1.3 s and covers
that gap. The production gap itself does not move -- the chunk boundaries shift
together -- so the cost is first-byte latency only.

**The playback models are not interchangeable.** Continuity starts its clock at
the first byte. Audible TTFA describes a listener who hears nothing until the
onset, and would therefore start the clock later and be harder to starve. The
0.1 s test is not invariant to that difference, so a continuity verdict is only
meaningful together with the model it was taken under. Do not mix them.

## Why a closed loop overstates capacity

The benchmark's default load is closed loop: `cN` keeps exactly N requests in
flight, issuing a new one as each finishes. That is the right shape for comparing
two configurations, because both sides see an identical offered load and neither
can win by being slower.

It is the wrong shape for capacity planning. In a closed loop, admission is
throttled by completion: the number in flight can never exceed N, so the queue
never grows, latency stays bounded and the measured RTFx is the server's
throughput *at exactly N in flight*. Real users do not wait for each other. They
arrive independently, bursts push the in-flight count above N, and both latency
and continuity degrade in the region a closed loop never visits. Reading "c32
sustains 106 RTFx" as "this GPU serves 32 concurrent users" overstates the
result: it is what the GPU delivers when it is fed exactly 32 at a time.

The same argument applies to continuity. A continuity percentage at c32 is
measured under a schedule that never exceeds 32 in flight; under Poisson arrivals
at the same mean load, some of the time there are more.

For an open-loop measurement, drive the client with `--arrival-rate <req/s>`
instead of `--concurrency`. Arrival times are then drawn from a Poisson process
with the given rate and the seed, and the server is free to fall behind. That is
the measurement to make before quoting a user count; the tables in this
repository do not make it.

## Aggregation

RTFx is aggregated over the point, not averaged over requests: total audio
seconds produced divided by the wall-clock duration of the point. That is the
number that scales with GPU cost. Per-request delivery rates are in
`records.jsonl` as `delivery_rtfx_audio_over_wall` if you want the distribution.

Latencies are reported as p50 and p95, never as a mean: the distribution under
load has a long tail, and a mean hides exactly the requests a user complains
about.

Warmup requests are issued into the same run and excluded from every statistic.
The count matters: a codec that captures CUDA graphs on first use needs enough
warmup to have captured the batch shapes the scored requests will hit, which is
why the protocol warms 8 requests at c8 and 100 at c32.
