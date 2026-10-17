# Segmented recurrent convolution prerequisite

The production MoME call now delegates to `resident::causal_convolution`, which
accepts explicit sequence starts, resident state slots and prior positions.
The shared implementation submits one vendor invocation for the packed rows;
scratch remains separate from input to avoid the vendor inplace write race.

On 7.156.99.58, inside Docker `inferfabric-92b`,
`native-build-head-batch/sequence_convolution_probe` passed:

```json
{"passed":true,"launches":27,"checked_values":94832640}
```

The probe uses that same production function, with widths 512/1024/6144 and
3/17/4096 total rows. Each invocation includes three sequences, with single-row
requests surrounding an extend. Three consecutive invocations reorder resident
slots and advance separate histories. Output is checked against a CPU convolution
oracle using the existing BF16 tolerance; the entire recurrent state is checked
exactly, including an untouched fourth resident. Fixture upload/readback is test
instrumentation, not a production activation transfer.

Native `program_engine_probe` also builds successfully with this helper. This
is a kernel prerequisite for multi-request execution, not full-model batching
qualification. The decoder still provides one sequence's controls. Batched
attention/index page tables, sampling, submission ABI and scheduler wiring
remain necessary before enabling multi-request forwards.
