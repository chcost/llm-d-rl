# quickstart/ - one worked example

**Nothing here is required.** This is how *we* run the integration on our own
cluster: a KubeRay definition, provisioning scripts, and the benchmark harness the
routing results come from. To adopt the integration itself, read
[`integrations/README.md`](../integrations/README.md) - it needs none of this.

```
quickstart/
  kuberay/      the cluster: two topology templates, one deploy.sh, provisioning
  benchmarks/   the measurement harness
    scripts/    framework-agnostic: run_on_head.sh, vllm_scrape.py, push-epp.sh
    verl/       run_test.sh + 9 workloads
    vime/       run script
    slime/      run script
  images/       Dockerfiles for the environment images
```

## Start here

```bash
export NAMESPACE=<your-namespace>
cd kuberay
./deploy.sh pvc        --framework verl   # once per namespace
./deploy.sh apply      --framework verl   # start Ray
./deploy.sh provision  --framework verl   # install verl + the integration
./deploy.sh check      --framework verl   # verify every node agrees
cd ../benchmarks && scripts/run_on_head.sh --mode epp --task gsm8k --steps 1
```

[`kuberay/README.md`](kuberay/README.md) explains each step, why provisioning is
separate from pod startup, and what to edit to change a version.

## How it is organised

The cluster templates are keyed by **topology**, not by framework - a CPU head
with GPU workers, or one pod owning the GPUs. Which framework runs is an image
plus a provisioning script plus a block of values, so adding a framework costs no
manifest. Supported today: verl (head + workers), vime and slime (single pod).

Framework-agnostic tooling lives in `benchmarks/scripts/`; only the driver and the
workloads are per framework.
