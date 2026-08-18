# Installing KubeRay

The steps below are based on the
[KubeRay Operator Helm Chart](https://github.com/ray-project/kuberay/tree/7092f76e6f08fa86ad21c37cd8216914dd215975/helm-chart/kuberay-operator).

The CRDs are installed separately from the operator, to keep the permission requirements for each independent.

### 1. Add the KubeRay Helm repository

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
```
To see whether the cluster you're working on has them, run:

```bash
oc get crd | grep ray
```

You should see:

```
rayclusters.ray.io
rayjobs.ray.io
rayservices.ray.io
```

If not, the CRDs can be installed with:

```bash
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/crd?ref=v1.5.1&timeout=90s"
```

But this requires admin.

### 2. Install the operator specifically into your namespace

```bash
helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.5.1 \
  --namespace <your-namespace> \
  --set singleNamespaceInstall=true \
  --skip-crds
```

These flags deploy the operator within a single namespace and skip the CRD installation, which also lets non-admin users deploy it on clusters that already have the CRDs.
