# Learned proximal networks for inverse problems

This repository contains code for the experiments in
[Plug-and-Play Methods Provably Converge Even with Improperly Trained Denoisers: Convergence by Architectural Design](https://arxiv.org/abs/2609.33087)
by Henry Pritchard and Rahul Parhi.

## Experiments

| Directory | Data | Image size | Inverse problem |
| --- | --- | ---: |-----------------|
| [`CT/`](CT/) | Mayo liver CT | 512 x 512 | Parallel-Beam CT|
| [`MRI/`](MRI/) | fastMRI multicoil knee | 320 x 320 | Multicoil MRI   |

## Required packages

Both experiments use the existing `research` Conda environment:

```bash
conda activate research
```

Shared requirements:

- Python 3.11
- PyTorch with CUDA support
- NumPy
- Matplotlib
- scikit-image
- tqdm
- TensorBoard

CT additionally requires `leapct`. MRI additionally requires `h5py`.

The current environment was verified with Python 3.11.15, PyTorch 2.13.0 with
CUDA 13.0, NumPy 2.4.6, Matplotlib 3.11.0, scikit-image 0.26.0, tqdm 4.69.0,
TensorBoard 2.20.0, `leapct` 1.26, and `h5py` 3.16.0.

## Data

The datasets are not stored in this repository:

- Download fastMRI from the [official fastMRI site](https://fastmri.med.nyu.edu/).
- Download the Mayo CT data from the
  [shared Google Drive folder](https://drive.google.com/drive/folders/1gKytBtkTtGxBLRcNInx2OLty4Gie3pCX).

The CT loader expects `train`, `val`, and `test` directories containing DICOM
files. The MRI loader expects the standard fastMRI `multicoil_train`,
`multicoil_val`, and `multicoil_test` directories. Update each experiment's
`DATA_ROOT` setting to point to the local dataset location.

## Training
The configuration block at the top of each `train.py` controls the architecture,
noise level, batch size, learning-rate schedule, validation frequency, Cuda device, etc...
Each run is written to `results/TIMESTAMP/`, including the training parameters in `training_parameters.txt`.
To resume a run, set `RESUME_FROM` in `train.py` to an intermediate checkpoint.  The standard PnP reconstruction uses `gamma=1`. For the relaxed proximal mapping used by the CT experiment, choose any gamma
strictly between zero and one.

## Testing
The configuration blocks at the top of `inverse_ct_test.py` and `inverse_mri_test.py` control all inputs, including 
the index of the test image, model location, denoiser strength, relaxed reconstruction parameter, etc...
They also contain the required parameters to control the forward model. 


Relaxed reconstruction runs an inner LBFGS solve at every outer iteration, so
it is slower than standard PnP. At zero-based outer iteration
`K`, the inner solve continues until its objective-gradient norm is less than
`c / (1 + K)^p`. The defaults are `c=1` and `p=1.6`, with validation requiring
`c>0` and `p>3/2`. `RELAXED_LBFGS_ITERATIONS` is a safety limit; the script
raises an error instead of silently accepting an inner solution that misses
the required tolerance. Relaxed inner solves use float64 because the later
global gradient tolerances are below the reliable float32 floor for a 320 x
320 variable.
