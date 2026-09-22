# Label definitions

The 12 target labels predicted in this project, all derived from the echocardiogram report paired with each ECG in the EchoNext dataset. Each is a binary flag (`1` = finding present at the stated threshold).

### `shd_moderate_or_greater_flag`
The primary target: whether the ECG is associated with clinically meaningful structural heart disease.

`1` = at least one moderate-or-greater structural heart disease finding on echo.

### `lvef_lte_45_flag`
`1` = left ventricular ejection fraction is less than or equal to 45%.

### `lvwt_gte_13_flag`
`1` = left ventricular wall thickness is greater than or equal to 13 mm.

### `aortic_stenosis_moderate_or_greater_flag`
`1` = moderate-or-greater aortic stenosis on echo.

### `aortic_regurgitation_moderate_or_greater_flag`
`1` = moderate-or-greater aortic regurgitation on echo.

### `mitral_regurgitation_moderate_or_greater_flag`
`1` = moderate-or-greater mitral regurgitation on echo.

### `tricuspid_regurgitation_moderate_or_greater_flag`
`1` = moderate-or-greater tricuspid regurgitation on echo.

### `pulmonary_regurgitation_moderate_or_greater_flag`
`1` = moderate-or-greater pulmonary regurgitation on echo.

### `rv_systolic_dysfunction_moderate_or_greater_flag`
`1` = moderate-or-greater right ventricular systolic dysfunction on echo.

### `pericardial_effusion_moderate_large_flag`
`1` = moderate or large pericardial effusion on echo.

### `pasp_gte_45_flag`
`1` = pulmonary artery systolic pressure is greater than or equal to 45 mmHg.

### `tr_max_gte_32_flag`
`1` = maximum tricuspid regurgitation velocity is greater than or equal to 3.2 m/s.

## Input features

### ECG waveform
The 12-lead, 250 Hz, 10-second waveform arrays. The main input for the CNN models.

### Tabular ECG features
`sex`, `age_at_ecg`, `ventricular_rate`, `atrial_rate`, `pr_interval`, `qrs_duration`, `qt_corrected`.

See [`README.md`](../README.md) for how these labels and features are used across the modeling tiers.
