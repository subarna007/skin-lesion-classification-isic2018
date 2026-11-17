| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUC | Melanoma sensitivity | DF sensitivity |
|---|---|---|---|---|---|---|
| ResNet-50 baseline (224 px) | 0.747 (0.724-0.769) | 0.713 (0.664-0.756) | 0.609 | 0.932 (0.919-0.944) | 0.622 (0.546-0.692) | 0.542 |
| ConvNeXt-Tiny (384 px) | 0.786 (0.766-0.808) | 0.767 (0.722-0.809) | 0.725 | 0.966 (0.957-0.974) | 0.821 (0.761-0.877) | 0.667 |

Test set: 1,508 images, a held-out split grouped by lesion. Brackets are 95% bootstrap intervals (1,000 resamples).

Melanoma screening view (flag when P(MEL) is above a threshold fixed on the validation set):

| Model | Threshold | Test sensitivity | Test specificity | Melanoma AUC |
|---|---|---|---|---|
| ResNet-50 baseline | 0.067 | 0.910 (0.864-0.952) | 0.636 | 0.869 |
| ConvNeXt-Tiny | 0.217 | 0.942 (0.904-0.975) | 0.755 | 0.933 |
