# ePillID Dataset

A **low-shot fine-grained benchmark dataset** for pill identification using computer vision.

## Overview

This project uses the **ePillID Benchmark**, a dataset designed for developing and evaluating computer vision models for pharmaceutical pill identification. The dataset reflects real-world challenges in image-based medication recognition, including **fine-grained visual differences between pills and limited labeled training data**.

The dataset is widely used for **low-shot learning, metric learning, and fine-grained classification tasks**.

**Paper:** https://arxiv.org/abs/2005.14288  

**Citation:**

```bibtex
@inproceedings{usuyama2020epillid,
  title={ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification},
  author={Usuyama, Naoto and Delgado, Natalia Larios and Hall, Amanda K and Lundin, Jessica},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
```

---

# Dataset Characteristics

### Subset Used in This Project

| Metric | Value |
|------|------|
| **Total Images** | 3,727 |
| **Unique Pill Types** | 960 |
| **Unique NDC Codes** | 107 |
| **Image Resolution** | 224 × 224 pixels (JPG) |
| **Front-Facing Images** | 2,034 (54.6%) |
| **Back-Facing Images** | 1,693 (45.4%) |

### Full Benchmark Dataset

The full ePillID dataset contains:

- **13,000+ images**
- **9,804 appearance classes**
- Two image appearances for **4,902 pill types**

Most appearance classes have **only one reference image**, making this dataset particularly suited for **low-shot learning scenarios**.

---

# Image Types

The dataset contains two primary image categories:

### Reference Images
- Captured under **controlled lighting conditions**
- Taken using **professional equipment**
- Consistent backgrounds

### Consumer Images
- Captured in **real-world environments**
- Variable lighting conditions
- Diverse backgrounds and capture devices

### Dual-Sided Coverage

Most pills are captured **from both sides**, meaning two images represent each pill appearance. This adds complexity to the classification problem.

---

# Image Organization

Images are organized in the following directory structure:

```
images/
└── fcn_mix_weight/
    └── dc_224/
        ├── 0.jpg
        ├── 10.jpg
        ├── 100.jpg
        └── ...
```

**Image Format**
- JPG
- Resolution: **224 × 224 pixels**
- Normalized size for **CNN training**

---

# Metadata Structure

The dataset includes a metadata file:

```
image_meta.csv
```

This file contains structured information for each image.

### File Statistics

- **Total Records:** 3,727
- **File Format:** CSV
- **File Size:** ~464 KB

### Column Descriptions

| Column | Type | Description | Example |
|------|------|------|------|
| `images` | String | Image filename | `0.jpg` |
| `image_path` | String | Relative path to image file | `fcn_mix_weight/dc_224/0.jpg` |
| `pilltype_id` | String | Unique pill identifier with hash suffix | `51285-0092-87_BE305F72` |
| `label_full` | String | Full label identifier | `51285-0092-87_BE305F72` |
| `label` | String | Abbreviated label without hash suffix | `51285-0092-87` |
| `label_code_id` | Integer | Manufacturer / product identifier | `51285` |
| `prod_code_id` | Integer | Product strength or dosage identifier | `92` |
| `is_front` | Boolean | Indicates front-facing image | `TRUE` |
| `is_ref` | Boolean | Indicates reference image | `FALSE` |
| `is_new` | Boolean | Indicates newly added image | `FALSE` |

---

# Data Distribution

```
Total Images:               3,727
Unique Pill Types:          960
Unique Label Codes:         107
Unique Product Codes:       817

Front-Facing Images:        2,034 (54.6%)
Back-Facing Images:         1,693 (45.4%)

Reference Images:           0 (0%)
Consumer Images:            3,727 (100%)

New Images:                 0 (0%)
Existing Images:            3,727 (100%)
```

---

# NDC (National Drug Code) Mapping

The dataset uses the **National Drug Code (NDC)** system for pharmaceutical identification.

NDC components include:

- **Label Code ID** → Manufacturer or product identifier
- **Product Code ID** → Strength or dosage identifier
- **Hash Suffix** → Additional identifier added by the dataset

Pill imprints and NDC codes can be verified at:

https://www.drugs.com/imprints.php

---

# Usage

This dataset is suitable for several machine learning tasks:

- **Fine-grained image classification**
- **Metric learning**
- **One-shot learning**
- **Siamese neural networks**
- **Low-shot recognition problems**

The normalized images and metadata structure make the dataset compatible with common deep learning frameworks such as:

- PyTorch
- TensorFlow
- Keras

---

# Disclaimer

This dataset and associated code are released for **research purposes only**.

---

# References

Full dataset repository:

https://github.com/usuyama/ePillID-benchmark/releases