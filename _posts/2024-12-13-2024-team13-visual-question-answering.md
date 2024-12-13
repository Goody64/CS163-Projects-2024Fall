---
layout: post
comments: true
title: Visual Question Answering
author: Jacob Goodman, Andrew Hong
date: 2024-12-12
---


>
Visual Question Answering (VQA) combines computer vision and natural language processing to enable AI systems to answer questions about images. This project explores and compares models like LSTM-CNN, HieCoAtten, and MMFT, evaluating their performance on datasets such as VQA v2, CLEVR, GQA, and DAQUAR. Using accuracy metrics and attention map visualizations, we uncover how these models process visual and textual data, highlighting their strengths and identifying areas for improvement.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Visual Question Answering (VQA) is a multi-disciplinary challenge at the intersection of computer vision and natural language processing. In this paper, we explore the evolution of VQA models, starting from foundational architectures to state-of-the-art approaches. We examine the integration of visual and textual information in early models like LSTM-CNN, the advancements introduced by attention-based models such as Stacked Attention Networks (SAN), and the transformative impact of multimodal transformers like MMFT. Additionally, we briefly discuss other notable models, such as MCBLP and HieCoAtten, highlighting their unique contributions to the field. By analyzing the performance, design, and implications of these models, we provide a comprehensive overview of the VQA landscape, concluding with an assessment of their current limitations and future potential.

## Early Fusion: LSTM-CNN

### Overview
Early fusion models in Visual Question Answering (VQA) aim to combine visual and textual features in a straightforward manner, typically by encoding each modality separately and merging them into a unified representation. The LSTM-CNN model, proposed in the VQA-v1 paper by Antol et al. (2015) [1] and expanded upon by Milnowski et al. (2015) [3], exemplifies this approach by utilizing convolutional neural networks (CNNs) to extract visual features and long short-term memory networks (LSTMs) to process questions. Despite its simplicity, the model laid the groundwork for later advancements by showcasing how effective multi-modal fusion could be achieved for answering questions about images.
### Architecture and Process
![LSTM-CNN]({{ '/assets/images/team13/LSTM-CNN-process.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Basic process for the LSTM-CNN* [3].

The LSTM-CNN model operates in two stages: visual feature extraction and textual question encoding, followed by fusion and prediction. The process is as follows:

1. Image Feature Extraction (CNN):
The model begins by passing the image through a pre-trained CNN, such as ResNet or VGG, which extracts high-level visual features. These features represent various aspects of the image, including objects, textures, and their spatial relationships. The output of this step is a fixed-length feature vector representing the entire image.

2. Question Encoding (LSTM):
The question is tokenized into a sequence of words and embedded into a dense vector space using an embedding layer. These word embeddings are then processed by an LSTM, which captures the sequential dependencies of the question and encodes it into a fixed-length vector. This vector encapsulates the semantic meaning of the question.

3. Fusion of Image and Question Features:
Once both the visual and textual features have been extracted, they are concatenated into a single vector. This combined representation merges the information from both modalities, enabling the model to use the visual context in conjunction with the question’s meaning for reasoning.

4. Prediction Layer:
The fused vector is passed through a fully connected layer (typically a dense layer), which produces the final output—either a class label (for multiple-choice questions) or a probability distribution over possible answers. This layer performs the task of generating the answer based on the combined understanding of the image and question.

This approach highlights a simple but effective method of fusing image and textual information early in the processing pipeline. However, it lacks the ability to focus on specific parts of the image in response to a question, which is a limitation that later attention-based models would address.

### Code Implementation
Below is a general implementation of the model based on the architecture and approach described in the paper.
>
While the original paper uses an older model architecture and hyperparameters, this implementation updates certain components, such as replacing the CNN with a pretrained GoogleNet model and simplifying the LSTM layer. The final model combines image features extracted by the CNN and question features processed by the LSTM, followed by a fusion layer to make predictions. This implementation aims to reflect the key ideas from the paper with modernized code.

keep that or not idc idk if i want it there or not ^^

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# 1. Image Feature Extraction (CNN) using GoogleNet
class CNN_Extractor(nn.Module):
    def __init__(self):
        super(CNN_Extractor, self).__init__()
        # Using pretrained GoogleNet model from ImageNet, removing final fully connected layer
        self.googlenet = models.googlenet(pretrained=True)
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, 256)

    def forward(self, x):
        # Extract image features through the modified GoogleNet model
        features = self.googlenet(x)
        return features

# 2. Question Encoding (LSTM) - Single Layer LSTM
class Question_Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super(Question_Encoder, self).__init__()
        # Embedding layer and LSTM for question encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=1)

    def forward(self, x):
        # Convert word indices to embeddings and pass through LSTM
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]

# 3. Fusion of Image and Question Features
class Feature_Fusion(nn.Module):
    def __init__(self, image_dim=256, question_dim=256, fusion_dim=512):
        super(Feature_Fusion, self).__init__()
        # Fully connected layer to combine image and question features
        self.fc = nn.Linear(image_dim + question_dim, fusion_dim)

    def forward(self, image_features, question_features):
        # Concatenate and fuse image and question features
        combined = torch.cat((image_features, question_features), dim=1)
        return self.fc(combined)

# 4. Prediction Layer
class Prediction_Layer(nn.Module):
    def __init__(self, input_dim=512, output_dim=10):
        super(Prediction_Layer, self).__init__()
        # Fully connected layer for final prediction
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Output predictions using softmax
        return F.softmax(self.fc(x), dim=1)

# Complete LSTM-CNN Model
class LSTM_CNN_Model(nn.Module):
    def __init__(self, vocab_size, output_dim=10):
        super(LSTM_CNN_Model, self).__init__()
        # Initialize submodules: CNN feature extractor, LSTM question encoder, feature fusion, and prediction layer
        self.cnn_extractor = CNN_Extractor()
        self.question_encoder = Question_Encoder(vocab_size)
        self.feature_fusion = Feature_Fusion()
        self.prediction_layer = Prediction_Layer(input_dim=512, output_dim=output_dim)

    def forward(self, image, question):
        # Extract image and question features, fuse them, and predict the final output
        image_features = self.cnn_extractor(image)
        question_features = self.question_encoder(question)
        fused_features = self.feature_fusion(image_features, question_features)
        return self.prediction_layer(fused_features)

```

### Results

The performance of the LSTM-CNN model was evaluated using the DAQUAR dataset. The model was tested on both the full dataset and a reduced version with 25 images and 297 QA pairs. The evaluation was done using standard metrics: accuracy and WUPS scores at thresholds of 0.9 and 0.0.

<h4><span style="color:black;">Explanation of WUPS Scores</span></h4>

WUPS (Word Overlap with Partial Matching Score) is a metric used to evaluate the semantic similarity between the predicted and ground truth answers. It is particularly useful in visual question answering (VQA) tasks where the output may not exactly match the ground truth answer but may still be semantically close. 

- **WUPS@0.9**: This score measures how closely the predicted answer matches the ground truth, with a high threshold (0.9) for word overlap. A higher score means the predicted answer is more likely to be semantically identical or nearly identical to the ground truth answer.
- **WUPS@0.0**: This score is less stringent, allowing for partial matches in the answers. A score of 0.0 means there is no threshold for word overlap, and even partially matching answers can achieve higher scores.

WUPS scores provide a more nuanced view of model performance compared to accuracy, as they consider semantic similarity and partial matches, which are important in natural language processing tasks.

<h4><span style="color:black;">Quantitative Results</span></h4>
Full DAQUAR Dataset

- **Multiple Words:** 
  - Accuracy: 17.49%
  - WUPS@0.9: 23.28%
  - WUPS@0.0: 57.76%

- **Single Word:** 
  - Accuracy: 19.43%
  - WUPS@0.9: 25.28%
  - WUPS@0.0: 59.13%

These results suggest that while the model handles both multiple-word and single-word queries, its performance is limited by the complexity and variation in the dataset. The WUPS scores at both thresholds indicate the model’s ability to produce answers with varying degrees of semantic similarity, though it struggles to provide precise answers (e.g., at WUPS@0.9).

Reduced DAQUAR Dataset (25 images, 297 QA pairs)

- **Multiple Words:**
  - Accuracy: 29.27%
  - WUPS@0.9: 36.50%
  - WUPS@0.0: 66.33%

- **Single Word:**
  - Accuracy: 34.68%
  - WUPS@0.9: 40.76%
  - WUPS@0.0: 67.95%

The model shows improved performance across all metrics when tested on the reduced dataset. The increased accuracy and WUPS scores indicate that the LSTM-CNN model performs better with less complexity and fewer variations in the data, with notable improvements in both single-word and multiple-word queries.

<h4><span style="color:black;">Comparison with Language-Only Model</span></h4>

For comparison, a language-only model was evaluated on the same datasets:

- **Full DAQUAR dataset:**
  - Accuracy: 17.06%
  - WUPS@0.9: 21.75%
  - WUPS@0.0: 53.24%

- **Reduced DAQUAR dataset:**
  - Accuracy: 32.32%
  - WUPS@0.9: 35.80%
  - WUPS@0.0: 63.02%

The language-only model showed lower performance than the LSTM-CNN model, especially with complex, visual-based queries. However, it performed reasonably well in terms of accuracy on the reduced dataset, though it struggled more with precise, context-aware answers as reflected by the WUPS scores.

<h4><span style="color:black;">Qualitative Results</span></h4>

Qualitative analysis of the model’s output revealed that the LSTM-CNN model is effective in generating answers aligned with the visual context in the images. The model shows good spatial reasoning when handling questions related to object counting or identifying attributes. However, it struggles with queries involving spatial relations, achieving only around 21 WUPS at 0.9, a challenging aspect that accounts for a significant portion of the DAQUAR dataset.

The model performs well on questions involving clearly defined objects and their attributes, such as identifying locations or colors. However, tasks that require understanding the spatial relationships between objects are more challenging. These types of queries, which involve questions like relative positions or sizes, often result in lower accuracy due to the difficulty in interpreting complex spatial interactions from the visual input. Additionally, questions about small objects, negations, and shapes tend to have reduced performance, likely due to limited training data for these scenarios. Despite these challenges, focusing on simpler or less ambiguous queries allows the model to better leverage available visual information.


## Attention Mechanisms: Learning to Focus on Relevant Features

## Transformers in VQA: Multimodal Fusion with MMFT

## Other Models in VQA

## Conclusion and Future Perspectives

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![VQA]({{ '/assets/images/team13/vqa-general-examples.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. VQA Examples: basic examples from the VQA V2 dataset* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## References

[1] Antol, S., Agrawal, A., Lu, J., et al. "VQA: Visual Question Answering." 2015. arXiv:1505.00468v1.

[2] Agrawal, A., Lu, J., Antol, S., et al. "VQA: Visual Question Answering." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2016. arXiv:1505.00468.

[3] Malinowski, M., Rohrbach, M., & Fritz, M. "Ask Your Neurons: A Neural-Based Approach to Answering Questions about Images." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2015. arXiv:1505.01121.

[4]

---
