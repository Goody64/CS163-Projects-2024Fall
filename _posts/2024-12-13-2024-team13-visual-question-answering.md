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

## Early Fusion: LSTM-CNN and the Foundations of VQA

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

[1] Antol, Agrawal, Lu, et al. "VQA: Visual Question Answering." *Proceedings of the IEEE International Conference on Computer Vision*. 2015.

---
