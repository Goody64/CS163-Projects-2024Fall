---
layout: post
comments: true
title: Visual Question Answering
author: Jacob Goodman, Andrew Hong
date: 2024-12-12
---


>
Visual Question Answering (VQA) is a field in artificial intelligence that focuses on enabling machines to answer questions about images by understanding both the visual and textual information. 
In this project, we will evaluate and compare the performance of various VQA models, including LSTM-CNN, HieCoAtten, and MMFT, across different datasets such as VQA v2, CLEVR, and GQA. 
Through a combination of quantitative accuracy metrics and qualitative analysis of attention maps, we aim to uncover insights into how these models reason and where improvements can be made.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

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

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

[2] Antol, Agrawal, Lu, et al. "VQA: Visual Question Answering." *Proceedings of the IEEE International Conference on Computer Vision*. 2015.

---
