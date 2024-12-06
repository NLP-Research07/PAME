# Faux Hate Detection with PAME Framework

This repository introduces a novel concept of "Faux Hate," which refers to the generation of hate speech fueled by fake news and misinformation. As fake narratives spread across platforms, they often trigger harmful speech, making it crucial to develop systems that can detect and mitigate such content effectively. To tackle this issue, we propose the Proximity-Aware Meta Embedding (PAME) framework. PAME combines advanced embedding techniques with proximity analysis to capture the complex relationships between fake news and the hate speech it can provoke. The framework enhances hate speech detection by considering the context of misinformation and how it influences the generation of harmful content. By integrating fake news detection with hate speech classification, PAME improves the ability to identify both directly harmful content and content that indirectly incites harm through the spread of false information. Our framework shows promising performance in distinguishing between genuine hate speech and Faux Hate, providing a more nuanced approach to content moderation.

## Qualitative Analysis

To gain a deeper understanding of the relationship between fake content and hate speech, we conducted a chi-square test to examine potential connections between their corresponding labels. The test was based on two hypotheses: the null hypothesis (H₀), which posits that no significant relationship exists between fake content (F) and hateful comments (H), and the alternative hypothesis (H₁), which suggests a statistically significant relationship between the two. The results of the chi-square test revealed a strong and statistically significant relationship between fake content and hate speech, with χ² (df=1, N=8014) = 1087, P=0.00, rejecting the null hypothesis in favor of the alternative hypothesis. Furthermore, we extended the analysis to explore the relationship between fake content and the severity of hate speech. The results of this test, χ² (df=2, N=5138) = 459, P=0.00, further confirmed a significant connection between fake content and the severity of hate speech. Additionally, we explored the relationship between fake content and the target class, which also showed a statistically significant association, with χ² (df=2, N=5138) = 1052, P=0.00. These findings collectively emphasize the strong interconnection between fake content and both the intensity and nature of hate speech, offering valuable insights into the dynamics of online harmful content.

## PAME Framework Overview

In the PAME (Proximity-Aware Meta Embedding) framework 

![PAME-Framework](https://github.com/NLP-Research07/PAME/blob/main/Architectural%20Diagrams/PAME-Framework.png)

we enhance both Hate and Fake speech detection by incorporating additional features derived from Euclidean Distances, which we obtain from auxiliary corpora. Specifically:

- **Hate Speech**: We use the **HASOC 2020 and 2021 datasets** as the auxiliary corpus.
- **Fake Speech**: We use the **Hostile dataset** as the auxiliary corpus.

Through pilot studies, we have experimented with several datasets and found that these corpora provide richer meta-information, contributing to the overall performance improvement of our framework.

### Data Distribution

Below is the data distribution for the auxiliary and target datasets used in our experiments:

| **Dataset**                        | **Hateful Comments** | **Non-Hate Comments** | **Fake Comments** | **Non-Fake Comments** |
|------------------------------------|----------------------|-----------------------|-------------------|-----------------------|
| **Hasoc 2020 and 2021 Combined**   | 2,280                | 5,277                 | -                 | -                     |
| **Constraint Dataset**             | -                    | -                     | 1,708             | 3,050                 |
| **Faux Hate Dataset (Train Data)** | 4,090                | 2,277                 | 3,305             | 3,063                 |
| **Faux Hate Dataset (Val Data)**   | 522                  | 275                   | 407               | 390                   |
| **Faux Hate Dataset (Test Data)**  | 488                  | 309                   | 406               | 391                   |

### Meta Embedding Features

The proposed Meta Embedding contains **770 features**, which typically exceed the base embedding size of a transformer model (768-dimensional embeddings). These additional features come from the Euclidean distance calculations between the Hate and Fake classes, derived from the auxiliary corpora.

The PAME framework relies on knowledge acquired during training for automatic reasoning and decision-making [1]. While some methods attempt to include contextual information for claim evaluation, they still require manual user input, which hinders real-time detection [2, 3]. Despite these challenges, PAME achieved a 2-3% improvement over baseline models, highlighting its potential and promising future for further development.

## joint Learning Framework

We have also built a joint learning framework to handle Fake and Hate content simultaneously

<img src="https://github.com/NLP-Research07/PAME/blob/main/Architectural%20Diagrams/Shared%20Layer%20Architecture.png" alt="Shared Layer Architecture" width="500"/>

and the results are as follows:

### Performance Comparison of Baseline (MTL) and Proposed Method (PAME) on Hate and Fake Tasks

| **Model**            | **Hate Acc**  | **Hate F1**  | **Fake Acc**  | **Fake F1**  |
|:--------------------:|:-------------:|:------------:|:-------------:|:------------:|
| **Baseline (MTL)**    |               |              |               |              |
| mBERT                | **74**        | **72**       | **75**        | **74**       |
| XLM-RoBERTa          | 69            | 69           | 72            | 72           |
| Electra              | 66            | 55           | 64            | 63           |
| ALBERT               | 67            | 57           | 63            | 62           |
| **Proposed Method (PAME)** |         |              |               |              |
| PAME + mBERT         | **95** (↑ 28%) | **94** (↑ 30%) | **75** (↑ 0%) | **75** (↑ 1%) |
| PAME + XLM-RoBERTa   | 89 (↑ 29%)    | 88 (↑ 27%)    | 75 (↑ 4%)     | 74 (↑ 3%)    |
| PAME + Electra       | 79 (↑ 20%)    | 77 (↑ 22%)    | 65 (↑ 1%)     | 65 (↑ 2%)    |
| PAME + ALBERT        | 73 (↑ 9%)     | 68 (↑ 11%)    | 64 (↑ 1%)     | 63 (↑ 1%)    |


## Experimentation Results

We have also experimented with different metrics like Cosine Similarity and Dot Product, but the results were not satisfactory. Euclidean Distance considers both the size and direction of differences between embeddings, which helps capture subtle variations in Faux Hate. In contrast, Cosine Similarity focuses only on the angle between embeddings, ignoring magnitude, which might miss important context or intensity differences. Similarly, the Dot Product measures how much two vectors point in the same direction but doesn't account for the magnitude of the vectors, making it less effective for capturing intensity differences—an essential factor for detecting nuanced categories like Faux Hate. Below, we provide the results obtained using Cosine Similarity and Dot Product for reference.

### Cosine Similarity Results with Best Performing Model (mBERT)

| **Task** | **Accuracy (%)** | **Macro F1 (%)** |
|:--------:|:----------------:|:----------------:|
| Hate     | 75               | 71               |
| Fake     | 75               | 74               |

### Dot Product Results with Best Performing Model (mBERT)

| **Task** | **Accuracy (%)** | **Macro F1 (%)** |
|:--------:|:----------------:|:----------------:|
| Hate     | 74               | 72               |
| Fake     | 75               | 75               |

### Ablation Study on Feature Selection

In this ablation study, we focused on performing experiments with different feature selections, while not removing core parts of the proposed framework, such as Centroid Computation or adding Euclidean Distance as meta-information. These elements are integral to the PAME framework, and removing them would make the model a normal Multi-Task Learning (MTL) model. Therefore, our experiments were restricted to feature selection variations, and the results are as follows:

#### mBERT with Only Hate Distance Included (769 Dimensions)

| **Task** | **Accuracy (%)** | **Macro F1 (%)** |
|:--------:|:----------------:|:----------------:|
| Fake     | 75.11            | 75.05            |
| Hate     | 86.03            | 83.88            |

#### mBERT with Only Fake Distance Included (769 Dimensions)

| **Task** | **Accuracy (%)** | **Macro F1 (%)** |
|:--------:|:----------------:|:----------------:|
| Fake     | 75.49            | 75.36            |
| Hate     | 75.36            | 71.73            |

#### XLM-RoBERTa with Only Hate Distance Included (769 Dimensions)

| **Task** | **Accuracy (%)** | **Macro F1 (%)** |
|:--------:|:----------------:|:----------------:|
| Fake     | 75.92            | 75.91            |
| Hate     | 85.03            | 82.83            |

#### XLM-RoBERTa with Only Fake Distance Included (769 Dimensions)

| **Task** | **Accuracy (%)** | **Macro F1 (%)** |
|:--------:|:----------------:|:----------------:|
| Fake     | 75.61            | 75.61            |
| Hate     | 75.55            | 71.70            |

### Observations

In this study, the ablation on feature selection shows how different feature combinations influence the model's performance. Although removing core components like Centroid Computation and adding Euclidean Distance as meta-information were not explored in this study, it is clear that the inclusion of both Fake and Hate distances together yields higher accuracy and F1 scores. The results indicate that separating the distances for Fake and Hate tasks can impact the overall performance, highlighting the importance of both Hate and Fake Features to deal with Faux Hate Data.


### Quality of Auxiliary Corpus

We acknowledge that the reliance on high-quality auxiliary datasets poses a limitation, especially in data-scarce environments. To address this, alternative methods, such as generating synthetic datasets using state-of-the-art **Large Language Models (LLMs)** like **ChatGPT, Gemini, Claude, Llama**, and **Mistral**, can be employed. These models have proven capable of generating synthetic data that closely mimics human-generated patterns, which can help mitigate the data scarcity issue.

Recent research has shown that synthetic data generated by LLMs can closely resemble human-generated data[4], particularly when fine-tuned and supplemented with techniques like grounding and prompt filtering to reduce artifacts. Despite the challenges in achieving complete authenticity, these models can effectively generate high-quality data.

In our experiments, the **HASOC dataset (for Hate Speech)** and the **Constraint dataset (for Fake News)** provided good quality data. While this approach has limitations, the performance improvements observed in detecting Hate Speech are significant, and we believe this method can be extended to other domains.

By incorporating **meta-data** into the base embeddings (768-dimensional), we improve the model's ability to capture nuanced contextual patterns that may not be fully represented in the original embeddings alone.


## References
[1] Wang, B., Ma, J., Lin, H., Yang, Z., Yang, R., Tian, Y., & Chang, Y. (2024, May). Explainable Fake News Detection With Large Language Model via Defense Among Competing Wisdom. In Proceedings of the ACM on Web Conference 2024 (pp. 2452-2463).

[2] Yi-Ju Lu and Cheng-Te Li. 2020. GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 505–514, Online. Association for Computational Linguistics.

[3] Shu, K., Cui, L., Wang, S., Lee, D., & Liu, H. (2019, July). defend: Explainable fake news detection. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 395-405).

[4] Veselovsky, V., Ribeiro, M.H., Arora, A., Josifoski, M., Anderson, A. and West, R., 2023. Generating faithful synthetic data with large language models: A case study in computational social science. arXiv preprint arXiv:2305.15041.



Note : Dataset and Code will be publicly available once the paper get accecpted. 
