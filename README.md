# phoneme-segmenter
Official repository for paper titled "Using AI to Automate Phonetic Transcription and Perform Forced Alignment for Clinical Application in the Assessment of Speech Sound Disorders" accepted at the AAAI 2025 Workshop on Large Language Models and Generative AI for Health.

# Introduction
Speech-language pathologists (S-LPs) routinely use phonetic transcription to profile and describe the characteristics of a child's speech in the assessment of speech sound disorders (SSDs). The literature identifies phonetic transcription as a demanding perceptual skill, with accuracy and reliability dependent on experience, available resources, and the nature of SSDs. Automatic speech recognition and segmentation techniques, which recognize, transcribe, and align audio file content, have been identified as a possible tool to improve the accuracy and efficiency of the auditory perceptual transcription undertaken by S-LPs. In this paper, we propose a model to automate phonetic transcriptions and perform forced alignment for childhood-disordered speech. Utilizing the state-of-the-art wav2vec 2.0 acoustic model and advanced post-processing algorithms, our model achieves a phoneme error rate of 0.15 and an $F_1$ Score of 82\% on the UltraSuite dataset. These results suggest a level of accuracy greater than what has been reported for auditory-perceptual transcription in the clinical setting.
Note: please set up the paths before running the code

# Proposed architecture
![Phoneme segmenter architecture](/assets/figures/fig_phoneme_segmenter_architecture.pdf)

# Evaluation metrics
1. Phoneme recognition performance: phoneme error rate (PER)
2. Phoneme segmentation performance: P, R, $F_1$, $R$-value
 

# Quantitative result
| Exp | Model                                    | Dataset                          | PER  | P    | R    | $F_1$ | $R$-value |
|-----|------------------------------------------|----------------------------------|------|------|------|-------|-----------|
| 1   | \citeauthor{zhu2022phone}$\dagger$       | UltraSuite                       | 0.20 | 0.45 | 0.73 | 0.55  | 0.35      |
| 2   | \citeauthor{ribeiro2019ultrasound}$\ast$ | UltraSuite                       | 0.63 | 0.75 | 0.70 | 0.73  | 0.76      |
| 3   | Phoneme Segmenter                        | UltraSuite                       | 0.15 | 0.82 | 0.82 | 0.82  | 0.85      |
| 4   | Phoneme Segmenter TL                     | TIMIT, TORGO, UltraSuite         | 0.12 | 0.85 | 0.86 | 0.86  | 0.88      |

# Qualitative result
![Qualitative Results](/assets/figures/fig_01M_BL1_003A_ORANGE_no_TL_TL_comparison.pdf)
