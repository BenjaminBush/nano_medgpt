# Nano Med GPT

## Overview

Nano Med GPT aims to develop a specialized, small-scale version of the GPT language model tailored sepcifically for medical note data. Although there are more advanced tools available, such as DAX Dragon Express (which help provide inspiration for this project), this important use case presents a strong opportunity for hands-on learning and development with transformer-based DNN architectures. 

## Problem

Providers spend nearly 2 hours per day outside of normal office hours writing notes in a patient's chart. This time consuming task can lead to clinician burnout. Generative AI can assist with provider documentation by suggesting autocompletion, which may reduce the time required to complete a note and therefore alleviate administrative burdens. 

## Key Features

Nano Med GPT is trained on clinician note data from [Mimic IV dataset](https://physionet.org/content/mimiciv/2.0/). Specifically, it is trained on the discharge notes (not the radiology notes). 

Nano Med GPT seeks to predict the next word in the sequence. It should be possible to modify the prediction scope to predict the next *n* words. 

The first version of Nano Med GPT should be based on the [GPT-2 model architecture](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), so that it may be trained locally on a CUDA-enabled GPU. In the future, we may explore finetuning an OpenAI GPT 3.5T model (or other) using the [OpenAI finetuning API](https://platform.openai.com/docs/guides/fine-tuning). 

## User Stories
Simone is a general surgery resident at Lakeview Hospital. Simone gets to the hospital at 6AM to start rounding on the patients that stayed overnight and review the notes produced by the surgery team on the night shift, as well as review the EHRs for patients scheduled for surgery today. Throughout their day, Simone assists with four apendectomies and checks in on five of the patients that stayed the night before. Simone has very little down-time during the day and is not able to write notes until much later in the evening. At 5PM, Simone begins to write their notes for the four patients who received appendectomies and the five patients who stayed the previous night. Simone spends about 10 minutes writing each note, but must stop at 6PM to sign-out to the night-shift surgery team. Simone resumes writing her notes at 7PM and completes them by 7:30PM. Although they are satisfied with their work, Simone is tired from the additional burden note-taking requires.

## Requirements
* Gain access to MIMIC IV Dataset
* Abide by [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/2.0/)
* Reproduce GPT-2 Model Architecture
* Train GPT-2 Model on Mimic IV dataset (discharge table) to predict the next word
* Evaluate the results of the model
* Produce detailed writeup that includes data processing/cleansing steps, exploratory data anlysis findings, evaluation metrics, results, interesting findings, and opportunities for future work. 