# MyFluentEcho – Code Directory

This directory contains all scripts used for data preparation, model training, evaluation, and integration of Whisper with the F5-TTS voice cloning system. Below is a description of each file and its purpose within the project.

---

## 📄 File Descriptions

### `F5-TTS_init.py`
Tests the full pipeline by integrating both the **pre-trained** and **fine-tuned** Whisper models with the F5-TTS system to generate fluent audio output.

### `clean_csvs.py`
Processes and corrects timestamp inconsistencies in CSV annotation files after combining multiple audio clips. Ensures each entry has proper start and end times aligned to the new combined clips.

### `combine_csvs_clips.py`
Combines every 3 CSV files and their corresponding audio clips into a single larger segment. This step is essential to match Whisper’s 30-second training context for more effective fine-tuning.

### `count_total_duration.py`
Calculates and reports the total duration of all audio clips in the dataset. (Total duration used in this project was approximately **5 hours**.)

### `data_prep.py`
Downloads audio clips and CSVs from the **SEP-28k** dataset. *(Note: This dataset was not ultimately used in the final implementation.)*

### `test_F5-TTS.py`
Standalone script to test **F5-TTS** functionality before integrating it with Whisper. Useful for validating voice cloning output independently.

### `whisper_stuttering_fine_tune.ipynb`
Main training notebook for **fine-tuning Whisper** (base and small models) on stuttered speech using the FluencyBank Timestamped dataset. Handles all steps from data loading to model evaluation.

### `whisper_stuttering_fine_tune.html`
HTML-exported version of the Jupyter notebook above for easy sharing and viewing of the fine-tuning workflow and results.

### `whisper_stuttering_fine_tune.py`
Python script version of the Jupyter notebook for running the fine-tuning process via command line or in headless environments.

### `whisper_utils.py`
Utility functions for initializing and loading Whisper models, performing transcription, and supporting integration with F5-TTS audio synthesis.

---

## 🧠 Summary

These scripts collectively power the full MyFluentEcho system — from data processing and Whisper fine-tuning to TTS synthesis using F5-TTS. This modular structure allows for flexible experimentation and evaluation of various model configurations and integration strategies.
