# Final Project – MyFluentEcho: Fluent Speech from Stuttered Input

## Overview  
This repository contains the final project for the Deep Learning course. The project introduces **MyFluentEcho**, a system designed to assist people who stutter by converting disfluent speech into **fluent audio output**, while preserving the original speaker's voice characteristics. Rather than building a TTS system from scratch, this project focuses on **fine-tuning OpenAI’s Whisper model** for stuttered speech recognition and using **F5TTS** (Fast, Few-shot, Fine-tunable Text-to-Speech) for voice cloning.

The system provides an alternative to traditional Delayed Auditory Feedback (DAF) by generating fluent speech from real-time input, enabling natural and personalized auditory feedback for users who stutter.

---

## Repository Structure


```
MyFluentEcho/
│
├── Proposal/
│   └── Initial project proposal (based on earlier TTS idea).
│
├── Final-Group-Project-Report/
│   └── Comprehensive report describing dataset, model architecture, training process, and results.
│
├── Final-Presentation/
│   └── Powerpoint PDF summarizing the final results and system design.
│
└── Code/
    ├── Audio-Samples/
    │   └── Samples used for evaluation of the sytem
    ├── data_splits/
    │   └── Train, test, validation splits for each run
    ├── logs/
    │   └── Storage of previous training data
    ├── model_results/
    │   └── Storage of previous model results in JSON format (not actual models)
    └── Code files explained in folder...

```

---

## Folder Descriptions

- **Proposal/**: Contains the original project proposal (now outdated).
- **Final-Group-Project-Report/**: The final report detailing all aspects of the MyFluentEcho system.
- **Final-Presentation/**: Final slides used for presenting the project.
- **Code/**: All scripts for dataset preparation, Whisper fine-tuning, evaluation, and TTS synthesis using F5TTS.

---

## How the System Works

1. **Stuttered Audio Input**  
   - Short clips of disfluent speech are input to the system.

2. **Fine-Tuned Whisper Model**  
   - A customized Whisper model, fine-tuned on the **FluencyBank Timestamped dataset**, transcribes the stuttered audio while correcting disfluencies like repetitions and prolongations.

3. **F5TTS Voice Cloning**  
   - The corrected text is passed to F5TTS, which synthesizes fluent audio while preserving the speaker’s original voice traits (e.g., accent, pitch, intonation).

4. **Fluent Output**  
   - The user hears a fluent version of their own voice, offering a modern, deep learning-based alternative to traditional DAF techniques.

---

## Performance Evaluation

- **Word Error Rate (WER)**: Used to evaluate the Whisper model’s transcription accuracy before and after fine-tuning.
- **Qualitative Analysis**: Specific transcription comparisons show the model’s improvement in disfluency handling.
- **Subjective Listening Tests**: Used to validate that the fluent speech output sounds natural and matches the speaker’s identity.

---

## Requirements

The main libraries and tools used include:

- `transformers` (HuggingFace)
- `datasets` (HuggingFace)
- `torchaudio`, `librosa`
- `OpenAI Whisper` ([Whisper Github Repo](https://github.com/openai/whisper))
- `F5TTS` ([F5TTS GitHub Repo](https://github.com/SWivid/F5-TTS))
- `PyTorch`, `TensorBoard`

<!--
## How to Use the Code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/casager/Final-Project-11.git
   cd Final-Project-X
   ```

2. **Install the required dependencies:**
   A `requirements.txt` file will be provided to install the necessary Python packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Training the Model:**
   - Place your training data (such as the LJSpeech dataset) into the `data/` directory.
   - To train the Text-to-Spectrogram model, run the following command:
     ```bash
     python Code/train.py
     ```

4. **Evaluating the Model:**
   - After training, evaluate the model by running:
     ```bash
     python Code/evaluate.py
     ```

5. **Generate Predictions:**
   - To generate a mel-spectrogram from new input text, use the following:
     ```bash
     python Code/generate_spectrogram.py --text "Hello, world"
     ```

## Performance Evaluation

The performance of the model will be evaluated using the following metrics:

- **Mean Squared Error (MSE)**: Measures the error between predicted and actual mel-spectrograms.
- **Spectral Distortion (SD)**: Quantifies the difference between predicted and ground-truth spectrograms.
- **Signal-to-Noise Ratio (SNR)**: Evaluates the quality of the audio waveform generated from the spectrogram.
- **Mean Opinion Score (MOS)**: Subjective evaluation of speech quality based on listener preferences.
- **Perceptual Evaluation of Speech Quality (PESQ)**: An objective metric for assessing the perceptual quality of the generated speech.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work is based on the **LJSpeech** dataset, which is publicly available and widely used for training text-to-speech models. We would also like to acknowledge the researchers whose papers and pre-existing models have contributed to the development of this project.
```

## Additional Notes:
1. **Dependencies**: Make sure to include a `requirements.txt` in your repository, which lists the necessary libraries (e.g., `torch`, `librosa`, etc.).
   
2. **Training Script**: The `train.py` script should contain the code to load the dataset, preprocess it, and train the model. Similarly, `evaluate.py` will handle model evaluation, and `generate_spectrogram.py` will allow users to input text and generate the corresponding mel-spectrogram.

3. **Dataset**: If you use the LJSpeech dataset, you might want to include a note on how to download it (e.g., direct link to the dataset) or where to place it in the directory structure.

4. **Model Description**: It may also be helpful to add a section on how the LSTM and CNN layers are combined and how the data is processed before training.

5. **License**: If you are using any pre-built models or datasets that have specific licensing requirements, make sure to mention it in the **Acknowledgements** or **License** section.

Now, you can proceed with creating the repository, pushing the code and this README file, and sharing it with your collaborators.
-->
