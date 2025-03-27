# Final Project - Text-to-Speech Synthesis

## Overview
This repository contains the work for the **Final Project** in our deep learning course. The project aims to develop a custom **Text-to-Speech (TTS)** model. The focus is on training a **Text-to-Spectrogram Conversion Model** using a **combination of LSTM and CNN** networks to generate mel-spectrograms from text input. A pre-built vocoder is used on the backend for generating audio from the spectrograms.

## Repository Structure

The repository is organized as follows:

```
Final-Project-X/
│
├── Proposal/
│   └── Proposal document outlining the project goals, methods, and approach.
│
├── Final-Group-Project-Report/
│   └── The final group report summarizing the methodology, results, and conclusions.
│
├── Final-Presentation/
│   └── Final presentation slides.
│
└── Code/
    └── The code for training the Text-to-Spectrogram model, including scripts for training, preprocessing, and evaluation.

```

## Folder Descriptions

- **Proposal/**: Contains the project proposal, where the problem, approach, and methodology are described.
- **Final-Group-Project-Report/**: Contains the final report summarizing the work done by the group, including model architecture, results, and conclusions.
- **Final-Presentation/**: Contains the slides for the final presentation given at the end of the course.
- **Code/**: Contains the code for training the Text-to-Spectrogram model, including scripts for data preprocessing, model training, and evaluation.

<!--
## How to Use the Code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Final-Project-X.git
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

## Contributors

- **FirstName LastName** (Group Member 1)
- **FirstName LastName** (Group Member 2)
- **FirstName LastName** (Group Member 3)

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
