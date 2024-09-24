import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import soundfile as sf
import gradio as gr
import librosa
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
asr_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
asr_encoder.eval()
asr_encoder.to(device)

# Load ASR transcription model (used to get initial transcription)
asr_transcription_processor = Wav2Vec2Processor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
)
asr_transcription_model = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
)
asr_transcription_model.eval()
asr_transcription_model.to(device)

# Load LLM decoder (Japanese T5 model)
llm_tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
llm_decoder = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
llm_decoder.to(device)


# Set special tokens
llm_tokenizer.pad_token = llm_tokenizer.eos_token

# Get dimensions
asr_output_dim = asr_encoder.config.hidden_size  # e.g., 768
llm_input_dim = llm_decoder.config.d_model  # e.g., 768

# Define the linear projector
projector = nn.Linear(asr_output_dim, llm_input_dim)
projector.to(device)
projector.eval()

# Load the trained model weights
checkpoint = torch.load("model_epoch3.pth", map_location=device)
projector.load_state_dict(checkpoint["projector_state_dict"])
llm_decoder.load_state_dict(checkpoint["llm_decoder_state_dict"])


def process_audio_and_generate_text(audio_path):
    # Load and preprocess audio
    audio_input, sample_rate = sf.read(audio_path)
    audio_input = librosa.resample(audio_input.T, sample_rate, 16000)

    # Ensure audio is mono
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)  # Convert to mono by averaging channels

    # Preprocess audio for the ASR transcription model
    inputs = asr_transcription_processor(
        audio_input,
        return_tensors="pt",
        sampling_rate=16000,
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Obtain initial ASR transcription
    with torch.no_grad():
        logits = asr_transcription_model(input_values, attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    initial_transcription = asr_transcription_processor.decode(predicted_ids[0])

    # Create a prompt for the LLM decoder
    prompt = f"以下の音声認識結果を修正してください: {initial_transcription}"
    # Translation: "Please correct the following speech recognition result:"

    # Tokenize the prompt for decoder input IDs
    decoder_input_ids = llm_tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).input_ids.to(device)

    # Get ASR encoder outputs for the audio features
    input_values = asr_processor(
        audio_input,
        return_tensors="pt",
        sampling_rate=16000,
        padding=True,
    ).input_values.to(device)

    with torch.no_grad():
        asr_encoder_outputs = asr_encoder(input_values)
        encoder_hidden_states = asr_encoder_outputs.last_hidden_state

    # Project encoder outputs
    projected_hidden_states = projector(encoder_hidden_states)

    # Prepare encoder outputs for decoder
    encoder_outputs = BaseModelOutput(last_hidden_state=projected_hidden_states)

    # Generate corrected transcription
    with torch.no_grad():
        outputs = llm_decoder.generate(
            input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            max_length=100,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    generated_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


# Gradio Interface
def transcribe(audio):
    if audio is None:
        return (
            "音声ファイルをアップロードしてください。"  # "Please upload an audio file."
        )
    # Save the audio file temporarily
    audio_path = "temp_audio.wav"
    sf.write(audio_path, audio[1], audio[0])
    # Process audio and generate text
    output_text = process_audio_and_generate_text(audio_path)
    # Remove the temporary audio file
    os.remove(audio_path)
    return output_text


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="numpy",
        label="日本語の音声をアップロードしてください",
    ),
    outputs=gr.Textbox(label="生成されたテキスト"),
    title="日本語音声認識とテキスト生成デモ",
    description="音声ファイルをアップロードすると、音声認識とテキスト生成を行います。",
)

if __name__ == "__main__":
    iface.launch()
