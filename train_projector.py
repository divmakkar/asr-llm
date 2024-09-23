import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import soundfile as sf
import time
from torch.optim import AdamW
import librosa

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ASR encoder (used for audio features)
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
asr_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
asr_encoder.to(device)

asr_transcription_model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
# Load ASR transcription model (used to get initial transcription)
asr_transcription_processor = Wav2Vec2Processor.from_pretrained(
    asr_transcription_model_id
)
asr_transcription_model = Wav2Vec2ForCTC.from_pretrained(asr_transcription_model_id)
asr_transcription_model.to(device)

# Load LLM decoder (Japanese T5 model)
llm_tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", legacy=True)
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

# Freeze ASR encoder
for param in asr_encoder.parameters():
    param.requires_grad = False

# Freeze ASR transcription model
for param in asr_transcription_model.parameters():
    param.requires_grad = True

# Ensure projector parameters are trainable
for param in projector.parameters():
    param.requires_grad = True

# Ensure LLM decoder parameters are trainable
for param in llm_decoder.parameters():
    param.requires_grad = True

# Load the Common Voice dataset for Japanese
dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "ja",
    split="train+validation",
    token="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",
)


# Define a custom dataset class
class ASRCorrectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sampling_rate = asr_transcription_processor.feature_extractor.sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Get audio and correct transcript
        audio = item["audio"]["array"]
        audio = librosa.resample(
            audio, orig_sr=item["audio"]["sampling_rate"], target_sr=self.sampling_rate
        )
        correct_transcript = item["sentence"]
        return audio, correct_transcript


# Instantiate the dataset and dataloader
asr_dataset = ASRCorrectionDataset(dataset)

# Define batch size
batch_size = 8  # Adjust based on GPU memory


# Collate function
def collate_fn(batch):
    audios = [item[0] for item in batch]
    correct_transcripts = [item[1] for item in batch]
    # Preprocess audio
    inputs = asr_transcription_processor(
        audios,
        return_tensors="pt",
        sampling_rate=asr_transcription_processor.feature_extractor.sampling_rate,
        padding=True,
    )
    input_values = inputs.input_values
    attention_mask = inputs.attention_mask
    return input_values, attention_mask, correct_transcripts


dataloader = DataLoader(
    asr_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=7,  # Adjust based on your CPU cores
)


def train_model(
    asr_encoder,
    asr_transcription_model,
    projector,
    llm_decoder,
    asr_transcription_processor,
    llm_tokenizer,
    dataloader,
    num_epochs=3,
    save_path="trained_model.pth",
):

    # Define optimizer
    optimizer = AdamW(
        list(projector.parameters())
        + list(llm_decoder.parameters())
        + list(asr_transcription_model.parameters())
        + list(asr_encoder.parameters()),
        lr=1e-5,
    )

    # Set models to training mode
    projector.train()
    llm_decoder.train()

    # ASR models are in eval mode
    asr_encoder.eval()
    asr_transcription_model.eval()

    total_steps = len(dataloader) * num_epochs
    print_every = 100  # Print loss every N batches

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (
            input_values,
            attention_masks,
            correct_transcripts,
        ) in enumerate(dataloader):
            input_values = input_values.to(device)
            attention_masks = attention_masks.to(device)

            # Obtain initial ASR transcription
            with torch.no_grad():
                logits = asr_transcription_model(input_values, attention_masks).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentences = asr_transcription_processor.batch_decode(
                predicted_ids
            )
            initial_transcriptions = predicted_sentences

            # Create prompts
            prompts = [
                f"以下の音声認識結果を修正してください: {trans}"
                for trans in initial_transcriptions
            ]

            # Tokenize decoder input IDs
            decoder_input_ids = llm_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(device)

            # Get ASR encoder outputs
            with torch.no_grad():
                asr_encoder_outputs = asr_encoder(input_values)
                encoder_hidden_states = asr_encoder_outputs.last_hidden_state

            # Project encoder outputs
            projected_hidden_states = projector(encoder_hidden_states)

            # Prepare encoder outputs
            encoder_outputs = BaseModelOutput(last_hidden_state=projected_hidden_states)

            # Tokenize correct transcripts as target output
            labels = llm_tokenizer(
                correct_transcripts,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(device)

            # Replace pad token ids with -100 to ignore in loss
            labels[labels == llm_tokenizer.pad_token_id] = -100

            # Forward pass
            outputs = llm_decoder(
                input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                labels=labels,
                return_dict=True,
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % print_every == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / (batch_idx + 1)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                    f"Loss: {avg_loss:.4f}, Time Elapsed: {elapsed:.2f}s"
                )

        # Save checkpoint after each epoch
        torch.save(
            {
                "projector_state_dict": projector.state_dict(),
                "llm_decoder_state_dict": llm_decoder.state_dict(),
                "asr_transcription_model_state_dict": asr_transcription_model.state_dict(),
                "asr_encoder_state_dict": asr_encoder.state_dict(),
            },
            f"model_epoch{epoch+1}.pth",
        )
        print(f"Checkpoint saved for epoch {epoch+1}")

    # Save the final trained model
    torch.save(
        {
            "projector_state_dict": projector.state_dict(),
            "llm_decoder_state_dict": llm_decoder.state_dict(),
            "asr_transcription_model_state_dict": asr_transcription_model.state_dict(),
            "asr_encoder_state_dict": asr_encoder.state_dict(),
        },
        save_path,
    )
    print(f"Training completed. Model saved to {save_path}")

    # Set models back to evaluation mode
    projector.eval()
    llm_decoder.eval()


if __name__ == "__main__":
    # Start training
    train_model(
        asr_encoder,
        asr_transcription_model,
        projector,
        llm_decoder,
        asr_transcription_processor,
        llm_tokenizer,
        dataloader,
        num_epochs=3,  # Adjust as needed
        save_path="trained_model.pth",
    )
