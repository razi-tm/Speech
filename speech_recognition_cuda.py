import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

import time
from pathlib import Path

import IPython
import sentencepiece as spm
from torchaudio.models.decoder import cuda_ctc_decoder
from torchaudio.utils import download_asset



def download_asset_external(url, key):
    path = Path(torch.hub.get_dir()) / "torchaudio" / Path(key)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, path)
    return str(path)


url_prefix = "https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01"
model_link = f"{url_prefix}/resolve/main/exp/cpu_jit.pt"
model_path = download_asset_external(model_link, "cuda_ctc_decoder/cpu_jit.pt")



speech_file = download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")
waveform, sample_rate = torchaudio.load(speech_file)
assert sample_rate == 16000
IPython.display.Audio(speech_file)


bpe_link = f"{url_prefix}/resolve/main/data/lang_bpe_500/bpe.model"
bpe_path = download_asset_external(bpe_link, "cuda_ctc_decoder/bpe.model")

bpe_model = spm.SentencePieceProcessor()
bpe_model.load(bpe_path)
tokens = [bpe_model.id_to_piece(id) for id in range(bpe_model.get_piece_size())]
print(tokens)



cuda_decoder = cuda_ctc_decoder(tokens, nbest=10, beam_size=10, blank_skip_threshold=0.95)



actual_transcript = "i really was very much afraid of showing him how much shocked i was at some parts of what he said"
actual_transcript = actual_transcript.split()

device = torch.device("cuda", 0)
acoustic_model = torch.jit.load(model_path)
acoustic_model.to(device)
acoustic_model.eval()

waveform = waveform.to(device)

feat = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, snip_edges=False)
feat = feat.unsqueeze(0)
feat_lens = torch.tensor(feat.size(1), device=device).unsqueeze(0)

encoder_out, encoder_out_lens = acoustic_model.encoder(feat, feat_lens)
nnet_output = acoustic_model.ctc_output(encoder_out)
log_prob = torch.nn.functional.log_softmax(nnet_output, -1)

print(f"The shape of log_prob: {log_prob.shape}, the shape of encoder_out_lens: {encoder_out_lens.shape}")

results = cuda_decoder(log_prob, encoder_out_lens.to(torch.int32))
beam_search_transcript = bpe_model.decode(results[0][0].tokens).lower()
beam_search_wer = torchaudio.functional.edit_distance(actual_transcript, beam_search_transcript.split()) / len(
    actual_transcript
)

print(f"Transcript: {beam_search_transcript}")
print(f"WER: {beam_search_wer}")



def print_decoded(cuda_decoder, bpe_model, log_prob, encoder_out_lens, param, param_value):
    start_time = time.monotonic()
    results = cuda_decoder(log_prob, encoder_out_lens.to(torch.int32))
    decode_time = time.monotonic() - start_time
    transcript = bpe_model.decode(results[0][0].tokens).lower()
    score = results[0][0].score
    print(f"{param} {param_value:<3}: {transcript} (score: {score:.2f}; {decode_time:.4f} secs)")



for i in range(10):
    transcript = bpe_model.decode(results[0][i].tokens).lower()
    score = results[0][i].score
    print(f"{transcript} (score: {score})")



beam_sizes = [1, 2, 3, 10]

for beam_size in beam_sizes:
    beam_search_decoder = cuda_ctc_decoder(
        tokens,
        nbest=1,
        beam_size=beam_size,
        blank_skip_threshold=0.95,
    )
    print_decoded(beam_search_decoder, bpe_model, log_prob, encoder_out_lens, "beam size", beam_size)



blank_skip_probs = [0.25, 0.95, 1.0]

for blank_skip_prob in blank_skip_probs:
    beam_search_decoder = cuda_ctc_decoder(
        tokens,
        nbest=10,
        beam_size=10,
        blank_skip_threshold=blank_skip_prob,
    )
    print_decoded(beam_search_decoder, bpe_model, log_prob, encoder_out_lens, "blank_skip_threshold", blank_skip_prob)

del cuda_decoder