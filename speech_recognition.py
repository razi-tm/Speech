import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


import time
from typing import List

import IPython
import matplotlib.pyplot as plt
from torchaudio.models.decoder import ctc_decoder
from torchaudio.utils import download_asset


bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
acoustic_model = bundle.get_model()


speech_file = download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")

IPython.display.Audio(speech_file)



waveform, sample_rate = torchaudio.load(speech_file)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)



tokens = [label.lower() for label in bundle.get_labels()]
print(tokens)



from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState


class CustomLM(CTCDecoderLM):
    """Create a Python wrapper around `language_model` to feed to the decoder."""

    def __init__(self, language_model: torch.nn.Module):
        CTCDecoderLM.__init__(self)
        self.language_model = language_model
        self.sil = -1  # index for silent token in the language model
        self.states = {}

        language_model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        with torch.no_grad():
            score = self.language_model(self.sil)

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if outstate not in self.states:
            score = self.language_model(token_index)
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.sil)




from torchaudio.models.decoder import download_pretrained_files

files = download_pretrained_files("librispeech-4-gram")

print(files)



LM_WEIGHT = 3.23
WORD_SCORE = -0.26

beam_search_decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=files.tokens,
    lm=files.lm,
    nbest=3,
    beam_size=1500,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
)




class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()


greedy_decoder = GreedyCTCDecoder(tokens)



actual_transcript = "i really was very much afraid of showing him how much shocked i was at some parts of what he said"
actual_transcript = actual_transcript.split()

emission, _ = acoustic_model(waveform)


greedy_result = greedy_decoder(emission[0])
greedy_transcript = " ".join(greedy_result)
greedy_wer = torchaudio.functional.edit_distance(actual_transcript, greedy_result) / len(actual_transcript)

print(f"Transcript: {greedy_transcript}")
print(f"WER: {greedy_wer}")



beam_search_result = beam_search_decoder(emission)
beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
beam_search_wer = torchaudio.functional.edit_distance(actual_transcript, beam_search_result[0][0].words) / len(
    actual_transcript
)

print(f"Transcript: {beam_search_transcript}")
print(f"WER: {beam_search_wer}")





tokens_str = "".join(beam_search_decoder.idxs_to_tokens(beam_search_result[0][0].tokens))
transcript = " ".join(tokens_str.split("|"))


beam_search_decoder.decode_begin()




for t in range(emission.size(1)):
    beam_search_decoder.decode_step(emission[0, t:t + 1, :])





beam_search_decoder.decode_end()
beam_search_result_inc = beam_search_decoder.get_final_hypothesis()




beam_search_transcript_inc = " ".join(beam_search_result_inc[0].words).strip()
beam_search_wer_inc = torchaudio.functional.edit_distance(
    actual_transcript, beam_search_result_inc[0].words) / len(actual_transcript)

print(f"Transcript: {beam_search_transcript_inc}")
print(f"WER: {beam_search_wer_inc}")

assert beam_search_result[0][0].words == beam_search_result_inc[0].words
assert beam_search_result[0][0].score == beam_search_result_inc[0].score
torch.testing.assert_close(beam_search_result[0][0].timesteps, beam_search_result_inc[0].timesteps)





timesteps = beam_search_result[0][0].timesteps
predicted_tokens = beam_search_decoder.idxs_to_tokens(beam_search_result[0][0].tokens)

print(predicted_tokens, len(predicted_tokens))
print(timesteps, timesteps.shape[0])







def plot_alignments(waveform, emission, tokens, timesteps, sample_rate):

    t = torch.arange(waveform.size(0)) / sample_rate
    ratio = waveform.size(0) / emission.size(1) / sample_rate

    chars = []
    words = []
    word_start = None
    for token, timestep in zip(tokens, timesteps * ratio):
        if token == "|":
            if word_start is not None:
                words.append((word_start, timestep))
            word_start = None
        else:
            chars.append((token, timestep))
            if word_start is None:
                word_start = timestep

    fig, axes = plt.subplots(3, 1)

    def _plot(ax, xlim):
        ax.plot(t, waveform)
        for token, timestep in chars:
            ax.annotate(token.upper(), (timestep, 0.5))
        for word_start, word_end in words:
            ax.axvspan(word_start, word_end, alpha=0.1, color="red")
        ax.set_ylim(-0.6, 0.7)
        ax.set_yticks([0])
        ax.grid(True, axis="y")
        ax.set_xlim(xlim)

    _plot(axes[0], (0.3, 2.5))
    _plot(axes[1], (2.5, 4.7))
    _plot(axes[2], (4.7, 6.9))
    axes[2].set_xlabel("time (sec)")
    fig.tight_layout()


plot_alignments(waveform[0], emission, predicted_tokens, timesteps, bundle.sample_rate)






def print_decoded(decoder, emission, param, param_value):
    start_time = time.monotonic()
    result = decoder(emission)
    decode_time = time.monotonic() - start_time

    transcript = " ".join(result[0][0].words).lower().strip()
    score = result[0][0].score
    print(f"{param} {param_value:<3}: {transcript} (score: {score:.2f}; {decode_time:.4f} secs)")






for i in range(3):
    transcript = " ".join(beam_search_result[0][i].words).strip()
    score = beam_search_result[0][i].score
    print(f"{transcript} (score: {score})")







beam_sizes = [1, 5, 50, 500]

for beam_size in beam_sizes:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        beam_size=beam_size,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam size", beam_size)







num_tokens = len(tokens)
beam_size_tokens = [1, 5, 10, num_tokens]

for beam_size_token in beam_size_tokens:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        beam_size_token=beam_size_token,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam size token", beam_size_token)







beam_thresholds = [1, 5, 10, 25]

for beam_threshold in beam_thresholds:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        beam_threshold=beam_threshold,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam threshold", beam_threshold)





lm_weights = [0, LM_WEIGHT, 15]

for lm_weight in lm_weights:
    beam_search_decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        lm_weight=lm_weight,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "lm weight", lm_weight)