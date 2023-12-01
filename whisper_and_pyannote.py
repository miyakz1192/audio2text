from pyannote.audio import Pipeline
from pyannote.audio import Audio
from scipy.io.wavfile import write
import numpy as np
import torchaudio
import sys
import os
import whisper
import tempfile

# I refered these sites
# https://qiita.com/sayo0127/items/e22fdc229d2dfd879f75
# https://stackoverflow.com/questions/10357992/how-to-generate-audio-from-a-numpy-array
# https://take-tech-engineer.com/torchaudio-save/


class WhisperAndPyannote:
    def __init__(self):
        pretrained = "pyannote/speaker-diarization"
        YOUR_TOKEN = os.environ["LLM_SVC_PYANNOTE_AUTH_TOKEN"]

        self.model = whisper.load_model("large")
        self.pipeline = Pipeline.from_pretrained(pretrained,
                                                 use_auth_token=YOUR_TOKEN)

    def _audio_crop(self, audio, audio_file_path, segment):
        waveform = None
        sample_rate = None
        try:
            waveform, sample_rate = audio.crop(audio_file_path, segment)
        except Exception as e:
            print("Error on audio.crop {e}", e)

        return waveform, sample_rate

    # output is these tuple array
    # (speaker name(str), text(str), start sec(float), end sec(float))
    def analyze(self, audio_file_path):
        diarization = self.pipeline(audio_file_path)
        audio = Audio(sample_rate=16000, mono=True)
        res = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # waveform, sample_rate = audio.crop(audio_file_path, segment)
            waveform, sample_rate = self._audio_crop(audio, 
                                                     audio_file_path, segment)
            if waveform is None:
                continue
            text = self._get_text_from_segment(waveform, sample_rate)
            rec = (speaker, text, segment.start, segment.end)
            print(f"[{segment.start:03.1f}s - {segment.end:03.1f}s] \
                    {speaker}: {text}", flush=True)
            res.append(rec)

        return res

    # waveform is one segment of audio
    def _get_text_from_segment(self, waveform, sample_rate):
        text = "INVALID"
        with tempfile.NamedTemporaryFile(delete=True, dir='/tmp/',
                                         prefix='temp_',
                                         suffix='.wav') as temp_file:
            temp_file_path = temp_file.name
            print(f"Temporary file created: {temp_file_path}")
            torchaudio.save(uri=temp_file_path, src=waveform,
                            sample_rate=sample_rate)
            print("Data written to the temporary file.", flush=True)
            text = self.model.transcribe(temp_file_path)["text"]
            print("transcribe done", flush=True)

        return text

# 上のアルゴリズムだと単純なので、単にmodelに推論させたときよりも、精度が悪いっぽい
# [0.8s - 2.5s] SPEAKER_00: あ、~~ちゃんこんにちは
# [3.0s - 4.3s] SPEAKER_02: Xです
# [4.7s - 6.0s] SPEAKER_00: Yです
# [7.4s - 8.8s] SPEAKER_00: 好きな食べ物は何ですか
# [9.2s - 10.0s] SPEAKER_01: 英語です
# [10.5s - 10.8s] SPEAKER_00: 英國?
# [11.5s - 16.0s] SPEAKER_00: じゃあ一緒にリンゴって同時に言おうせーのリンゴ
# [14.6s - 16.0s] SPEAKER_01: りんご

# やっぱり、それぞれのsegment毎に単発でmodelに推論させると、
# 前の出力結果が参照できないので、精度が悪くなるのだろう
# だから、次のSPEAKERの出力を得るために、
# 前のwavを全部くっつけてmodelに入力させ、
# その結果から、前得た文字列を取っ払えば、今回得たいものが得られるはず。
