#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import tempfile
from whisper_and_pyannote import *
import os
import traceback

# settting of import messaging
sys.path.append("messaging")
from messaging import *


class Audio2TextService:
    def __init__(self):
        self.wap = WhisperAndPyannote()

    def main_loop(self):
        while True:
            try:
                self.unit_work()
            except Exception as e:
                print(f"An error occurred while unit work: {e}")
                traceback.print_exc()

            time.sleep(10)

    def make_temp_wavfile(self, a2t_rec):
        # a2t_rec is an Audio2TextRecord object
        with tempfile.NamedTemporaryFile(delete=False, dir='/tmp/',
                                         prefix='temp_',
                                         suffix='.bin') as temp_file:
            temp_file_path = temp_file.name

            # バイト型のデータをファイルに書き込む
            with open(temp_file_path, 'wb') as file:
                file.write(a2t_rec.raw_audio_byte)

        return temp_file_path

    def unit_work(self):
        print("Getting new req from queue")
        rec = Audio2TextServiceReqMessaging().connect_and_basic_get_record()
        if rec is None:
            return

        temp_audio_file_path = self.make_temp_wavfile(rec)
        print(self.wap.analyze(temp_audio_file_path))
        os.remove(temp_audio_file_path)


#print(WhisperAndPyannote().analyze("../rec1.wav"))
Audio2TextService().main_loop()