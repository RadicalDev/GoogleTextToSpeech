__author__ = 'jfindley'
from gevent import monkey
monkey.patch_all()

from gevent.pool import Pool
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer
from auditok.io import BufferAudioSource
from retrying import retry
from types import IntType, StringType
from contextlib import closing

import wave
import os
import requests
import json
import gevent
import time
import httplib2
import base64
import argparse
import sys

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

SERVICE_URL = 'https://www.googleapis.com/auth/cloud-platform'


def retry_main(exception):
    print "Connection error to google: ", exception
    return True


class Google(object):
    def __init__(self,
                 discovery_file_path,
                 max_threads=1,
                 quota=5,
                 max_continuous_silence=0,
                 min_segment_length=0.01,
                 max_segment_length=20,
                 output_mode='formatted',
                 quiet=False):

        self.discovery_file = discovery_file_path
        self.max_threads = max_threads
        self.divisor = 2
        self.max_continuous_silence=max_continuous_silence
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.output_mode = output_mode
        self.quiet=quiet

        self.quota = quota
        self.pools = []

    def split_to_mono(self, stereo_path):

        with closing(wave.open(stereo_path, 'r')) as wf:
            try:
                nframes = wf.getnframes()
                nchannels = wf.getnchannels()
                framerate = wf.getframerate()
                sample_width = wf.getsampwidth()
                audio_duration = nframes / float(framerate)
                frame_size = wf._framesize
                bit_align = nchannels * sample_width
                frames = wf.readframes(nframes)
            except wave.Error:
                raise Exception("Invalid audio format")

        if not self.quiet:
            print "Audio parameters: Frames: {0}, Channels: {1}, Rate: {2}, Width: {3}, Duration: {4}".format(nframes, nchannels, framerate, sample_width, audio_duration)

        assert nchannels == 2, "This function takes stereo SLIN16 PCM audio only"
        assert (nframes % 2) == 0, "Number of frames should be even"
        assert framerate in [16000, 8000], "16k or 8k sample rates should be used"

        frames_per_channel = [frames[x - 2:x] for x in xrange(2, len(frames) + 1, 2)]
        left, right = ["".join(y) for y in [frames_per_channel[offset::nchannels] for offset in xrange(nchannels)]]

        assert len(left) == len(right), "Left and right channels should be the same length"

        left = BufferAudioSource(left, sampling_rate=framerate, sample_width=sample_width, channels=1)
        right = BufferAudioSource(right, sampling_rate=framerate, sample_width=sample_width, channels=1)

        return {
            'nframes': nframes,
            'nchannels': nchannels,
            'frame_rate': framerate,
            'frame_width': sample_width,
            'bit_aligh': bit_align,
            'duration': audio_duration,
            'a_leg': left,
            'b_leg': right,
        }

    @retry(retry_on_exception=retry_main, stop_max_attempt_number=3, wait_random_min=1000, wait_random_max=2000)
    def upload_audio(self, speech, sample_rate):
        credentials = GoogleCredentials.get_application_default().create_scoped([SERVICE_URL])

        with open(self.discovery_file, 'r') as f:
            doc = f.read()

        speech_content = base64.b64encode(speech)
        service = discovery.build_from_document(doc, credentials=credentials, http=httplib2.Http())
        service_request = service.speech().recognize(
            body={
                'initialRequest': {
                    'encoding': 'LINEAR16',
                    'sampleRate': sample_rate
                },
                'audioRequest': {
                    'content': speech_content.decode('UTF-8')
                    }
                })
        response = service_request.execute()
        return json.dumps(response)

    def upload_audio_for_result(self, raw_audio_data, **kwargs):

        raw_audio_data = "".join(raw_audio_data)
        try:
            r = self.upload_audio(raw_audio_data, kwargs['sample_rate'])
            raw_audio_data = ""
        except requests.ConnectionError:
            return "%UPLOADFAILED"

        try:
            json_response = json.loads(r)
            if not json_response:
                # Received empty response
                return "%INAUDIBLE"

            results = json_response['responses'][0]['results'][0]['alternatives'][0]['transcript']

            if not results:
                # Received empty transcript
                return "%INAUDIBLE"
        except Exception, e:
            print "ERROR: ", e
            # Exception occurred on this segment
            return "%INAUDIBLE"

        return results

    def transcribe(self, channel, data, start, end, **kwargs):
        assert (isinstance(self.divisor, IntType) and self.divisor % 2 == 0 and self.divisor > 0), "Divisor Must be non-zero even integer"
        divisor = float(self.divisor)
        request_start = time.time()
        result = self.upload_audio_for_result(data, **kwargs)
        request_end = time.time()
        data = ""

        return [(request_end-request_start, channel, start / divisor, end / divisor, result)]

    def greenlet(self, segment, sample_rate, sample_width, channels):
        results = []

        # Start time of request for rate limiting
        start = time.time()
        for time_taken, channel, segment_start, segment_end, transcript in self.transcribe(*segment, sample_rate=sample_rate, sample_width=sample_width, channels=channels):
            # Completion time of request for rate limiting
            end = time.time()

            # Sleep if it's too soon to make another request
            if end-start < self.quota:
                gevent.sleep(end-start)

            start = time.time()

            args = {
                'time_taken': time_taken,
                'channel': channel,
                'start': segment_start,
                'end': segment_end,
                'transcript': transcript
            }

            results.append(args)
        return results

    def batch(self, segments, audio_duration, sample_rate, sample_width, channels):
        start_time = time.time()
        batch_pool = Pool(self.max_threads)
        results = []
        segments_processed = 0
        segments_to_process = len(segments)
        stop = False
        while not stop:
            for _ in xrange(self.max_threads):
                segment_index = segments_processed
                if segment_index >= segments_to_process:
                    stop = True
                    break
                results.append(batch_pool.spawn(self.greenlet, segments[segment_index], sample_rate, sample_width, channels))
                segments_processed += 1
            batch_pool.join()

            for result in results:
                result = result.get()[0]
                if self.output_mode == 'csv':
                    print "{time_taken:.2f},{channel},{start:.1f},{end:.1f},{transcript}".format(**result)
                elif self.output_mode == 'json':
                    print json.dumps(result)
                else:
                    print '[T: {time_taken:<5.2f} - Channel {channel}] {start:<6.1f} - {end:<6.1f}: {transcript}'.format(**result)


            results = []
        total_processing_time = time.time() - start_time
        if not self.quiet:
            print "Total processing time: {0} seconds, recording duration: {1} seconds".format(total_processing_time, audio_duration)

    def transcribe_audio(self, stereo_path, channels_to_process):

        if not os.path.isfile(stereo_path):
            raise Exception("Audio file does not exist.")

        data = self.split_to_mono(stereo_path)

        a_leg = data['a_leg']
        b_leg = data['b_leg']

        data['a_leg'] = None
        data['b_leg'] = None

        validator = AudioEnergyValidator(sample_width=data['frame_width'], energy_threshold=45)
        trimmer = StreamTokenizer(validator,
                                  min_length=self.min_segment_length,
                                  max_length=self.max_segment_length,
                                  max_continuous_silence=self.max_continuous_silence,
                                  mode=StreamTokenizer.DROP_TAILING_SILENCE)

        segments = []
        if channels_to_process in ['A', 'AB']:
            a_source = ADSFactory.ads(audio_source=a_leg, record=True, block_size=data['frame_rate'] / self.divisor)
            a_source.open()
            trimmer.tokenize(a_source, callback=lambda data, start, end: segments.append(("A", data, start, end)))

        if channels_to_process in ['B', 'AB']:
            b_source = ADSFactory.ads(audio_source=b_leg, record=True, block_size=data['frame_rate'] / self.divisor)
            b_source.open()
            trimmer.tokenize(b_source, callback=lambda data, start, end: segments.append(("B", data, start, end)))

        segments = sorted(segments, key=lambda x: x[3])
        self.batch(segments, data['duration'], data['frame_rate'], data['frame_width'], data['nchannels'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--application-credentials-file', type=StringType, help="Path to API credentials file. Required")
    parser.add_argument('-d', '--api-discovery-file', type=StringType, help="Path to API discovery file. Required")
    parser.add_argument('-f', '--file', type=StringType, help="Path to audio file. Required")
    parser.add_argument('-c', '--channels-to-process', default='AB', choices=['A', 'B', 'AB'], help="Channels to process: [A, B, AB]")
    parser.add_argument('-t', '--threads', type=IntType, default=1, help="Number of threads to use when processing segments in parallel.")
    parser.add_argument('-p', '--request-period', type=IntType, default=100, help="Request rate period e.g., 100 seconds")
    parser.add_argument('-r', '--requests-per-period', type=IntType, default=20, help="Requests allowed per period e.g., 20 per 100 seconds")
    parser.add_argument('-s', '--max-continuous-silence', type=IntType, default=0, help="Maximum allowable silence duration before a segment is terminated")
    parser.add_argument('-m', '--min-segment-length', type=IntType, default=0.01, help="Minimum segment duration. Larger values include more silence")
    parser.add_argument('-M', '--max-segment-length', type=IntType, default=20, help="Maximum segment duration. Smaller yields better timestamps, but can interfere with recognition")
    parser.add_argument('-o', '--output_mode', default='formatted', choices=['formatted', 'csv', 'json'], help="Output mode. formatted or csv")
    parser.add_argument('-q', '--quiet', action='store_true', help="Suppress metadata, implicit if output_mode is csv or json")

    args = parser.parse_args()

    working_directory = os.getcwd()

    if args.output_mode in ['csv', 'json']:
        args.quiet = True

    if not args.file:
        parser.print_help()
        print "ERROR: No path to audio file provided"
        sys.exit(1)

    if not args.application_credentials_file:
        if not args.quiet:
            print "Looking for speech_service_account.json in {0}".format(working_directory)
        args.application_credentials_file = os.path.join(working_directory, "speech_service_account.json")

    if not args.api_discovery_file:
        if not args.quiet:
            print "Looking for speech-discovery_google_rest_v1.json in {0}".format(working_directory)
        args.api_discovery_file = os.path.join(working_directory, 'speech-discovery_google_rest_v1.json')

    if not os.path.exists(args.application_credentials_file):
        print "Path to credentials file is invalid"
        sys.exit(1)

    if not os.path.exists(args.api_discovery_file):
        print "Path to discovery file is invalid"
        sys.exit(1)

    if not os.path.exists(args.file):
        print "Path to audio file is invalid"
        sys.exit(1)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.application_credentials_file

    quota = args.request_period/args.requests_per_period
    if not args.quiet:
        print "Processing {0}: Channels - {1}, Threads: {2}, Rate: 1 request every {3} seconds".format(args.file, args.channels_to_process, args.threads, quota)

    google = Google(args.api_discovery_file,
                    max_threads=args.threads,
                    quota=quota,
                    max_continuous_silence=args.max_continuous_silence,
                    min_segment_length=args.min_segment_length,
                    max_segment_length=args.max_segment_length,
                    output_mode=args.output_mode,
                    quiet=args.quiet)
    try:
        google.transcribe_audio(args.file, args.channels_to_process)
    except Exception as e:
        print "ERROR: ", e
