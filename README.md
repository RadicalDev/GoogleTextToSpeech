# GoogleTextToSpeech
Google Speech API Transcriber

    Transcribes a stereo 8k or 16k wav file via the google alpha speech API. 
    Prints results broken down by channel with  speech segment timestamps. 
    
Operation:
    The stereo file is inspected, then split into independent channels in memory, then auditok tokenizes each channel based on 
    values lower than a given threshold indicating silence. Each segment is sent to google (preferably in parallel when their limits are better),     and the responses are collected and sorted according to the time in which they occurred. 
    
    It is not safe to adjust the thread count, yet, due to the very low limits set by Google of 0.2 requests per second.

    Returns results broken down by channel with  speech segment timestamps. 

Requires:
  auditok
  gevent
  requests
  retrying
  oauth2client
  google-api-python-client
  googleapis-common-protos

Note:
	if speech_service_account.json and speech-discovery_google_rest_v1.json are in the current directory when this script
	is executed, the -a and -d flags don't need to be specified. 

Examples:
  python transcribe.py -f audio.wav      -> Get results for both channels
  python transcribe.py -f audio.wav -c A -> Get results just for the left channel
  python transcribe.py -f audio.wav -c B -> Get results just for the right channel
