y and engagement, 8: subscribe for updates
2025-05-12 20:50:05,708 - INFO - __main__ - Creating YouTube Short
2025-05-12 20:50:05,709 - INFO - helper.minor_helper - Starting YouTube short creation
2025-05-12 20:50:05,709 - INFO - automation.shorts_maker_V - Scaling durations by factor 0.74 to fit max duration of 25s
2025-05-12 20:50:05,709 - INFO - automation.shorts_maker_V - Starting parallel execution of major steps
2025-05-12 20:50:05,710 - INFO - automation.parallel_tasks - Starting task: generate_text_clips
2025-05-12 20:50:05,711 - INFO - automation.shorts_maker_V - Generating text clips in parallel
2025-05-12 20:50:05,711 - INFO - automation.parallel_tasks - Starting task: generate_audio
2025-05-12 20:50:05,711 - INFO - helper.text - Generating 8 text clips in parallel
2025-05-12 20:50:05,712 - INFO - helper.text - Using ThreadPoolExecutor with dill for text generation
2025-05-12 20:50:05,712 - INFO - automation.shorts_maker_V - Generating audio clips in parallel
2025-05-12 20:50:05,712 - INFO - automation.parallel_tasks - Starting task: fetch_videos
2025-05-12 20:50:05,713 - INFO - helper.audio - Generating 8 audio clips in parallel
2025-05-12 20:50:05,733 - INFO - automation.shorts_maker_V - Fetching background videos in parallel
2025-05-12 20:50:05,740 - INFO - helper.fetch - Fetching videos for 8 queries in parallel
2025-05-12 20:50:05,746 - INFO - helper.fetch - Fetching 2 videos using pixabay API
2025-05-12 20:50:05,761 - INFO - helper.fetch - Fetching 2 videos using pixabay API
2025-05-12 20:50:05,788 - INFO - helper.fetch - Fetching 2 videos using pixabay API
2025-05-12 20:50:05,809 - INFO - helper.fetch - Fetching 2 videos using pexels API
2025-05-12 20:50:05,866 - INFO - helper.fetch - Fetching 2 videos using pexels API
2025-05-12 20:50:05,887 - INFO - helper.fetch - Fetching 2 videos using pixabay API
2025-05-12 20:50:05,910 - INFO - helper.fetch - Fetching 2 videos using pexels API
2025-05-12 20:50:05,964 - INFO - helper.fetch - Fetching 2 videos using pixabay API
2025-05-12 20:50:05,993 - ERROR - helper.text - Error creating text clip: y1 must be greater than or equal to y0
2025-05-12 20:50:06,295 - INFO - automation.voiceover - Speech synthesized for text [This could change ho...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_5_1747063205.mp3]
2025-05-12 20:50:06,301 - INFO - automation.voiceover - Speech synthesized for text [Curious to know more...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_6_1747063205.mp3]
2025-05-12 20:50:06,407 - INFO - automation.voiceover - Speech synthesized for text [Big news for Apple f...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_1_1747063205.mp3]
2025-05-12 20:50:06,421 - INFO - automation.voiceover - Speech synthesized for text [A product renaissanc...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_2_1747063205.mp3]
2025-05-12 20:50:06,452 - INFO - automation.voiceover - Speech synthesized for text [Hit that subscribe b...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_7_1747063205.mp3]
2025-05-12 20:50:06,517 - INFO - automation.voiceover - Speech synthesized for text [Expect groundbreakin...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_3_1747063205.mp3]
2025-05-12 20:50:06,644 - INFO - automation.voiceover - Speech synthesized for text [Rumors suggest new a...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_4_1747063205.mp3]
2025-05-12 20:50:06,646 - INFO - automation.voiceover - Speech synthesized for text [LazyCreator presents...] and saved to [D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194\audio_section_0_1747063205.mp3]
2025-05-12 20:50:06,716 - INFO - helper.audio - Generated 8 audio clips in 1.00 seconds
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2024-02-12T20:43:46.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 8164, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2024-02-12T20:43:46.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 16.67, 'bitrate': 8167, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 8164, 'video_fps': 30.0, 'video_duration': 16.67, 'video_n_frames': 500}
ffmpeg -i automation\temp\video_downloads\pixabay_200281.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'iso5', 'minor_version': '512', 'compatible_brands': 'iso5iso6mp41', 'encoder': 'Lavf60.16.100'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': 'eng', 'default': True, 'size': [360, 640], 'bitrate': 719, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'handler_name': '?Mainconcept Video Media Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'Lavc60.31.102 libx264'}}], 'input_number': 0}], 'duration': 1.67, 'bitrate': 725, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [360, 640], 'video_bitrate': 719, 'video_fps': 29.97002997002997, 'video_duration': 1.67, 'video_n_frames': 50}
ffmpeg -i automation\temp\video_downloads\pexels_31984115.mp4 -loglevel error -f image2pipe -vf scale=360:640 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:09,686 - INFO - helper.text - Generated 7 text clips in 3.98 seconds
2025-05-12 20:50:09,768 - INFO - automation.parallel_tasks - Completed task: generate_text_clips
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'iso5', 'minor_version': '512', 'compatible_brands': 'iso5iso6mp41', 'encoder': 'Lavf60.16.100'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [360, 640], 'bitrate': 562, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'handler_name': 'Core Media Video', 'vendor_id': '[0][0][0][0]', 'encoder': 'Lavc60.31.102 libx264'}}], 'input_number': 0}], 'duration': 3.43, 'bitrate': 567, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [360, 640], 'video_bitrate': 562, 'video_fps': 30.0, 'video_duration': 3.43, 'video_n_frames': 102}
ffmpeg -i automation\temp\video_downloads\pexels_31775441.mp4 -loglevel error -f image2pipe -vf scale=360:640 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-05-04T08:48:19.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [360, 640], 'bitrate': 494, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-05-04T08:48:19.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 9.0, 'bitrate': 498, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [360, 640], 'video_bitrate': 494, 'video_fps': 30.0, 'video_duration': 9.0, 'video_n_frames': 270}
ffmpeg -i automation\temp\video_downloads\pexels_7774632.mp4 -loglevel error -f image2pipe -vf scale=360:640 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-01-15T00:08:37.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [360, 640], 'bitrate': 524, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-01-15T00:08:37.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 14.96, 'bitrate': 527, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [360, 640], 'video_bitrate': 524, 'video_fps': 25.0, 'video_duration': 14.96, 'video_n_frames': 374}
ffmpeg -i automation\temp\video_downloads\pexels_6498987.mp4 -loglevel error -f image2pipe -vf scale=360:640 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-04-20T07:49:52.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [540, 960], 'bitrate': 780, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-04-20T07:49:52.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.0, 'bitrate': 783, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [540, 960], 'video_bitrate': 780, 'video_fps': 25.0, 'video_duration': 12.0, 'video_n_frames': 300}
ffmpeg -i automation\temp\video_downloads\pexels_7580118.mp4 -loglevel error -f image2pipe -vf scale=540:960 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:12,859 - INFO - helper.fetch - Fetching 2 videos using pixabay API
2025-05-12 20:50:12,947 - INFO - helper.fetch - Fetched 1 videos for query 'augmented reality glasses'
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-08-20T20:35:01.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 918, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-08-20T20:35:01.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 253, 'metadata': {'creation_time': '2021-08-20T20:35:01.000000Z', 'handler_name': 'L-SMASH Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 20.27, 'bitrate': 1176, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 918, 'video_fps': 30.0, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 253, 'video_duration': 20.27, 'video_n_frames': 608}
ffmpeg -i automation\temp\video_downloads\pixabay_85740.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2020-02-04T15:03:04.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [1080, 2048], 'bitrate': 3466, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2020-02-04T15:03:04.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 18.76, 'bitrate': 3469, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [1080, 2048], 'video_bitrate': 3466, 'video_fps': 25.0, 'video_duration': 18.76, 'video_n_frames': 469}
ffmpeg -i automation\temp\video_downloads\pexels_3682813.mp4 -loglevel error -f image2pipe -vf scale=1080:2048 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2023-08-18T01:01:50.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 11612, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2023-08-18T01:01:50.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.08, 'bitrate': 11615, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 11612, 'video_fps': 29.97002997002997, 'video_duration': 12.08, 'video_n_frames': 362}
ffmpeg -i automation\temp\video_downloads\pixabay_176489.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:15,125 - INFO - helper.fetch - Fetched 1 videos for query 'curiosity and engagement'
2025-05-12 20:50:15,372 - INFO - helper.fetch - Fetched 1 videos for query 'subscribe for updates'
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2023-08-17T00:43:10.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [1080, 1920], 'bitrate': 5729, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2023-08-17T00:43:10.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 25.0, 'bitrate': 5732, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [1080, 1920], 'video_bitrate': 5729, 'video_fps': 30.0, 'video_duration': 25.0, 'video_n_frames': 750}
ffmpeg -i automation\temp\video_downloads\pixabay_176320.mp4 -loglevel error -f image2pipe -vf scale=1080:1920 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:16,846 - INFO - automation.parallel_tasks - Completed task: generate_audio
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2017-11-07T11:22:11.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 13574, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2017-11-07T11:22:11.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 256, 'metadata': {'creation_time': '2017-11-07T11:22:11.000000Z', 'handler_name': 'L-SMASH Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 20.05, 'bitrate': 13799, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 13574, 'video_fps': 30.0, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 256, 'video_duration': 20.05, 'video_n_frames': 601}
ffmpeg -i automation\temp\video_downloads\pixabay_12716.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2018-12-06T02:18:31.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 16693, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2018-12-06T02:18:31.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 137, 'metadata': {'creation_time': '2018-12-06T02:18:31.000000Z', 'handler_name': 'L-SMASH Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 16.62, 'bitrate': 16817, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 16693, 'video_fps': 25.0, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 137, 'video_duration': 16.62, 'video_n_frames': 415}
ffmpeg -i automation\temp\video_downloads\pixabay_19627.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:25,401 - INFO - helper.fetch - Fetched 1 videos for query 'groundbreaking technology'
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-11-06T20:24:03.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 22003, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-11-06T20:24:03.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 253, 'metadata': {'creation_time': '2021-11-06T20:24:03.000000Z', 'handler_name': 'L-SMASH Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 15.04, 'bitrate': 22262, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 22003, 'video_fps': 25.0, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 253, 'video_duration': 15.04, 'video_n_frames': 376}
ffmpeg -i automation\temp\video_downloads\pixabay_94395.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:29,469 - INFO - helper.fetch - Fetched 1 videos for query 'excited Apple fans'
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2022-08-14T01:30:07.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 22510, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2022-08-14T01:30:07.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 189, 'metadata': {'creation_time': '2022-08-14T01:30:07.000000Z', 'handler_name': 'Vimeo Artax Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 20.99, 'bitrate': 22700, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 22510, 'video_fps': 29.97002997002997, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 189, 'video_duration': 20.99, 'video_n_frames': 629}
ffmpeg -i automation\temp\video_downloads\pixabay_127737.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2022-06-28T04:47:31.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 25091, 'fps': 50.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2022-06-28T04:47:31.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.76, 'bitrate': 25097, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 25091, 'video_fps': 50.0, 'video_duration': 12.76, 'video_n_frames': 638}
ffmpeg -i automation\temp\video_downloads\pixabay_121981.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'iso5', 'minor_version': '512', 'compatible_brands': 'iso5iso6mp41', 'encoder': 'Lavf59.27.100'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': 'eng', 'default': True, 'size': [3840, 2160], 'bitrate': 25048, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'handler_name': 'AVID Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'Lavc59.37.100 libx264'}}], 'input_number': 0}], 'duration': 40.04, 'bitrate': 25025, 'start': 0.04, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 25048, 'video_fps': 25.0, 'video_duration': 40.04, 'video_n_frames': 1001}
ffmpeg -i automation\temp\video_downloads\pixabay_231485.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:42,178 - INFO - helper.fetch - Fetched 1 videos for query 'technology interaction'
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2020-08-25T22:32:48.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 24534, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2020-08-25T22:32:48.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 253, 'metadata': {'creation_time': '2020-08-25T22:32:48.000000Z', 'handler_name': 'L-SMASH Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 57.79, 'bitrate': 24792, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 24534, 'video_fps': 29.97002997002997, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 253, 'video_duration': 57.79, 'video_n_frames': 1731}
ffmpeg -i automation\temp\video_downloads\pixabay_47601.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2022-09-02T21:45:58.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 24712, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2022-09-02T21:45:58.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 189, 'metadata': {'creation_time': '2022-09-02T21:45:58.000000Z', 'handler_name': 'Vimeo Artax Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 26.01, 'bitrate': 24894, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 24712, 'video_fps': 29.97002997002997, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 189, 'video_duration': 26.01, 'video_n_frames': 779}
ffmpeg -i automation\temp\video_downloads\pixabay_129920.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:50:46,787 - INFO - helper.fetch - Fetched 1 videos for query '2027 product renaissance'
2025-05-12 20:50:47,540 - INFO - helper.fetch - Fetched 1 videos for query 'Apple product announcement'
2025-05-12 20:50:47,548 - INFO - helper.fetch - Fetched videos for 8 queries in 41.81 seconds
2025-05-12 20:50:47,552 - INFO - automation.parallel_tasks - Completed task: fetch_videos
2025-05-12 20:50:47,556 - INFO - automation.parallel_tasks - All tasks executed in 41.85 seconds
2025-05-12 20:50:47,559 - INFO - automation.shorts_maker_V - Fetched videos for 8 queries
2025-05-12 20:50:47,560 - INFO - automation.shorts_maker_V - Generated 8 audio clips
2025-05-12 20:50:47,561 - INFO - automation.shorts_maker_V - Generated 7 text clips
2025-05-12 20:50:47,561 - INFO - automation.shorts_maker_V - Processing background videos
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2022-08-14T01:30:07.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 22510, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2022-08-14T01:30:07.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 189, 'metadata': {'creation_time': '2022-08-14T01:30:07.000000Z', 'handler_name': 'Vimeo Artax Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 20.99, 'bitrate': 22700, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 22510, 'video_fps': 29.97002997002997, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 189, 'video_duration': 20.99, 'video_n_frames': 629}
ffmpeg -i automation\temp\video_downloads\pixabay_127737.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2023-08-18T01:01:50.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 11612, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2023-08-18T01:01:50.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.08, 'bitrate': 11615, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 11612, 'video_fps': 29.97002997002997, 'video_duration': 12.08, 'video_n_frames': 362}
ffmpeg -i automation\temp\video_downloads\pixabay_176489.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2023-08-17T00:43:10.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [1080, 1920], 'bitrate': 5729, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2023-08-17T00:43:10.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 25.0, 'bitrate': 5732, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [1080, 1920], 'video_bitrate': 5729, 'video_fps': 30.0, 'video_duration': 25.0, 'video_n_frames': 750}
ffmpeg -i automation\temp\video_downloads\pixabay_176320.mp4 -loglevel error -f image2pipe -vf scale=1080:1920 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2017-11-07T11:22:11.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 13574, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2017-11-07T11:22:11.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 256, 'metadata': {'creation_time': '2017-11-07T11:22:11.000000Z', 'handler_name': 'L-SMASH Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 20.05, 'bitrate': 13799, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 13574, 'video_fps': 30.0, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 256, 'video_duration': 20.05, 'video_n_frames': 601}
ffmpeg -i automation\temp\video_downloads\pixabay_12716.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-05-04T08:48:19.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [360, 640], 'bitrate': 494, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-05-04T08:48:19.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 9.0, 'bitrate': 498, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [360, 640], 'video_bitrate': 494, 'video_fps': 30.0, 'video_duration': 9.0, 'video_n_frames': 270}
ffmpeg -i automation\temp\video_downloads\pexels_7774632.mp4 -loglevel error -f image2pipe -vf scale=360:640 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2022-06-28T04:47:31.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 25091, 'fps': 50.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2022-06-28T04:47:31.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.76, 'bitrate': 25097, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 25091, 'video_fps': 50.0, 'video_duration': 12.76, 'video_n_frames': 638}
ffmpeg -i automation\temp\video_downloads\pixabay_121981.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-04-20T07:49:52.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [540, 960], 'bitrate': 780, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-04-20T07:49:52.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.0, 'bitrate': 783, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [540, 960], 'video_bitrate': 780, 'video_fps': 25.0, 'video_duration': 12.0, 'video_n_frames': 300}
ffmpeg -i automation\temp\video_downloads\pexels_7580118.mp4 -loglevel error -f image2pipe -vf scale=540:960 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2024-02-12T20:43:46.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 8164, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2024-02-12T20:43:46.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 16.67, 'bitrate': 8167, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 8164, 'video_fps': 30.0, 'video_duration': 16.67, 'video_n_frames': 500}
ffmpeg -i automation\temp\video_downloads\pixabay_200281.mp4 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:51:58,230 - INFO - helper.process - Processing 8 background clips in parallel
2025-05-12 20:51:58,302 - INFO - helper.process - Using ThreadPoolExecutor with dill for background processing
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2023-08-17T00:43:10.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [1080, 1920], 'bitrate': 5729, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2023-08-17T00:43:10.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 25.0, 'bitrate': 5732, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [1080, 1920], 'video_bitrate': 5729, 'video_fps': 30.0, 'video_duration': 25.0, 'video_n_frames': 750}
ffmpeg -ss 13.875451 -i automation\temp\video_downloads\pixabay_176320.mp4 -ss 1.000000 -loglevel error -f image2pipe -vf scale=1080:1920 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2023-08-18T01:01:50.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 11612, 'fps': 29.97002997002997, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2023-08-18T01:01:50.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.08, 'bitrate': 11615, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 11612, 'video_fps': 29.97002997002997, 'video_duration': 12.08, 'video_n_frames': 362}
ffmpeg -ss 6.783116 -i automation\temp\video_downloads\pixabay_176489.mp4 -ss 1.000000 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-05-04T08:48:19.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [360, 640], 'bitrate': 494, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-05-04T08:48:19.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 9.0, 'bitrate': 498, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [360, 640], 'video_bitrate': 494, 'video_fps': 30.0, 'video_duration': 9.0, 'video_n_frames': 270}
ffmpeg -ss 2.990454 -i automation\temp\video_downloads\pexels_7774632.mp4 -ss 1.000000 -loglevel error -f image2pipe -vf scale=360:640 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': True, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2017-11-07T11:22:11.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 13574, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2017-11-07T11:22:11.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}, {'input_number': 0, 'stream_number': 1, 'stream_type': 'audio', 'language': None, 'default': True, 'fps': 48000, 'bitrate': 256, 'metadata': {'creation_time': '2017-11-07T11:22:11.000000Z', 'handler_name': 'L-SMASH Audio Handler', 'vendor_id': '[0][0][0][0]'}}], 'input_number': 0}], 'duration': 20.05, 'bitrate': 13799, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 13574, 'video_fps': 30.0, 'default_audio_input_number': 0, 'default_audio_stream_number': 1, 'audio_fps': 48000, 'audio_bitrate': 256, 'video_duration': 20.05, 'video_n_frames': 601}
ffmpeg -ss 10.001348 -i automation\temp\video_downloads\pixabay_12716.mp4 -ss 1.000000 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42mp41isomavc1', 'creation_time': '2021-04-20T07:49:52.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [540, 960], 'bitrate': 780, 'fps': 25.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2021-04-20T07:49:52.000000Z', 'handler_name': 'L-SMASH Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 12.0, 'bitrate': 783, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [540, 960], 'video_bitrate': 780, 'video_fps': 25.0, 'video_duration': 12.0, 'video_n_frames': 300}
ffmpeg -ss 6.139923 -i automation\temp\video_downloads\pexels_7580118.mp4 -ss 1.000000 -loglevel error -f image2pipe -vf scale=540:960 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
{'video_found': True, 'audio_found': False, 'metadata': {'major_brand': 'mp42', 'minor_version': '0', 'compatible_brands': 'mp42isomavc1', 'creation_time': '2024-02-12T20:43:46.000000Z'}, 'inputs': [{'streams': [{'input_number': 0, 'stream_number': 0, 'stream_type': 'video', 'language': None, 'default': True, 'size': [3840, 2160], 'bitrate': 8164, 'fps': 30.0, 'codec_name': 'h264', 'profile': '(High)', 'metadata': {'creation_time': '2024-02-12T20:43:46.000000Z', 'handler_name': 'Vimeo Artax Video Handler', 'vendor_id': '[0][0][0][0]', 'encoder': 'AVC Coding'}}], 'input_number': 0}], 'duration': 16.67, 'bitrate': 8167, 'start': 0.0, 'default_video_input_number': 0, 'default_video_stream_number': 0, 'video_codec_name': 'h264', 'video_profile': '(High)', 'video_size': [3840, 2160], 'video_bitrate': 8164, 'video_fps': 30.0, 'video_duration': 16.67, 'video_n_frames': 500}
ffmpeg -ss 11.000101 -i automation\temp\video_downloads\pixabay_200281.mp4 -ss 1.000000 -loglevel error -f image2pipe -vf scale=3840:2160 -sws_flags bicubic -pix_fmt rgb24 -vcodec rawvideo -
2025-05-12 20:52:06,140 - INFO - helper.process - Processed background clip 1/8
2025-05-12 20:52:07,316 - INFO - helper.process - Processed background clip 2/8
2025-05-12 20:52:09,155 - INFO - helper.process - Processed background clip 3/8
2025-05-12 20:53:51,722 - INFO - helper.process - Processed background clip 4/8
2025-05-12 20:53:52,570 - INFO - helper.process - Processed background clip 5/8
2025-05-12 20:53:53,257 - INFO - helper.process - Processed background clip 6/8
2025-05-12 20:54:15,113 - INFO - helper.process - Processed background clip 7/8
2025-05-12 20:54:23,091 - INFO - helper.process - Processed background clip 8/8
2025-05-12 20:54:23,517 - INFO - helper.process - Processed 8 background clips in 145.32 seconds
2025-05-12 20:54:23,527 - INFO - automation.shorts_maker_V - Assembling final video
2025-05-12 20:54:28,573 - INFO - automation.shorts_maker_V - Rendering final video using parallel renderer
2025-05-12 20:54:28,687 - INFO - automation.parallel_renderer - Rendering 7 clips with 7 processes
2025-05-12 20:54:28,690 - INFO - automation.parallel_renderer - === Section Info ===
2025-05-12 20:54:28,691 - INFO - automation.parallel_renderer - Clip 0: Section 0 - 'LazyCreator presents: Apple’s ...'
2025-05-12 20:54:28,694 - INFO - automation.parallel_renderer - Clip 1: Section 1 - 'Big news for Apple fans!'
2025-05-12 20:54:28,695 - INFO - automation.parallel_renderer - Clip 2: Section 2 - 'A product renaissance is comin...'
2025-05-12 20:54:28,696 - INFO - automation.parallel_renderer - Clip 3: Section 3 - 'Expect groundbreaking innovati...'
2025-05-12 20:54:28,696 - INFO - automation.parallel_renderer - Clip 4: Section 4 - 'Rumors suggest new augmented r...'
2025-05-12 20:54:28,697 - INFO - automation.parallel_renderer - Clip 5: Section 5 - 'This could change how we inter...'
2025-05-12 20:54:28,697 - INFO - automation.parallel_renderer - Clip 6: Section 6 - 'Curious to know more?'
2025-05-12 20:54:28,699 - INFO - automation.parallel_renderer - Rendering clips in parallel
Rendering clips:   0%|                                                                            | 0/7 [00:00<?, ?it/s]
Traceback (most recent call last):
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=17388, pipe_handle=1528)
                                                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 1, in <module>
    from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=17388, pipe_handle=776)
                                                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
    ~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
    ~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                                  run_name="__mp_main__")
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                                  run_name="__mp_main__")
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "<frozen runpy>", line 88, in _run_code
  File "D:\youtube-shorts-automation\main.py", line 9, in <module>
    from automation.shorts_maker_V import YTShortsCreator_V
  File "D:\youtube-shorts-automation\main.py", line 9, in <module>
    from automation.shorts_maker_V import YTShortsCreator_V
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 9, in <module>
    from moviepy  import ( # for video editing
    ...<2 lines>...
    )
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 9, in <module>
    from moviepy  import ( # for video editing
    ...<2 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\__init__.py", line 4, in <module>
    from moviepy.audio import fx as afx
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\__init__.py", line 4, in <module>
    from moviepy.audio import fx as afx
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\fx\__init__.py", line 5, in <module>
    from moviepy.audio.fx.AudioDelay import AudioDelay
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\fx\__init__.py", line 5, in <module>
    from moviepy.audio.fx.AudioDelay import AudioDelay
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\fx\AudioDelay.py", line 5, in <module>
    from moviepy.audio.AudioClip import CompositeAudioClip
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\fx\AudioDelay.py", line 5, in <module>
    from moviepy.audio.AudioClip import CompositeAudioClip
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\AudioClip.py", line 13, in <module>
    from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\AudioClip.py", line 13, in <module>
    from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\io\ffmpeg_audiowriter.py", line 7, in <module>
    from moviepy.config import FFMPEG_BINARY
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\audio\io\ffmpeg_audiowriter.py", line 7, in <module>
    from moviepy.config import FFMPEG_BINARY
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\config.py", line 41, in <module>
    FFMPEG_BINARY = get_exe()
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\moviepy\config.py", line 41, in <module>
    FFMPEG_BINARY = get_exe()
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\imageio\plugins\ffmpeg.py", line 173, in get_exe
    return imageio_ffmpeg.get_ffmpeg_exe()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\imageio\plugins\ffmpeg.py", line 173, in get_exe
    return imageio_ffmpeg.get_ffmpeg_exe()
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\imageio_ffmpeg\_utils.py", line 33, in get_ffmpeg_exe
    raise RuntimeError(
    ...<2 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\imageio_ffmpeg\_utils.py", line 33, in get_ffmpeg_exe
    raise RuntimeError(
    ...<2 lines>...
    )
RuntimeError: No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.
RuntimeError: No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "D:\youtube-shorts-automation\main.py", line 9, in <module>
    from automation.shorts_maker_V import YTShortsCreator_V
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 23, in <module>
    from helper.minor_helper import measure_time, cleanup_temp_directories
  File "D:\youtube-shorts-automation\helper\minor_helper.py", line 5, in <module>
    import nltk
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\__init__.py", line 133, in <module>
    from nltk.collocations import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\collocations.py", line 36, in <module>
    from nltk.metrics import (
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\__init__.py", line 18, in <module>
    from nltk.metrics.association import (
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\association.py", line 26, in <module>
    from scipy.stats import fisher_exact
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\__init__.py", line 624, in <module>
    from ._stats_py import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\_stats_py.py", line 38, in <module>
    from scipy import sparse
  File "<frozen importlib._bootstrap>", line 1412, in _handle_fromlist
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\__init__.py", line 146, in __getattr__
    return _importlib.import_module(f'scipy.{name}')
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\__init__.py", line 315, in <module>
    from . import csgraph
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\__init__.py", line 187, in <module>
    from ._laplacian import laplacian
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\_laplacian.py", line 7, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\__init__.py", line 131, in <module>
    from ._isolve import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\__init__.py", line 4, in <module>
    from .iterative import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\iterative.py", line 5, in <module>
    from scipy.linalg import get_lapack_funcs
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\__init__.py", line 203, in <module>
    from ._misc import *
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1334, in _find_and_load_unlocked
KeyboardInterrupt
Traceback (most recent call last):
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=17388, pipe_handle=1592)
                                                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
    ~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                                  run_name="__mp_main__")
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "D:\youtube-shorts-automation\main.py", line 9, in <module>
    from automation.shorts_maker_V import YTShortsCreator_V
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 23, in <module>
    from helper.minor_helper import measure_time, cleanup_temp_directories
  File "D:\youtube-shorts-automation\helper\minor_helper.py", line 5, in <module>
    import nltk
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\__init__.py", line 133, in <module>
    from nltk.collocations import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\collocations.py", line 36, in <module>
    from nltk.metrics import (
    ...<4 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\__init__.py", line 18, in <module>
    from nltk.metrics.association import (
    ...<5 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\association.py", line 26, in <module>
    from scipy.stats import fisher_exact
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\__init__.py", line 624, in <module>
    from ._stats_py import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\_stats_py.py", line 38, in <module>
    from scipy import sparse
  File "<frozen importlib._bootstrap>", line 1412, in _handle_fromlist
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\__init__.py", line 146, in __getattr__
    return _importlib.import_module(f'scipy.{name}')
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\__init__.py", line 315, in <module>
    from . import csgraph
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\__init__.py", line 187, in <module>
    from ._laplacian import laplacian
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\_laplacian.py", line 7, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\__init__.py", line 131, in <module>
    from ._isolve import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\__init__.py", line 4, in <module>
    from .iterative import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\iterative.py", line 5, in <module>
    from scipy.linalg import get_lapack_funcs
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\__init__.py", line 203, in <module>
    from ._misc import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\_misc.py", line 4, in <module>
    from .lapack import get_lapack_funcs
  File "<string>", line 1, in <module>
    from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=17388, pipe_handle=1776)
                                                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\lapack.py", line 850, in <module>
    from scipy.linalg import _flapack
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
KeyboardInterrupt
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
    ~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                                  run_name="__mp_main__")
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "D:\youtube-shorts-automation\main.py", line 9, in <module>
    from automation.shorts_maker_V import YTShortsCreator_V
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 23, in <module>
    from helper.minor_helper import measure_time, cleanup_temp_directories
  File "D:\youtube-shorts-automation\helper\minor_helper.py", line 5, in <module>
    import nltk
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\__init__.py", line 133, in <module>
    from nltk.collocations import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\collocations.py", line 36, in <module>
    from nltk.metrics import (
    ...<4 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\__init__.py", line 18, in <module>
    from nltk.metrics.association import (
    ...<5 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\association.py", line 26, in <module>
    from scipy.stats import fisher_exact
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\__init__.py", line 624, in <module>
    from ._stats_py import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\_stats_py.py", line 38, in <module>
    from scipy import sparse
  File "<frozen importlib._bootstrap>", line 1412, in _handle_fromlist
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\__init__.py", line 146, in __getattr__
    return _importlib.import_module(f'scipy.{name}')
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\__init__.py", line 315, in <module>
    from . import csgraph
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\__init__.py", line 187, in <module>
    from ._laplacian import laplacian
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\_laplacian.py", line 7, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\__init__.py", line 131, in <module>
    from ._isolve import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\__init__.py", line 4, in <module>
    from .iterative import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\iterative.py", line 5, in <module>
    from scipy.linalg import get_lapack_funcs
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\__init__.py", line 203, in <module>
    from ._misc import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\_misc.py", line 4, in <module>
    from .lapack import get_lapack_funcs
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\lapack.py", line 850, in <module>
    from scipy.linalg import _flapack
KeyboardInterrupt
Traceback (most recent call last):
Traceback (most recent call last):
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "D:\youtube-shorts-automation\main.py", line 9, in <module>
    from automation.shorts_maker_V import YTShortsCreator_V
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 23, in <module>
    from helper.minor_helper import measure_time, cleanup_temp_directories
  File "D:\youtube-shorts-automation\helper\minor_helper.py", line 5, in <module>
    import nltk
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\__init__.py", line 133, in <module>
    from nltk.collocations import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\collocations.py", line 36, in <module>
    from nltk.metrics import (
    ...<4 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\__init__.py", line 18, in <module>
    from nltk.metrics.association import (
    ...<5 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\association.py", line 26, in <module>
    from scipy.stats import fisher_exact
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\__init__.py", line 624, in <module>
    from ._stats_py import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\_stats_py.py", line 38, in <module>
    from scipy import sparse
  File "<frozen importlib._bootstrap>", line 1412, in _handle_fromlist
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\__init__.py", line 146, in __getattr__
    return _importlib.import_module(f'scipy.{name}')
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\__init__.py", line 315, in <module>
    from . import csgraph
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\__init__.py", line 187, in <module>
    from ._laplacian import laplacian
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\_laplacian.py", line 7, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\__init__.py", line 131, in <module>
    from ._isolve import *
  File "<string>", line 1, in <module>
    from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=17388, pipe_handle=1684)
                                                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\__init__.py", line 4, in <module>
    from .iterative import *
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\iterative.py", line 5, in <module>
    from scipy.linalg import get_lapack_funcs
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
    ~~~~~~~^^^^^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\__init__.py", line 203, in <module>
    from ._misc import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\_misc.py", line 4, in <module>
    from .lapack import get_lapack_funcs
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\lapack.py", line 883, in <module>
    p1 = regex_compile(r'with bounds (?P<b>.*?)( and (?P<s>.*?) storage){0,1}\n')
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                                  run_name="__mp_main__")
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\re\__init__.py", line 287, in compile
    def compile(pattern, flags=0):

  File "<frozen runpy>", line 287, in run_path
KeyboardInterrupt
  File "<frozen runpy>", line 98, in _run_module_code

During handling of the above exception, another exception occurred:

  File "<frozen runpy>", line 88, in _run_code
Traceback (most recent call last):
  File "D:\youtube-shorts-automation\main.py", line 9, in <module>
    from automation.shorts_maker_V import YTShortsCreator_V
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 23, in <module>
    from helper.minor_helper import measure_time, cleanup_temp_directories
  File "D:\youtube-shorts-automation\helper\minor_helper.py", line 5, in <module>
    import nltk
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\__init__.py", line 133, in <module>
    from nltk.collocations import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\collocations.py", line 36, in <module>
    from nltk.metrics import (
    ...<4 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\__init__.py", line 18, in <module>
    from nltk.metrics.association import (
    ...<5 lines>...
    )
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\nltk\metrics\association.py", line 26, in <module>
    from scipy.stats import fisher_exact
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\__init__.py", line 624, in <module>
    from ._stats_py import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\stats\_stats_py.py", line 38, in <module>
    from scipy import sparse
  File "<string>", line 1, in <module>
    from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=17388, pipe_handle=1768)
                                                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1412, in _handle_fromlist
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\__init__.py", line 146, in __getattr__
    return _importlib.import_module(f'scipy.{name}')
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 131, in _main
    prepare(preparation_data)
    ~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                                  run_name="__mp_main__")
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\__init__.py", line 315, in <module>
    from . import csgraph
  File "<frozen runpy>", line 287, in run_path
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\__init__.py", line 187, in <module>
    from ._laplacian import laplacian
  File "<frozen runpy>", line 96, in _run_module_code
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\csgraph\_laplacian.py", line 7, in <module>
    from scipy.sparse.linalg import LinearOperator
  File "<frozen runpy>", line 42, in __exit__
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\__init__.py", line 131, in <module>
    from ._isolve import *
KeyboardInterrupt
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\__init__.py", line 4, in <module>
    from .iterative import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\sparse\linalg\_isolve\iterative.py", line 5, in <module>
    from scipy.linalg import get_lapack_funcs
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\__init__.py", line 203, in <module>
    from ._misc import *
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\_misc.py", line 4, in <module>
    from .lapack import get_lapack_funcs
  File "D:\youtube-shorts-automation\venv\Lib\site-packages\scipy\linalg\lapack.py", line 850, in <module>
    from scipy.linalg import _flapack
KeyboardInterrupt
2025-05-12 20:55:59,494 - INFO - __main__ - Running global cleanup of temporary files
2025-05-12 20:55:59,496 - INFO - helper.minor_helper - Cleaning up temporary directories in D:\youtube-shorts-automation\automation\temp
2025-05-12 20:55:59,498 - INFO - __main__ - Performing final cleanup of all temporary files
2025-05-12 20:55:59,499 - INFO - helper.minor_helper - Cleaning up temporary directories in D:\youtube-shorts-automation\automation\temp
2025-05-12 20:55:59,499 - INFO - helper.minor_helper - Removing temporary directory: D:\youtube-shorts-automation\automation\temp\audio_clips
2025-05-12 20:55:59,502 - INFO - helper.minor_helper - Removing temporary directory: D:\youtube-shorts-automation\automation\temp\generated_images
2025-05-12 20:55:59,503 - INFO - helper.minor_helper - Removing temporary directory: D:\youtube-shorts-automation\automation\temp\shorts_v_1747048348
2025-05-12 20:55:59,520 - INFO - helper.minor_helper - Removing temporary directory: D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194
2025-05-12 20:55:59,521 - WARNING - helper.minor_helper - Failed to remove directory D:\youtube-shorts-automation\automation\temp\shorts_v_1747063194: [WinError 32] The process cannot access the file because it is being used by another process: 'D:\\youtube-shorts-automation\\automation\\temp\\shorts_v_1747063194\\audio_section_0_1747063205.mp3'
2025-05-12 20:55:59,522 - INFO - helper.minor_helper - Removing temporary directory: D:\youtube-shorts-automation\automation\temp\video_downloads
2025-05-12 20:55:59,536 - WARNING - helper.minor_helper - Failed to remove directory D:\youtube-shorts-automation\automation\temp\video_downloads: [WinError 32] The process cannot access the file because it is being used by another process: 'D:\\youtube-shorts-automation\\automation\\temp\\video_downloads\\pexels_7580118.mp4'
concurrent.futures.process._RemoteTraceback:
"""
Traceback (most recent call last):
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\queues.py", line 262, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AttributeError: Can't get local object 'VideoClip.__init__.<locals>.<lambda>'
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\youtube-shorts-automation\automation\parallel_renderer.py", line 251, in render_clips_in_parallel
    idx, path = future.result()
                ~~~~~~~~~~~~~^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\_base.py", line 449, in result
    return self.__get_result()
           ~~~~~~~~~~~~~~~~~^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\_base.py", line 401, in __get_result
    raise self._exception
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\queues.py", line 262, in _feed
    obj = _ForkingPickler.dumps(obj)
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^
AttributeError: Can't get local object 'VideoClip.__init__.<locals>.<lambda>'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\youtube-shorts-automation\main.py", line 337, in <module>
    main(creator_type)
    ~~~~^^^^^^^^^^^^^^
  File "D:\youtube-shorts-automation\main.py", line 282, in main
    result = generate_youtube_short(
        topic,
    ...<2 lines>...
        creator_type=creator_type
    )
  File "D:\youtube-shorts-automation\main.py", line 180, in generate_youtube_short
    video_path = creator_type.create_youtube_short(
        title=title,  # Use the generated title
    ...<8 lines>...
        edge_blur=False
    )
  File "D:\youtube-shorts-automation\helper\minor_helper.py", line 31, in wrapper
    result = func(*args, **kwargs)
  File "D:\youtube-shorts-automation\automation\shorts_maker_V.py", line 327, in create_youtube_short
    render_clips_in_parallel(
    ~~~~~~~~~~~~~~~~~~~~~~~~^
        section_clips,
        ^^^^^^^^^^^^^^
    ...<6 lines>...
        section_info=section_info
        ^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "D:\youtube-shorts-automation\automation\parallel_renderer.py", line 248, in render_clips_in_parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
         ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\_base.py", line 647, in __exit__
    self.shutdown(wait=True)
    ~~~~~~~~~~~~~^^^^^^^^^^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\concurrent\futures\process.py", line 846, in shutdown
    self._executor_manager_thread.join()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\adhbu\AppData\Local\Programs\Python\Python313\Lib\threading.py", line 1092, in join
    self._handle.join(timeout)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^
KeyboardInterrupt
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1747063563.287067    2416 init.cc:232] grpc_wait_for_shutdown_with_timeout() timed out.
PS D:\youtube-shorts-automation> ffmpeg --version
ffmpeg version N-118586-g629e8a2425-20250301 Copyright (c) 2000-2025 the FFmpeg developers
  built with gcc 14.2.0 (crosstool-NG 1.26.0.120_4d36f27)
  configuration: --prefix=/ffbuild/prefix --pkg-config-flags=--static --pkg-config=pkg-config --cross-prefix=x86_64-w64-mingw32- --arch=x86_64 --target-os=mingw32 --enable-gpl --enable-version3 --disable-debug --enable-shared --disable-static --disable-w32threads --enable-pthreads --enable-iconv --enable-zlib --enable-libfreetype --enable-libfribidi --enable-gmp --enable-libxml2 --enable-lzma --enable-fontconfig --enable-libharfbuzz --enable-libvorbis --enable-opencl --disable-libpulse --enable-libvmaf --disable-libxcb --disable-xlib --enable-amf --enable-libaom --enable-libaribb24 --enable-avisynth --enable-chromaprint --enable-libdav1d --enable-libdavs2 --enable-libdvdread --enable-libdvdnav --disable-libfdk-aac --enable-ffnvcodec --enable-cuda-llvm --enable-frei0r --enable-libgme --enable-libkvazaar --enable-libaribcaption --enable-libass --enable-libbluray --enable-libjxl --enable-libmp3lame --enable-libopus --enable-librist --enable-libssh --enable-libtheora --enable-libvpx --enable-libwebp --enable-libzmq --enable-lv2 --enable-libvpl --enable-openal --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenh264 --enable-libopenjpeg --enable-libopenmpt --enable-librav1e --enable-librubberband --enable-schannel --enable-sdl2 --enable-libsnappy --enable-libsoxr --enable-libsrt --enable-libsvtav1 --enable-libtwolame --enable-libuavs3d --disable-libdrm --enable-vaapi --enable-libvidstab --enable-vulkan --enable-libshaderc --enable-libplacebo --disable-libvvenc --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxvid --enable-libzimg --enable-libzvbi --extra-cflags=-DLIBTWOLAME_STATIC --extra-cxxflags= --extra-libs=-lgomp --extra-ldflags=-pthread --extra-ldexeflags= --cc=x86_64-w64-mingw32-gcc --cxx=x86_64-w64-mingw32-g++ --ar=x86_64-w64-mingw32-gcc-ar --ranlib=x86_64-w64-mingw32-gcc-ranlib --nm=x86_64-w64-mingw32-gcc-nm --extra-version=20250301
  libavutil      59. 57.100 / 59. 57.100
  libavcodec     61. 33.102 / 61. 33.102
  libavformat    61.  9.107 / 61.  9.107
  libavdevice    61.  4.100 / 61.  4.100
  libavfilter    10.  9.100 / 10.  9.100
  libswscale      8. 13.100 /  8. 13.100
  libswresample   5.  4.100 /  5.  4.100
  libpostproc    58.  4.100 / 58.  4.100
Unrecognized option '-version'.
Error splitting the argument list: Option not found
PS D:\youtube-shorts-automation> ffmpeg
ffmpeg version N-118586-g629e8a2425-20250301 Copyright (c) 2000-2025 the FFmpeg developers
  built with gcc 14.2.0 (crosstool-NG 1.26.0.120_4d36f27)
  configuration: --prefix=/ffbuild/prefix --pkg-config-flags=--static --pkg-config=pkg-config --cross-prefix=x86_64-w64-mingw32- --arch=x86_64 --target-os=mingw32 --enable-gpl --enable-version3 --disable-debug --enable-shared --disable-static --disable-w32threads --enable-pthreads --enable-iconv --enable-zlib --enable-libfreetype --enable-libfribidi --enable-gmp --enable-libxml2 --enable-lzma --enable-fontconfig --enable-libharfbuzz --enable-libvorbis --enable-opencl --disable-libpulse --enable-libvmaf --disable-libxcb --disable-xlib --enable-amf --enable-libaom --enable-libaribb24 --enable-avisynth --enable-chromaprint --enable-libdav1d --enable-libdavs2 --enable-libdvdread --enable-libdvdnav --disable-libfdk-aac --enable-ffnvcodec --enable-cuda-llvm --enable-frei0r --enable-libgme --enable-libkvazaar --enable-libaribcaption --enable-libass --enable-libbluray --enable-libjxl --enable-libmp3lame --enable-libopus --enable-librist --enable-libssh --enable-libtheora --enable-libvpx --enable-libwebp --enable-libzmq --enable-lv2 --enable-libvpl --enable-openal --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenh264 --enable-libopenjpeg --enable-libopenmpt --enable-librav1e --enable-librubberband --enable-schannel --enable-sdl2 --enable-libsnappy --enable-libsoxr --enable-libsrt --enable-libsvtav1                                                                                                                          --enable-libtwolame --enable-libuavs3d --disable-libdrm --enable-vaapi --enable-libvidstab --enable-vulkan --enable-libshaderc --enable-libplacebo --disable-libvvenc --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxvid --enable-libzimg --enable-libzvbi --extra-cflags=-DLIBTWOLAME_STATIC --extra-cxxflags= --extra-libs=-lgomp --extra-ldflags=-pthread --extra-ldexeflags= --cc=x86_64-w64-mingw32-gcc --cxx=x86_64-w64-mingw32-g++ --ar=x86_64-w64-mingw32-gcc-ar --ranlib=x86_64-w64-mingw32-gcc-ranlib --nm=x86_64-w64-mingw32-gcc-nm --extra-version=20250301
  libavcodec     61. 33.102 / 61. 33.102
  libavformat    61.  9.107 / 61.  9.107
  libavdevice    61.  4.100 / 61.  4.100
  libavfilter    10.  9.100 / 10.  9.100
  libswscale      8. 13.100 /  8. 13.100
  libswresample   5.  4.100 /  5.  4.100
  libpostproc    58.  4.100 / 58.  4.100
Universal media converter
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...