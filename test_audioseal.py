from audioseal import AudioSeal
import os 


model = AudioSeal.load_generator("./checkpoint.th", nbits=16)

# We add the batch dimension to the single audio to mimic the batch watermarking
audios = audio.unsqueeze(0)

watermark = model.get_watermark(audios, sample_rate=sr)
watermarked_audio = audios + watermark

# Alternatively, you can also call forward() function directly with different tune-down / tune-up rate
watermarked_audio = model(audios, sample_rate=sr, alpha=1)