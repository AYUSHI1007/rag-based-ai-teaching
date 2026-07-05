import whisper
print(whisper.__file__)
import json

model = whisper.load_model("turbo")
result = model.transcribe("audios/output3.mp3",word_timestamps=False)
print(result["segments"])
chunks=[]
for segment in result["segments"]:
    chunks.append({"start":segment["start"],"end":segment["end"],"text":segment["text"]})
print(chunks)
with open("output.json", "w") as f:
    json.dump(chunks, f)