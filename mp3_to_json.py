import json
import whisper
import os
model = whisper.load_model("turbo")
audios=os.listdir("audios")
for audio in audios:
    json_file = f"jsons/{audio}.json"
    if os.path.exists(json_file):
        print(f"Skipping {audio}")
        continue
    number=audio.split(".")[0]
    print(audio)
    print(number)
    result = model.transcribe(f"audios/{audio}",word_timestamps=False)
    # result = model.transcribe(f"audios/output3.mp3",word_timestamps=False)
    print(result["segments"])
    chunks=[]
    for segment in result["segments"]:
        chunks.append({"number":number,"start":segment["start"],"end":segment["end"],"text":segment["text"]})
    print(chunks)
    chunks_with_metadata={"chunks":chunks,"text":result["text"]}
    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f)