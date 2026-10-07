# Design Note — Offline Satellite Tile Classifier

## What this is

A small but complete slice of what the brief describes: one endpoint that
takes in a satellite tile, classifies it into one of 7 land-use types using a
model that runs entirely on this machine, saves the result, and lets someone
query it afterward — all without touching the internet, and all on a normal
CPU. On the 210-tile eval set, it got 91.9% accuracy and 0.920 macro-F1, and
runs consistently fast — a median of 13.4ms per tile, with even the slowest
tile in the batch only taking 15ms. That kind of tight, predictable timing
matters if this were ever feeding something like a real-time control loop
that needs each frame handled on schedule, not just "fast on average."

## How a tile actually moves through the system

```
curl/analyst → FastAPI (app/main.py)
                 ├→ classifier.predict()  (app/classifier.py)
                 │    ├→ preprocessing.py  (the one and only copy of resize/normalize)
                 │    ├→ frozen SqueezeNet backbone → 512-number feature vector
                 │    └→ trained Logistic Regression head → 7 probabilities
                 └→ storage.save_prediction()  (SQLite, app/storage.py)
```

- A `POST /classify` comes in with image bytes. If the file can't even be
  read as an image, that's a 400 (the caller's problem); if something goes
  wrong on our side while processing a valid image, that's a 500 (our
  problem). I kept that distinction deliberate rather than lumping every
  failure into one error code.
- The image gets resized from 64×64 up to 224×224 and normalized the way
  ImageNet-trained models expect. Resizing it directly like that only works
  cleanly because every one of the 1,260 tiles I checked is genuinely
  square — no stretching or distortion involved.
- The frozen backbone turns that tile into a feature vector; the head turns
  that vector into 7 probabilities, one per class. The backbone itself is
  never trained further — with only 150 example images per class, reusing
  an already-trained model beats trying to train one from nothing, and the
  brief explicitly allows this.
- Whichever class has the highest probability becomes the prediction. If
  that top probability is below a confidence cutoff, the tile also gets
  flagged as uncertain — but importantly, the label itself never gets
  overwritten with the word "uncertain" (more on why below).
- Every classification gets written as one row in SQLite: filename, a
  content hash, the predicted label, the confidence, the full set of 7
  probabilities, whether it was flagged, which model version made the call,
  and a timestamp. The response also hands back that row's ID, so any single
  API call can be traced straight back to exactly where it landed in the
  database.

## Picking the model — by testing, not by guessing

Rather than just assuming one architecture would be the right pick, I
actually tested 3 pretrained backbones (SqueezeNet1_1, MobileNetV2, ResNet18)
against 3 different classifier heads (Logistic Regression, a calibrated
Linear SVM, and a small neural network) — 9 combinations total. All of them
were trained on the same 892 tiles and scored on a separate 158-tile
validation slice that I never touched again until the very end. Honestly, all
9 combinations came out within about 1.5 points of each other on macro-F1 —
on only 158 images, that's basically noise, not a meaningful difference. So I
made the tie-break call in advance, before seeing results: if it's this
close, go with whichever option is simplest and fastest. That ended up being
SqueezeNet plus Logistic Regression — it happened to score best *and* trained
in a tenth of a second. I ran the whole pipeline twice from scratch just to
check, and it reproduced every single reported number exactly both times.

## The decision I actually spent the most time thinking through: what to do when the model isn't sure

I considered three options: always just trust whatever the model says;
force anything below a confidence cutoff to say "uncertain" outright; or keep
the model's real guess but separately flag it as low-confidence.

I went with the third option. Blindly trusting every guess hides real risk.
Overwriting the label with "uncertain" throws away information and makes a
mess of anything trying to query results by label later. Keeping the raw
guess and flagging it separately means the decision about what to *do* with
an uncertain tile can be made later, at query time, rather than being baked
in permanently. And the threshold itself wasn't a guess.I swept a range of
values on the validation data and 0.85 came out on top. On tiles the model
had never seen, that threshold flagged 14.8% of them, and those flagged tiles
accounted for 70.6% of every mistake the model made — a flagged tile was
roughly 14 times more likely to be wrong than one that wasn't flagged. Net
effect: about 85% of tiles can just be trusted and auto-processed at 97.2%
accuracy, and the riskier 15% get routed for a second look. To be honest
about the limits of this 5 of the 17 total mistakes were ones the model
was actually confident about, which no single global cutoff can catch. Doing
this per class instead of with one shared number is the obvious next step.

## What gets saved, and why each piece matters

I save the full probability breakdown for every tile, not just the winning
label if a stored result is ever questioned later, it can be re-examined
without having to re-run the model from scratch. The content hash lets me
spot duplicate submissions or catch a tile that got mixed up somewhere
upstream. Saving the model version and a timestamp with every row means
that months from now, if someone asks "which model actually produced this
result," there's a real answer. And if the same tile gets submitted twice, I
just add a new row rather than overwrite the old one — every classification
is its own event in the record, not a single piece of state that keeps
getting replaced.

## What "being able to query the results" actually means here

For this slice, it just means running plain SQL directly against the SQLite
file grouping results by label, or pulling every uncertain tile from a
given day. I didn't build a dedicated query endpoint on top of that, and
that was a deliberate choice, not something I ran out of time for: the brief
asked specifically for the core ingest-classify-store path, and being able
to query the data with plain SQL already proves the storage is shaped the
way an analyst would actually need it.

## What I assumed, since I couldn't ask anyone

- Every tile stands alone I'm not accounting for any relationship between
  tiles, and I'm trusting that a training tile's folder name is its correct
  label.
- The eval CSV's labels are correct, full stop.
- One shared confidence threshold is good enough for a slice like this;
  a smarter per-class version is a known next step, not something I built.
- Whoever queries this system later is doing occasional analyst-style
  lookups, not hammering it with constant production traffic.
- The eval set is a fair stand-in for what real tiles will actually look
  like if that's wrong, most of these numbers stop meaning much.

## What I'd actually want to ask GalaxEye

- How many tiles come in, and how fast? That changes whether batching or
  queuing needs to be part of the design at all.
- When the model gets something wrong versus when it correctly flags
  something for review which mistake actually costs more? That trade-off
  is really what should be setting the threshold, not just the statistics
  on their own.
- If a tile gets flagged uncertain, does it go to a person, or to a bigger,
  slower model for a second opinion?
- Are these 7 categories permanent, or does the system eventually need to
  recognize tiles that don't fit any of them?
- If the model gets upgraded later, should old predictions get reprocessed
  with the new one, or just stay as they were? (I built `model_version`
  into the storage specifically so either answer is easy to support.)

## What would actually break first

Two things, and I'd rather say so plainly than pretend otherwise. Everything
runs one tile at a time, on a single CPU process about 75 tiles a second,
which is plenty here but would fall over the moment tiles start arriving
faster than that. And SQLite only allows one write at a time, so a burst of
traffic backs up right behind that same bottleneck. Neither matters at this
scale, but if this ever had to handle real production volume, the fix is a
queue, batching the model calls together, and swapping SQLite for something
like Postgres that can actually handle multiple writers at once.
```