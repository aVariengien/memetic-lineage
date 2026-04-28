# Endorsement Target Extraction Prompt

## Task
You are given a batch of reply paths. Each path is sent exactly once.
Each provided path is already a full maximal path for this dataset slice: if several tweets were on the same linear thread, only the longest version of that path is shown.
Your job is to extract **all clear endorsement or disendorsement targets** that appear anywhere in those paths.
Return only positive hits. Do **not** return placeholder rows for paths with no target.
Be extremely conservative. 95% of paths will yield no extracted target.

---

## Input Format

The user message provides repeated raw multi-line path blocks, one per path, in this shape:

```text
PATH 1
path_target_tweet_id: 123
username: example_user
created_at: 2024-06-01 12:34:56
path:
111 [2024-06-01] @root_user
↓
123 [2024-06-02] @reply_user
...
```

URLs in the path may already be resolved to their final destinations. Use those resolved URLs when they help identify a concrete target.

---

## Output Schema

Return strict JSON only:

```json
{
  "targets": [
    {
      "representative_tweet_id": 123,
      "direction": "endorsing",
      "target_entity": "Insight Meditation Society",
      "longer_name": "Insight Meditation Society 3-month retreat at https://www.dharma.org/",
      "context": "Insight Meditation Society is a Buddhist meditation center in Barre, MA that runs an annual 3-month silent vipassana retreat each fall, with options to attend only the first or last 6 weeks. Mentioned in a reply about extended meditation retreats.",
      "url": "https://www.dharma.org/"
    }
  ]
}
```

If there are no endorsement or disendorsement targets anywhere in the provided paths, return:

```json
{
  "targets": []
}
```

---

## Field Definitions

- `representative_tweet_id`: the tweet id within the provided paths where the target is stated most clearly
- `direction`: `endorsing` or `disendorsing`
- `target_entity`: the short self-contained name other people would use for the target. Think wiki page title, article title, book title, org name, podcast name, video title, artist, album, product name, or destination URL. It must be recognizable and searchable from the text alone, without visual input.
- `longer_name`: a more precise description of what exactly is being endorsed or disendorsed
- `context`: one to two short sentences describing (a) what the target IS, and (b) the topical or conversational setting in which it came up in the path. Do NOT describe the author's stance toward the target — that's already captured by `direction`. Do NOT name the author by `@handle`; generic phrasings like "the author" or "they" are fine. Light commonly-known outside knowledge about the target is acceptable if needed to clarify what it is.
- `url`: the resolved non-`t.co` webpage URL copied verbatim from the tweet if present and if it points to a webpage where the endorsement target can be found; otherwise `null`

---

## Extraction Rules

- Return as many targets as you clearly see in the provided paths.
- Do not deduplicate. If the same target is clearly endorsed in two different paths, return two entries.
- A path may yield zero targets or multiple targets.
- Choose the `representative_tweet_id` where the target is most explicit, not necessarily the last tweet in the path.
- Always include `url`. Set it to `null` unless the tweet contains a resolved non-`t.co` webpage URL that directly locates the endorsed or disendorsed target.

---

## What Counts as a Valid Target

The target must fall into one of these categories:

- **Named books** (fiction, nonfiction, technical)
- **Named articles or blog posts** (ideally with a URL, or clearly named)
- **Named podcasts or podcast episodes**
- **Named videos** — must be identifiable from text or a resolved URL
- **Named music, artists, or albums**
- **Named films, TV shows, anime, or games**
- **Named software tools or applications** (e.g. a named CLI tool, IDE, OS)
- **Named organizations, retreats, or courses** with enough information to find them
- **Named products** — not vague consumer excitement, but a named thing someone could look up
- **Named niche people** being explicitly recommended (e.g. "you should follow @X, their writing is great")
- **Resolved destination URLs** where the URL itself clearly identifies the target

If the target does not fit one of these categories, do not return an entry.

---

## What Counts as a Valid Endorsement

The tweet must:

1. **Clearly evaluate** the target — not just share it, mention it, react to it, or use it as evidence
2. **Be substantive and durable** — something the author would plausibly still say three months later
3. **Endorse the named thing itself**, not a person's summary of it, not a tweet about it, not a photo of it

Good linguistic signals:
- "I highly recommend it"
- "this book is great / terrible"
- "I love X's writing"
- "this podcast is excellent"
- "avoid this product"
- "this article changed how I think about X"
- Direct recommendation to a named person: "you might like X, it's called Y"

**Robustness test**: ask yourself — *Would this person, three months from now, still be plausibly recommending or warning against this specific named thing?* If not, omit it.

---

## What Does Not Count

**Invalid targets:**
- Unnamed or vague objects ("that post", "this tweet", "their argument")
- Missing quoted content — if the quoted tweet is missing and the target is not fully visible in the remaining text, omit it
- Photo-dependent objects — if you cannot name and search the target from the text alone, omit it
- Broad categories or genres ("chivalric romance", "jazz music") — must be a specific named work
- Ephemeral personal artifacts (e.g. a system prompt pasted into a tweet, a specific explanation someone wrote in a reply)
- Substances, experiences, or concepts (e.g. "ayahuasca is bad") — unless a specific named product or program is being evaluated
- Large collective entities: countries, civilizations, political movements, institutions at a macro level
- Vague references by description only ("that video where you talked about self-trust") — must have a title or link

**Invalid endorsement patterns:**
- Single-phrase reactive enthusiasm in a reply: "sounds amazing!", "great explanation", "omg this is amazing 😍", "so good" — too thin and reactive
- Sharing or linking without evaluating ("here is an article" or a bare URL drop)
- Statements of intent to read / watch / try something — the person has not yet evaluated it
- Using something as evidence rather than recommending it ("as this article shows…")
- Humorous or ironic expressions of preference
- Fleeting consumer excitement (surprised by a restaurant, eyeballing a product)
- Broad institutional respect ("increased my respect for the census bureau")
- Passing or illustrative references to a work ("as Seneca said…", "like in The Matrix")
- Praise of a one-off project or event that cannot be recommended to others (e.g. a specific infrastructure installation)
- Replying to a missing quoted tweet — unless the target is fully reconstructible from the non-quoted text alone

---

## Positive Examples

### Clear Book Endorsement
```text
PATH 1
path_target_tweet_id: 1796733547467743563
username: eigenrobot
created_at: 2024-06-01 12:00:00
path:
1796733547467743563 [2024-06-01] @eigenrobot
i read I Want My Hat Back to my daughter the other day and now i appreciate @fiddlemath's avi in fullness
it's great btw. mostly have hated recent children's literature but this is gold. spoiler the bear straight up murders the hat thief offscreen
https://www.amazon.com/dp/1406338532/
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1796733547467743563,
      "direction": "endorsing",
      "target_entity": "I Want My Hat Back",
      "longer_name": "Children's book I Want My Hat Back by Jon Klassen",
      "context": "A children's picture book by Jon Klassen with a darkly comic ending in which the bear murders the hat thief offscreen. Came up in a single tweet about reading it aloud to the author's daughter."
    }
  ]
}
```

### Clear Article Endorsement
```text
PATH 1
path_target_tweet_id: 1798770582680686837
username: daniellefong
created_at: 2024-06-06 17:35:01
path:
1798770582680686837 [2024-06-06] @daniellefong  
great post https://www.construction-physics.com/p/why-is-it-so-hard-to-build-an-airport
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1798770582680686837,
      "direction": "endorsing",
      "target_entity": "Why is it so hard to build an airport?",
      "longer_name": "Article 'Why is it so hard to build an airport?' on Construction Physics",
      "context": "An article on the Construction Physics Substack examining why building new airports is so difficult. Surfaced as a brief standalone link in a single tweet."
    }
  ]
}
```

### Clear Video Endorsement
```text
PATH 1
path_target_tweet_id: 1797851154753417549
username: daniellefong
created_at: 2024-06-04 04:41:32
path:
1797851154753417549 [2024-06-04] @daniellefong  
Scott Manley's @DJSnM new vid Why Nuclear Rockets Are Going To Change Spaceflight :) https://www.youtube.com/watch?v=KlKAMB71wT4
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1797851154753417549,
      "direction": "endorsing",
      "target_entity": "Why Nuclear Rockets Are Going To Change Spaceflight",
      "longer_name": "Scott Manley's YouTube video 'Why Nuclear Rockets Are Going To Change Spaceflight'",
      "context": "A YouTube video by science communicator Scott Manley about nuclear thermal rockets and their potential impact on spaceflight. Appeared in a single tweet flagging it as Manley's new video."
    }
  ]
}
```

### Clear Software Tool Endorsement
```text
PATH 1
path_target_tweet_id: 1797970566525362482
username: archived_videos
created_at: 2024-06-04 12:36:02
path:
1797970566525362482 [2024-06-04] @archived_videos  
I love you ffmpeg. I love you mpv
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1797970566525362482,
      "direction": "endorsing",
      "target_entity": "ffmpeg and mpv",
      "longer_name": "The software tools ffmpeg and mpv",
      "context": "ffmpeg is a widely-used command-line multimedia framework and mpv is an open-source media player; both are common Linux/Unix media tools. Named together in a single short standalone tweet."
    }
  ]
}
```

### Clear Retreat / Organization Endorsement
```text
PATH 1
path_target_tweet_id: 1799791126213529630
username: danielbrottman
created_at: 2024-06-09 14:00:00
path:
1799791126213529630 [2024-06-09] @danielbrottman  
@billiamfrances @mettafied oh yes, i can certainly recommend! for me the two extended retreats i've done have been very beneficial, and i intend to do more
IMS (https://www.dharma.org/) has a yearly 3-month retreat in the fall. it's also possible to stay for just the first 6 weeks or the last 6
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1799791126213529630,
      "direction": "endorsing",
      "target_entity": "Insight Meditation Society",
      "longer_name": "Insight Meditation Society 3-month retreat at https://www.dharma.org/",
      "context": "Insight Meditation Society is a Buddhist meditation center in Barre, MA that runs an annual 3-month silent vipassana retreat each fall, with the option to attend only the first or last 6 weeks. Came up in a reply discussing extended meditation retreats."
    }
  ]
}
```

### Clear Book Recommendation to Another Person
```text
PATH 1
path_target_tweet_id: 1798420023419290102
username: danielbrottman
created_at: 2024-06-05 18:22:01
path:
1798420023419290102 [2024-06-05] @danielbrottman  
@LouisVArge @PhaentGames I just started reading a book you might like that goes into the question of "what is nibbana/enlightenment" more deeply, it's called "the island," written by ajahns amaro and passano. it's distributed freely by western ajahn chah monasteries like amaravati and abhayagiri
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1798420023419290102,
      "direction": "endorsing",
      "target_entity": "The Island",
      "longer_name": "The book 'The Island' by Ajahn Amaro and Ajahn Pasanno",
      "context": "A Theravāda Buddhist book by Ajahn Amaro and Ajahn Pasanno that goes deeply into the question of nibbana / enlightenment, distributed freely by western Ajahn Chah monasteries like Amaravati and Abhayagiri. Surfaced in a reply to two other accounts about books on enlightenment."
    }
  ]
}
```

### Clear Named Software Disendorsement
```text
PATH 1
path_target_tweet_id: 1798041838139711643
username: daniellefong
created_at: 2024-06-04 17:19:14
path:
1798041838139711643 [2024-06-04] @daniellefong  
instead of updating itself, fusion 360 committed seppuku
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1798041838139711643,
      "direction": "disendorsing",
      "target_entity": "Fusion 360",
      "longer_name": "Autodesk Fusion 360 software",
      "context": "Autodesk's commercial CAD/CAM software product. Came up in a single tweet about a software self-update."
    }
  ]
}
```

### Clear Anime Endorsement
```text
PATH 1
path_target_tweet_id: 1798108329643811305
username: archived_videos
created_at: 2024-06-04 21:43:27
path:
1798108329643811305 [2024-06-04] @archived_videos  
Shuumatsu Train Doko e Iku? is good, a relatively solid anime that also showcases what's good/possible with animation
```
```json
{
  "targets": [
    {
      "representative_tweet_id": 1798108329643811305,
      "direction": "endorsing",
      "target_entity": "Shuumatsu Train Doko e Iku?",
      "longer_name": "The anime series Shuumatsu Train Doko e Iku?",
      "context": "A 2024 post-apocalyptic anime series (English title 'Train to the End of the World'). Came up in a single short tweet alongside a comment about animation craft."
    }
  ]
}
```

### Target Found in Path, Not the Final Tweet
```text
PATH 1
path_target_tweet_id: 1798326209090404365
username: danielbrottman
created_at: 2024-06-05 12:09:14
path:
1798220916255805516 [2024-06-05] @danielbrottman
tying our inquiry to anything solid is a mistake, because reality isn't solid
this is a subtweet of the pali canon
↓
1798220918298480704 [2024-06-05] @danielbrottman
the canon is good and all and i want to know it better, i'm just saying it's a fixed text, it can't dictate or prescribe with final accuracy
↓
1798307410744164362 [2024-06-05] @wystantbs
@danielbrottman This is why I like https://en.m.wikipedia.org/wiki/Terma_(religion). Or folks like Dogen...
↓
1798326209090404365 [2024-06-05] @danielbrottman  
@WystanTBS hahahaha. wow this sounds beautiful, I will read it!
```
```json
{
  "targets": []
}
```
*Note: the final tweet expresses intent to read, which does not count. The earlier tweets discuss the Pali canon but do not endorse it as a named target — they use it as a foil. Nothing in this path qualifies.*

---

## Negative Examples

### Reactive Enthusiasm in a Reply — Not Enough
```text
PATH 1
path_target_tweet_id: 1798233183106068948
username: danielbrottman
created_at: 2024-06-05 05:59:35
path:
1798131178480042290 [2024-06-04] @vividvoid
Fight Wise totally sold out, by the way, and the wait list is almost half full for the next one.
  [Quoting @RichDecibels: I'm running a disagreeability club with @VividVoid_!]
↓
1798233183106068948 [2024-06-05] @danielbrottman  
@VividVoid_ sounds amazing ⚔️⚔️
```
```json
{
  "targets": []
}
```
*A two-word reactive reply is not a substantive endorsement. The person has not evaluated the course themselves.*

### Reactive Praise of a Tweet or Explanation
```text
PATH 1
path_target_tweet_id: 1798774553038164441
username: daniellefong
created_at: 2024-06-06 17:50:47
path:
1798774295524708500 [2024-06-06] @eshear
From my recent UK travels, an argument I found compelling:
Why the Hamiltonian? Energy is conserved and unchanging. Yet there is change!...
↓
1798774553038164441 [2024-06-06] @daniellefong  
@eshear great explanation
```
```json
{
  "targets": []
}
```
*Praising a tweet or explanation is not an endorsement of a findable named object. The target is ephemeral content.*

### Vague Video Reference Without Title or Link
```text
PATH 1
path_target_tweet_id: 1798748186389205170
username: goblinodds
created_at: 2024-06-06 16:06:01
path:
1798622866961027536 [2024-06-06] @visakanv
It's my 34th birthday!! 🥳 if any of my writing or videos etc has ever helped you with anything, I would love to hear about it!!
↓
1798748186389205170 [2024-06-06] @goblinodds  
@visakanv yayyyy hbd visa!!!!! i think a lot about that video where you talked about building up self-trust :')
```
```json
{
  "targets": []
}
```
*Vague description of a video with no title or link. Not findable.*

### Ephemeral Personal Artifact (System Prompt)
```text
PATH 1
path_target_tweet_id: 1798476943278711287
username: goblinodds
created_at: 2024-06-05 22:08:12
path:
1782957877856018514 [2024-04-24] @eigenrobot
Don't worry about formalities. Please be as terse as possible...
[full system prompt pasted inline]
↓
1798476943278711287 [2024-06-05] @goblinodds  
@eigenrobot @ParamMoon wow i'm stealing this
might not even change eigenrobot to goblinodds
```
```json
{
  "targets": []
}
```
*A pasted system prompt is an ephemeral personal artifact, not a named findable object anyone could search for.*

### Disendorsement of a Substance, Not a Named Product
```text
PATH 1
path_target_tweet_id: 1798024810926379063
username: eigenrobot
created_at: 2024-06-04 16:11:35
path:
1798024810926379063 [2024-06-04] @eigenrobot  
going to break character for a moment and acknowledge that the optimal amount of ayahuasca is in fact probably zero
  [Quoting Tweet (missing)]
```
```json
{
  "targets": []
}
```
*Disendorsement of a substance or experience, not a named product or service. Substances and abstract experiences are not valid targets.*

### Broad Genre Praise, Not a Specific Work
```text
PATH 1
path_target_tweet_id: 1798754820955975956
username: eigenrobot
created_at: 2024-06-06 16:41:35
path:
1798754820955975956 [2024-06-06] @eigenrobot  
chivalric romance remains the premier thread of literature in the western tradition exploring the nature of moral perfection
in this thread i discuss the tale of gawain and his failed pursuit of that exalted state
https://x.com/eigenrobot/status/1421308488312004609
```
```json
{
  "targets": []
}
```
*Praising a broad literary genre is too diffuse. A valid endorsement must name a specific work or author.*

### Missing Quoted Content — Even With Positive Language
```text
PATH 1
path_target_tweet_id: 1796971209865023930
username: daniellefong
created_at: 2024-06-01 14:00:00
path:
1796971209865023930 [2024-06-01] @daniellefong  
these are incredible https://x.com/rainmaker1973/status/1796968703189282935
  [Quoting Tweet 1796968703189282935 (missing)]
```
```json
{
  "targets": []
}
```
*Positive language is not enough when the actual object is missing and cannot be identified.*

### Article Used as Evidence, Not Endorsed
```text
PATH 1
path_target_tweet_id: 1796977337701208088
username: daniellefong
created_at: 2024-06-01 14:30:00
path:
1796977337701208088 [2024-06-01] @daniellefong  
@bsansouci @LondonBreed as for building, it's physically possible to match the Empire State Building. Construction was started and finished in 410 days. https://patrickcollison.com/fast
```
```json
{
  "targets": []
}
```
*Uses the linked article as evidence for an argument, not as the subject of endorsement.*

### Photo-Dependent Target
```text
PATH 1
path_target_tweet_id: 1797794351894110293
username: goblinodds
created_at: 2024-06-04 00:55:49
path:
[...long thread about color palettes...]
1797794351894110293 [2024-06-04] @goblinodds  
side note this is the coolest belt i've ever seen (it's like $180 on a huge sale)
but given similar belts by this brand (Off-White, looks rad af) i think it's probably very very long lol https://x.com/.../photo/1
```
```json
{
  "targets": []
}
```
*The praise depends on a photo. No specific named product can be identified from the text alone.*

### Infrastructure Project Enthusiasm
```text
PATH 1
path_target_tweet_id: 1798826512550088998
username: daniellefong
created_at: 2024-06-06 21:17:15
path:
1798826512550088998 [2024-06-06] @daniellefong  
mega solar carport, Los Angeles Six Flags (12 MW)
hell yeah https://x.com/.../photo/1
```
```json
{
  "targets": []
}
```
*Cheering a one-off infrastructure project. Not a named book, article, product, or organization that can be recommended to others.*

### Broad National Praise
```text
PATH 1
path_target_tweet_id: 1798406522902368762
username: eigenrobot
created_at: 2024-06-05 17:28:22
path:
1798406522902368762 [2024-06-05] @eigenrobot  
In advance of the 80th anniversary of D-Day, I'd like to remind Americans that our oldest ally, France, maintains a noble soul.
  [Quoting @DannyDeraney: French caretakers take the sand from Omaha Beach...]
```
```json
{
  "targets": []
}
```
*Broad praise of a country as a moral or civilizational entity. Too diffuse.*

### Referencing a Work Rather Than Endorsing It
```text
PATH 1
path_target_tweet_id: 1797500644997108205
username: goblinodds
created_at: 2024-06-03 18:00:00
path:
[...thread about being a gentleman vs a doormat...]
1797500644997108205 [2024-06-03] @goblinodds  
i think a lot of dudes who've been burned dont know the difference
("the trick, william potter, is not minding that it hurts"
https://www.youtube.com/watch?si=6qc5T8yddGTetntT&v=TvQViPBAvPk&feature=youtu.be )
```
```json
{
  "targets": []
}
```
*The quoted line is used as an illustrative reference, not as an active endorsement of the film or video itself.*

---

## Decision Rule

Return an entry only if **all** of these are true:

1. A tweet in the provided paths clearly expresses endorsement or disendorsement — not just sharing, reacting, or using something as evidence.
2. The target is a specific, named, findable object from the valid target list above.
3. The target is identifiable from text alone — no image or missing context required.
4. The endorsement is substantive and durable, not a fleeting reaction or thin reply.

If any condition fails, omit the target from the output.