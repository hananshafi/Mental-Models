#!/usr/bin/env python3
"""Extract real ToMi first-order belief questions (true- and false-belief) as a
sanity set to confirm the loaded checkpoint reproduces the reported ToMi behavior."""
import json, os, re, random
random.seed(0)
TXT="third_party/src/tomi/tomi_balanced_story_types/fb_all_test.txt"
TRACE="third_party/src/tomi/tomi_balanced_story_types/fb_all_test.trace"

# parse txt into (story, question, answer) blocks aligned with trace lines
records=[]
story=[]
for line in open(TXT):
    line=line.rstrip("\n")
    if not line: continue
    if "\t" in line:  # question line: "N question<TAB>answer<TAB>facts"
        parts=line.split("\t")
        q=re.sub(r'^\d+ ','',parts[0]).strip()
        ans=parts[1].strip()
        records.append({"story":" ".join(story),"question":q,"answer":ans})
        # a new block starts after each question if numbering resets; handle by resetting when next '1 ' seen
        story=[]
    else:
        num=line.split(" ",1)[0]
        txt=re.sub(r'^\d+ ','',line)
        if num=="1": story=[txt]
        else: story.append(txt)

traces=[l.rstrip("\n") for l in open(TRACE) if l.strip()]
assert len(records)==len(traces), f"misalign {len(records)} vs {len(traces)}"

fb=[]; tb=[]
for rec,tr in zip(records,traces):
    fields=tr.split(",")
    qtype=fields[-2]; stype=fields[-1]
    rec["qtype"]=qtype; rec["stype"]=stype
    if qtype.startswith("first_order") and "tom" in qtype:
        if qtype.endswith("_tom") and "no_tom" not in qtype:  # false-belief established
            fb.append(rec)
        elif "no_tom" in qtype:
            tb.append(rec)

random.shuffle(fb); random.shuffle(tb)
sel=fb[:60]+tb[:60]
os.makedirs("projects/bigtom/runs/e1", exist_ok=True)
with open("projects/bigtom/runs/e1/e1_sanity.jsonl","w") as f:
    for r in sel: f.write(json.dumps(r)+"\n")
print(f"sanity: {len(sel)} items ({min(60,len(fb))} first-order FALSE-belief, {min(60,len(tb))} first-order TRUE-belief)")
print("example FB:", fb[0]["story"][:150], "| Q:", fb[0]["question"], "| A:", fb[0]["answer"], "|", fb[0]["qtype"])
