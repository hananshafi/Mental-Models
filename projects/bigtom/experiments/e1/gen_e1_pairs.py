#!/usr/bin/env python3
"""
E1: ToMi belief minimal-pairs (sensitivity + specificity).

For each scenario we emit 4 controlled stories built from ToMi's own templates/vocab:
  fb       : observer B exits BEFORE the object is moved  -> false belief -> gold = loc1 (original)
  tb       : observer B stays present when object is moved -> true belief  -> gold = loc2 (new)
  fb_pert  : fb + belief-IRRELEVANT distractor (a 3rd agent moves a DIFFERENT object) -> gold = loc1 (unchanged)
  tb_pert  : tb + same distractor                                                     -> gold = loc2 (unchanged)

fb vs tb differ ONLY by the observer's exit sentence -> tests belief SENSITIVITY.
x vs x_pert differ by a belief-irrelevant move -> tests SPECIFICITY (answer must NOT change).
Question is always "Where will B look for the OBJ?" (canonical Sally-Anne, first-order).
Gold is fully determined by construction, so no dependence on any external scorer.
"""
import json, random, re, os

random.seed(42)
ROOT = "third_party/src/tomi/tomi_balanced_story_types/fb_all_test.txt"

def load_vocab(path):
    agents=set(); objs=set(); conts=set(); rooms=set()
    for line in open(path):
        line=line.rstrip("\n")
        m=re.match(r'^\d+ (\w+) entered the (\w+)\.$', line)
        if m: agents.add(m.group(1)); rooms.add(m.group(2))
        m=re.match(r'^\d+ The (\w+) is in the (\w+)\.$', line)
        if m: objs.add(m.group(1)); conts.add(m.group(2))
        m=re.match(r'^\d+ (\w+) moved the (\w+) to the (\w+)\.$', line)
        if m: agents.add(m.group(1)); objs.add(m.group(2)); conts.add(m.group(3))
    return sorted(agents), sorted(objs), sorted(conts), sorted(rooms)

AG, OBJ, CONT, ROOM = load_vocab(ROOT)

def color_noun(c):
    # containers look like "green_bucket", "blue_treasure_chest"
    parts=c.split("_"); return parts[0], "_".join(parts[1:])

def distinct_locs():
    while True:
        a,b=random.sample(CONT,2)
        ca,na=color_noun(a); cb,nb=color_noun(b)
        # avoid substring collisions in scoring: different color AND different noun
        if ca!=cb and na!=nb and a not in b and b not in a:
            return a,b

def loc_disjoint(loc1, loc2):
    """pick loc3 with distinct color AND noun from loc1/loc2 and no substring collision."""
    c1,n1=color_noun(loc1); c2,n2=color_noun(loc2)
    while True:
        c=random.choice(CONT); cc,nc=color_noun(c)
        if cc in (c1,c2) or nc in (n1,n2): continue
        if any(c in x or x in c for x in (loc1,loc2)): continue
        return c

def make_scenario(i):
    A,B,C = random.sample(AG,3)                 # A moves, B is the observer/questioned, C is distractor
    obj, obj2 = random.sample(OBJ,2)
    room, room2 = random.sample(ROOM,2)
    loc1, loc2 = distinct_locs()                # obj original, obj new
    loc3 = loc_disjoint(loc1, loc2)             # distractor container, disjoint from loc1/loc2
    q = f"Where will {B} look for the {obj}?"

    def sents_fb():   # B exits before the move -> false belief
        return [f"{A} entered the {room}.",
                f"{B} entered the {room}.",
                f"The {obj} is in the {loc1}.",
                f"{B} exited the {room}.",
                f"{A} moved the {obj} to the {loc2}."]
    def sents_tb():   # B present at the move -> true belief
        return [f"{A} entered the {room}.",
                f"{B} entered the {room}.",
                f"The {obj} is in the {loc1}.",
                f"{A} moved the {obj} to the {loc2}."]
    # belief-irrelevant distractor: 3rd agent enters/exits an UNRELATED room; NO move,
    # NO reuse of loc1/loc2 (loc3 disjoint). Cannot change B's belief about obj.
    distractor = [f"{C} entered the {room2}.",
                  f"The {obj2} is in the {loc3}.",
                  f"{C} exited the {room2}."]

    rows=[]
    def emit(cond, sents, gold, wrong):
        rows.append(dict(pair_id=i, condition=cond,
                         story=" ".join(sents), question=q,
                         answer=gold, wrong_answer=wrong,
                         observer=B, obj=obj, loc1=loc1, loc2=loc2))
    emit("fb", sents_fb(),          gold=loc1, wrong=loc2)
    emit("tb", sents_tb(),          gold=loc2, wrong=loc1)
    emit("fb_pert", sents_fb()+distractor, gold=loc1, wrong=loc2)
    emit("tb_pert", sents_tb()+distractor, gold=loc2, wrong=loc1)
    return rows

N=500
out=[]
for i in range(N):
    out.extend(make_scenario(i))
os.makedirs("projects/bigtom/runs/e1", exist_ok=True)
with open("projects/bigtom/runs/e1/e1_pairs.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print(f"wrote {len(out)} rows ({N} scenarios x 4 conditions)")
print("example fb :", out[0]["story"], "|| Q:", out[0]["question"], "|| gold:", out[0]["answer"])
print("example tb :", out[1]["story"], "|| gold:", out[1]["answer"])
print("example pert:", out[2]["story"][:200], "...")
