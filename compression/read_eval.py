import json

def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}

cv = load("eval_chatv1.jsonl")
sc = load("eval_scaleup.jsonl")

print("==================== MULTI-TURN ====================")
for k in sorted(x for x in cv if cv[x]["type"] == "multiturn"):
    c, s = cv[k], sc[k]
    print("##### %s | check: %s" % (k, c["checks"][0]))
    for i, u in enumerate(c["turns"]):
        print("  [T%d user] %s" % (i + 1, u))
        print("   chat-v1: %s" % c["answers"][i][:200].replace("\n", " "))
        print("   scaleup: %s" % s["answers"][i][:200].replace("\n", " "))
    print()

print("==================== SINGLE (format/fact) ====================")
for k in sorted(x for x in cv if cv[x]["type"] == "single"):
    c, s = cv[k], sc[k]
    if c["category"] not in ("format", "fact"):
        continue
    print("##### %s [%s] %s" % (k, c["category"], c["prompt"][:90]))
    print("   chat-v1: %s" % c["answer"][:200].replace("\n", " "))
    print("   scaleup: %s" % s["answer"][:200].replace("\n", " "))
    print()
