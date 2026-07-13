import json, urllib.request, math, sys
port = sys.argv[1]; model = sys.argv[2]
text = open("artifacts/holdout_ko.txt", encoding="utf-8").read()
# 문자 ~1500자 청크(≈512토큰) 40개
chunks, step = [], 1500
for i in range(0, min(len(text), step*40), step):
    c = text[i:i+step].strip()
    if len(c) > 200: chunks.append(c)
tot_nll, tot_tok = 0.0, 0
for c in chunks:
    body = json.dumps({"model":model,"prompt":c,"max_tokens":1,"echo":True,"logprobs":1,"temperature":0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", body, {"Content-Type":"application/json"})
    lp = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["logprobs"]["token_logprobs"]
    for x in lp:
        if x is not None:
            tot_nll += -x; tot_tok += 1
print(f"PPL({model}) = {math.exp(tot_nll/tot_tok):.4f}  over {tot_tok} tokens, {len(chunks)} chunks")
