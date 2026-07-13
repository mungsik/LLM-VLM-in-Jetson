import json, urllib.request
SYS = "당신은 한국어로 답하는 친절한 AI 비서입니다."
def chat(port, model, messages):
    body = json.dumps({"model":model,"messages":messages,"temperature":0.3,"max_tokens":150,
                       "frequency_penalty":0.4,"presence_penalty":0.3,"repetition_penalty":1.0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body, {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"].strip()

# 시나리오: (턴1 유저, 턴2 유저, 턴3 유저)
SCEN = [
  ("역할/이름 기억",
   "너의 역할은 가짜연구소 마을을 지키는 NPC고, 내 이름은 멍식이야.",
   "내 이름이 뭐고, 너의 역할이 뭐라고?",
   "그럼 마을에 위험이 오면 넌 어떻게 할 거야?"),
  ("주제 이어가기",
   "나는 파이썬으로 웹 크롤러를 만들고 있어.",
   "requests랑 BeautifulSoup 중에 뭐가 좋을까?",
   "방금 말한 거 중에 초보한테 추천하는 걸로 예시 코드 짧게 보여줘."),
]
for name, port, model in [("v2", 8003, "phi4-sft-v2"), ("v3", 8002, "phi4-sft-v3")]:
    print(f"################## {name} ##################")
    for title, t1, t2, t3 in SCEN:
        msgs=[{"role":"system","content":SYS}]
        print(f"----- {title} -----")
        for i,u in enumerate([t1,t2,t3],1):
            msgs.append({"role":"user","content":u})
            r=chat(port,model,msgs)
            msgs.append({"role":"assistant","content":r})
            print(f"  U{i}: {u}")
            print(f"  A{i}: {r[:200]}")
        print()
