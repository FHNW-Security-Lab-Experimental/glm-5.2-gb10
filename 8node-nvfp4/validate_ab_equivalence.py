import json, sys, urllib.request
BASE="http://192.168.88.101:8000/v1"; MODEL="glm-5.2-nvfp4"
TAG=sys.argv[1] if len(sys.argv)>1 else "ref"
FILLER=("The quarterly maintenance log records routine checks of the cooling loop, the "
        "power distribution units, and the network fabric across all eight nodes. ")
def post(msgs, mx):
    body=json.dumps({"model":MODEL,"messages":msgs,"max_tokens":mx,"temperature":0.0,
        "seed":12345,"stream":False}).encode()
    r=urllib.request.Request(BASE+"/chat/completions",data=body,headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=1800).read())
    ch=d["choices"][0]; m=ch["message"]
    return {"content":(m.get("content") or ""),"reason_len":len(m.get("reasoning_content") or ""),
            "finish":ch.get("finish_reason"),"ntok":d.get("usage",{}).get("completion_tokens")}
sec="The vault access code for spark-a175 is PLUM-4731-OWL."
fill=FILLER*max(1,(32000*4)//len(FILLER)); cut=len(fill)//2
hay=fill[:cut]+"\n\n"+sec+"\n\n"+fill[cut:]
cases={}
cases["math"]=post([{"role":"user","content":"What is 17*23? Reply with the number then the word DONE."}],400)
cases["needle32k"]=post([{"role":"user","content":hay+"\n\nWhat is the vault access code for spark-a175? Reply with only the code."}],400)
cases["gen32k"]=post([{"role":"user","content":hay+"\n\nIn exactly two sentences, summarize what the maintenance log covers."}],600)
print(json.dumps({"tag":TAG,"cases":cases}))
