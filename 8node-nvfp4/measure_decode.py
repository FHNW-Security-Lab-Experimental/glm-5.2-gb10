import json, sys, time, urllib.request
BASE="http://192.168.88.101:8000/v1"; MODEL="glm-5.2-nvfp4"
N=int(sys.argv[1]) if len(sys.argv)>1 else 4000
MX=int(sys.argv[2]) if len(sys.argv)>2 else 450
FILLER=("The quarterly maintenance log records routine checks of the cooling loop, the "
        "power distribution units, and the network fabric across all eight nodes. ")
fill=FILLER*max(1,(N*4)//len(FILLER))
prompt=fill+"\n\nWrite a thorough ~400 word technical analysis of cluster reliability implied by this log. Be detailed."
body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],
    "max_tokens":MX,"temperature":0.0,"seed":7,"stream":True,
    "stream_options":{"include_usage":True}}).encode()
req=urllib.request.Request(BASE+"/chat/completions",data=body,headers={"Content-Type":"application/json"})
t0=time.time();tfirst=None;tlast=t0;chunks=0;usage=None
with urllib.request.urlopen(req,timeout=6000) as r:
    for raw in r:
        l=raw.decode("utf-8","ignore").strip()
        if not l.startswith("data:"):continue
        d=l[5:].strip()
        if d=="[DONE]":break
        try:o=json.loads(d)
        except:continue
        if o.get("usage"):usage=o["usage"]
        ch=o.get("choices") or []
        if not ch:continue
        de=ch[0].get("delta",{})
        if de.get("content") or de.get("reasoning") or de.get("reasoning_content"):
            if tfirst is None:tfirst=time.time()
            chunks+=1;tlast=time.time()
u=usage or {}; ct=u.get("completion_tokens"); pt=u.get("prompt_tokens")
ttft=(tfirst or t0)-t0; win=max(1e-6,tlast-(tfirst or t0))
real=(ct-1)/win if ct else 0
print("prompt_tokens=%s completion=%s TTFT=%.1fs tok/chunk=%.2f REAL_decode=%.1f tok/s wall=%.1fs"
  % (pt, ct, ttft, (ct/chunks if ct and chunks else 0), real, time.time()-t0))
