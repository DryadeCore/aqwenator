"""Which requests make up the time-to-first-token tail of an aq-serve run.

Usage: python scripts/trace_tail.py <server.log>   (server started with AQ_TRACE=1)
Reads the `[req]` lines: prompt and reused tokens, queue wait, time to first token and the
steps the request took part in; prints the slowest requests and the tail by new-token bucket.
"""
import re, sys
rows=[]
for l in open(sys.argv[1], errors='ignore'):
    if l.startswith('[req]'):
        m=re.search(r'id (\d+) prompt (\d+) reused (\d+) wait_ms ([\d.]+) ttft_ms ([\d.]+) steps (\d+)', l)
        if m:
            i,p,r,w,t,s=m.groups(); rows.append((int(i),int(p),int(r),float(w),float(t),int(s)))
print("requests", len(rows))
import statistics as st
def q(v,p): v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
tt=[r[4] for r in rows]
print("server-side ttft p50/p90/p99/max", [round(q(tt,p)) for p in (.5,.9,.99)], round(max(tt)))
print("miss (reused==0):", sum(1 for r in rows if r[2]==0), " partial (<90% reuse, prompt>4k):", sum(1 for r in rows if r[1]>4000 and 0<r[2]<0.9*r[1]))
slow=sorted(rows,key=lambda r:-r[4])[:25]
print("slowest 25:  prompt reused new wait_ms ttft_ms steps  ms/step")
for i,p,r,w,t,s in slow:
    print(f"  {p:6d} {r:6d} {p-r:6d} {w:8.1f} {t:8.1f} {s:5d} {(t-w)/max(s,1):7.1f}")
# share of tail by cause
tail=[r for r in rows if r[4]>=q(tt,.95)]
print("top 5%: n", len(tail), "median new", st.median([r[1]-r[2] for r in tail]), "median wait", st.median([r[3] for r in tail]), "median steps", st.median([r[5] for r in tail]),
      "wait-dominated", sum(1 for r in tail if r[3]>0.5*r[4]), "big-new(>=3k)", sum(1 for r in tail if r[1]-r[2]>=3000), "miss", sum(1 for r in tail if r[2]==0))
for lo,hi in ((0,256),(256,1000),(1000,3000),(3000,10**9)):
    b=[r for r in rows if lo<=r[1]-r[2]<hi]
    if b: print(f"new {lo}-{hi}: n {len(b)} ttft p50/p90/max {q([r[4] for r in b],.5):.0f}/{q([r[4] for r in b],.9):.0f}/{max(r[4] for r in b):.0f}  wait p50/p90/max {q([r[3] for r in b],.5):.0f}/{q([r[3] for r in b],.9):.0f}/{max(r[3] for r in b):.0f}  steps p50/p90 {q([r[5] for r in b],.5)}/{q([r[5] for r in b],.9)}")
