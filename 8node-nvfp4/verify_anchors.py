import sys
patch = open(sys.argv[1]).read()
live  = open(sys.argv[2]).read()
SENT = "# GLM52_DCP_PERF2"
def extract(name):
    i = patch.index("    %s = (" % name)
    lp = patch.index("(", i)
    j = patch.index('\\n")', lp)          # literal backslash-n quote close-paren = tuple end
    expr = patch[lp:j+4]
    return eval(expr)
for name in ("a_r1", "a_r3", "a_r2"):
    a = extract(name)
    print("%-5s len=%-4d in_live=%s" % (name, len(a), a in live))
