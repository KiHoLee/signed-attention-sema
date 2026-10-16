#!/usr/bin/env bash
# Submission gate (writing standard 12): every check exits non-zero on failure.
set -e
cd "$(dirname "$0")/.."
MAIN=main
LIMIT=${LIMIT:-13}
latexmk -C >/dev/null 2>&1; latexmk -pdf -interaction=nonstopmode $MAIN.tex >/dev/null 2>&1 || { echo "FAIL: latexmk"; exit 1; }
pages=$(python3 -c "import fitz,sys;print(len(fitz.open('out/$MAIN.pdf')))")
[ "$pages" -le "$LIMIT" ] || { echo "FAIL: $pages pages (limit $LIMIT)"; exit 1; }
! grep -qE '^!|LaTeX Error|^\./[a-z_]+\.tex:[0-9]+:' out/$MAIN.log || { echo "FAIL: LaTeX errors"; grep -E '^!|LaTeX Error|^\./[a-z_]+\.tex:[0-9]+:' out/$MAIN.log | head; exit 1; }
! grep -q 'Overfull' out/$MAIN.log || { echo "FAIL: overfull boxes"; grep Overfull out/$MAIN.log | head; exit 1; }
! grep -qi 'undefined' out/$MAIN.log || { echo "FAIL: undefined references"; grep -i undefined out/$MAIN.log | head; exit 1; }
# mangled control sequences and bare macro stubs (12.11)
! perl -ne 'print "$ARGV:$.: $_" if /[\x00-\x08\x0b\x0c\x0e-\x1f]/' $MAIN.tex references.bib fig1_system.tex | grep -q . || { echo "FAIL: control character"; exit 1; }
! grep -nE '(^|[^\\a-zA-Z])(imes|ef\{|abel\{|extbf\{|rac\{|ootnote\{|ite\{|egin\{|nd\{)' $MAIN.tex | grep -q . || { echo "FAIL: bare macro stub"; exit 1; }
# banned words and prose punctuation (4.1, 3.6)
python3 - <<'PY'
import re,sys
s=open("main.tex",encoding="utf-8").read()
body=s[s.index(r"\begin{abstract}"):s.index(r"\bibliographystyle")]
# strip comments, math, and citations before the prose checks
t=re.sub(r"(?<!\\)%.*","",body)
t=re.sub(r"\\begin\{(equation|align|algorithmic|table|tabular)\*?\}.*?\\end\{\1\*?\}","",t,flags=re.S)
t=re.sub(r"\$[^$]*\$","",t); t=re.sub(r"\\[a-zA-Z]+\{[^}]*\}","",t)
banned=["baseline","benchmark","strateg","headline (result|number|gain|claim|figure)","for free","trick","magic","recipe","huge","dramatic","drastic","luckily","fortunately","remarkably","surprisingly","strikingly","cheap","expensive","nice ","neat","elegant","obviously","of course","a lot of","lots of","utilize","leverage","facilitate","elucidate","ameliorate","ascertain","in order to","aforementioned","plethora","myriad","e.g.","i.e.","etc."]
bad=[w for w in banned if re.search(r"\b"+re.escape(w),t,flags=re.I)]
if bad: print("FAIL banned:",bad); sys.exit(1)
semi=[l for l in t.split("\n") if ";" in l]
if semi: print("FAIL prose semicolon:",semi[:3]); sys.exit(1)
dash=[l for l in t.split("\n") if "---" in l or "--" in l and not re.search(r"\d--\d",l)]
if dash: print("FAIL prose dash:",dash[:3]); sys.exit(1)
PY
# bibliography: every entry cited, every cite present
python3 - <<'PY'
import re,sys
s=open("main.tex",encoding="utf-8").read(); b=open("references.bib",encoding="utf-8").read()
c=set(k.strip() for m in re.finditer(r"\\(?:bstctl)?cite\{([^}]*)\}",s) for k in m.group(1).split(","))
k=set(re.findall(r"@\w+\{([^,]+),",b))
if c!=k: print("FAIL bib:",sorted(c-k),sorted(k-c)); sys.exit(1)
PY
echo "gate passed: $pages pages, 0 errors, 0 overfull"
