import sys

#python wrapper_src_trg.py NEWS12-Jn-Jk-Dev-Jk.txt NEWS12-Jn-Jk-Dev-Jn.txt >NEWS12-Jn-Jk-Dev-ref.xml 2>NEWS12-Jn-Jk-Dev-src.xml

if len(sys.argv) != 3:
    print >>sys.stderr, 'Usage: %s src-plain trg-plain >ref.xml 2>src.xml'%sys.argv[0]
    sys.exit()

src = sys.argv[1]
trg = sys.argv[2]

title_beg =\
"<?xml version=\"1.0\" encoding=\"utf-8\"?>" + \
"\n<TransliterationCorpus" + \
"\n  TargetLang = \"JP\"" + \
"\n  CorpusSize = \"3871\"" + \
"\n  CorpusFormat = \"UTF8\"" + \
"\n  SourceLang = \"English\"" + \
"\n  CorpusID = \"Western_Name_Jp_Transliteration\"" + \
"\n  CorpusType = \"\">" 

title_end = "</TransliterationCorpus>"
sent_end = "</Name>"

src_lines=open(src).readlines()
trg_lines=open(trg).readlines()

print title_beg
print >>sys.stderr, title_beg

for i, src_line in enumerate(src_lines):
    src_wrapper = "<SourceName>%s</SourceName>"%"".join(src_line.strip().split())
    trg_wrapper = "<TargetName ID=\"1\">%s</TargetName>"%"".join(trg_lines[i].strip().split())
    sent_beg = "<Name id=\"%d\">"%(i+1)
    print sent_beg
    print src_wrapper
    print trg_wrapper
    print sent_end
    print >>sys.stderr,sent_beg
    print >>sys.stderr,src_wrapper
    print >>sys.stderr,sent_end
print title_end
print >>sys.stderr, title_end
    
