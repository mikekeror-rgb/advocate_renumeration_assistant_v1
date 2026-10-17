import csv,sys
rows=list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else 'eval/judge_handcheck.csv')))
done=[r for r in rows if r['MY_VERDICT (True/False)'].strip()]
agree=sum(r['judge_verdict'].strip().lower()==r['MY_VERDICT (True/False)'].strip().lower() for r in done)
print(f"{agree}/{len(done)} agree ({agree/len(done):.0%})")
for r in done:
    if r['judge_verdict'].strip().lower()!=r['MY_VERDICT (True/False)'].strip().lower(): print('DISAGREE',r['run'],r['id'])
