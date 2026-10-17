import json
from pathlib import Path
from tokenizers import Tokenizer
t=Tokenizer.from_file('/models/92B/tokenizer.json')
encode=lambda text:t.encode(text,add_special_tokens=False).ids
head=encode('<|message_start|>user\nRemember that the access code is 7319.\n')
filler=encode('This entry records routine maintenance with no change to the access code.\n')
tail=encode('\nWhat is the access code? Reply with the code only.<|message_end|><|message_start|>assistant\n</think>')
cases=[]
for n in (529,1024,2040):
    count=n-len(head)-len(tail)
    ids=head+(filler*((count+len(filler)-1)//len(filler)))[:count]+tail
    assert len(ids)==n
    cases.append(dict(case=f'access-code-{n}',input_ids=ids,output_ids=[0]*8))
Path('swa-wrap-reference-input.json').write_text(json.dumps(cases,indent=2)+'\n')
