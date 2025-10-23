
import re
from typing import List, Dict

class RuleBasedDetector:
    def __init__(self):
        self.rules = {
            'python': [
                {'title':'Dangerous eval/exec usage','pattern':r'\b(eval|exec)\s*\(','severity':'High','recommendation':'Avoid eval/exec; use safe parsing.'},
                {'title':'Hardcoded secret','pattern':r"(api_key|secret|token|password)\s*=\s*['\"][^'\"]+['\"]",'severity':'High','recommendation':'Move secrets to environment variables.'},
                {'title':'Possible SQL via string concat','pattern':r"\bexecute\b.*\+|SELECT.*\+|INSERT.*\+","severity":"High","recommendation":"Use parameterized queries."},
                {'title':'Weak hash (md5|sha1)','pattern':r'hashlib\.(md5|sha1)\(','severity':'Medium','recommendation':'Use SHA256 or bcrypt.'},
            ],
            'javascript': [
                {'title':'InnerHTML assignment','pattern':r'\.innerHTML\s*=','severity':'High','recommendation':'Sanitize/encode user input.'},
                {'title':'eval usage','pattern':r'\beval\s*\(','severity':'High','recommendation':'Avoid eval.'},
                {'title':'document.write XSS','pattern':r'document\.write\(','severity':'High','recommendation':'Escape user input.'},
            ]
        }

    def analyze(self, code: str, language: str='python') -> List[Dict]:
        checks = self.rules.get(language.lower(), [])
        findings = []
        for r in checks:
            for m in re.finditer(r['pattern'], code, re.IGNORECASE):
                findings.append({
                    'title': r['title'],
                    'severity': r['severity'],
                    'span': (m.start(), m.end()),
                    'snippet': code[max(0,m.start()-80):m.end()+80],
                    'recommendation': r['recommendation']
                })
        return findings
