import ast, symtable, builtins
from pathlib import Path
from agents import v4_backend as b
src=Path('agents/v4_backend.py').read_text();tree=ast.parse(src)
def extract(name):
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.get_source_segment(src,n)
code='\n\n'.join(extract(n) for n in ('primitive_search','search','reference_search'))
code=code.replace('primitive_search(', 'v4_primitive_search(').replace('def search(', 'def v4_search(').replace('def reference_search(', 'def v4_reference_search(')
code=code.replace('nonlocal incumbent, first_win','nonlocal incumbent, first_win, deadline').replace('first_win = clock() - began','first_win = clock() - began\n                deadline = min(deadline, clock() + 0.025)')
code=code.replace('if incumbent is None or successor.score > nodes[incumbent][4]:\n                            incumbent = child','if incumbent is None:\n                            deadline = min(deadline, clock() + 0.025)\n                        if incumbent is None or successor.score > nodes[incumbent][4]:\n                            incumbent = child')
def globals_used(table):
 names={s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
 for child in table.get_children():names |= globals_used(child)
 return names
names=sorted(n for n in globals_used(symtable.symtable(code,'v4_search','exec')) if hasattr(b,n) and not hasattr(builtins,n))
header='''"""V4 search derived from frozen P3, with bounded post-win refinement.

Parent bytes remain untouched. Only V4 image planning uses these functions.
A first fast incumbent is publicly verified before starting its 25ms refinement
window; reference incumbents are still independently replayed before dispatch.
"""\n\n'''
header+='from agents.v4_backend import (\n'+''.join('    '+n+',\n' for n in names)+')\n\n'
Path('agents/v4_search.py').write_text(header+code+'\n')
# Add dedicated image-agent dispatch methods, retaining parent cache/rebind logic.
def method(cls,name):
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 m=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return '\n'.join(src.splitlines()[m.lineno-1:m.end_lineno])
# ReferenceAgent is defined as an alias to an earlier Agent; identify via source line.
import inspect
refstep=inspect.getsource(b.ReferenceAgent.step).replace('reference_search(', 'v4_reference_search(')
refclass='\n\nclass V4ReferenceAgent(ReferenceAgent):\n'+refstep+'\n'
p=Path('agents/v4_planner.py');s=p.read_text();s=s.replace('class PlannerAdapter(',refclass+'\n\nclass PlannerAdapter(')
methods=method('PlannerAdapter','step').replace('return super().step(state)','return self._search_step(state)')
searchstep=method('Agent','step') # AST selects final Agent class, which is unique named Agent here
# There are multiple Agent definitions? ensure selected method is fast.
assert 'result = search(' in searchstep
searchstep=searchstep.replace('def step(', 'def _search_step(').replace('result = search(', 'result = v4_search(')
fallback=method('PlannerAdapter','_fallback').replace('ReferenceAgent()', 'V4ReferenceAgent()')
s+='\n'+methods+'\n\n'+searchstep+'\n\n'+fallback+'\n'
used=globals_used(symtable.symtable(s,'planner','exec'))
imports=sorted(n for n in used if hasattr(b,n) and not hasattr(builtins,n) and n not in ('PlannerAdapter','reference_state_key','StructuredAgent','BackendPlannerAdapter'))
s='from agents.v4_backend import (\n'+''.join('    '+n+',\n' for n in imports)+')\nfrom agents.v4_search import v4_search, v4_reference_search\n\n'+s
p.write_text(s)
p=Path('tools/package_v4.py');s=p.read_text().replace('    "v4_planner",','    "v4_search",\n    "v4_planner",');p.write_text(s)
