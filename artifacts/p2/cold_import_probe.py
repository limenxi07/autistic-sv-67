"""One-shot cold artifact import/setup measurement; run in a fresh pinned process."""
import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0,str(Path.cwd()))
from tools.benchmark_p1 import load_agent,provenance

source,output=sys.argv[1:]
start=perf_counter();agent=load_agent(source);imported=perf_counter()
instance=agent();constructed=perf_counter()
report=provenance()
report.update(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              cold_import_seconds=imported-start,construction_seconds=constructed-imported,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              torch_imported='torch' in sys.modules,
              p_core_cpus=Path('/sys/bus/event_source/devices/cpu_core/cpus').read_text().strip(),
              e_core_cpus=Path('/sys/bus/event_source/devices/cpu_atom/cpus').read_text().strip(),
              process_affinity=sorted(os.sched_getaffinity(0)))
Path(output).write_text(json.dumps(report,indent=2)+'\n')
