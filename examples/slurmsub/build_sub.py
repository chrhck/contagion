import json
import os
import time
import pyssub.sbatch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-c", "--configs", required=True, nargs="+", dest="configs")
parser.add_argument("-o", "--outfile", required=True, dest="outfile")
parser.add_argument("--outdir", required=True, dest="outdir")
parser.add_argument("-n", default=1, dest="n_rep", type=int)
args = parser.parse_args()

logfile = "/scratch4/chaack/logs/{macros[jobname]}_"+ time.strftime("%d_%b_%Y_%H_%M_%S", time.gmtime(time.time())) + ".out"

script = pyssub.sbatch.SBatchScript(
    "/scratch4/chaack/software/scripts/contagion/examples/run_contagion.py",
    "-c {macros[config]} --out {macros[outputfile]} -n {macros[n_rep]}"
)

script.options.update({
    "job-name": "{macros[jobname]}",
    "time": "01:30:00",
    "chdir": "/var/tmp",
    "error": logfile,
    "output": logfile
    })

script.transfer_executable = False

scriptfile = args.outfile + ".script"
with open(scriptfile, "w") as stream:
    json.dump(script, stream, cls=pyssub.sbatch.SBatchScriptEncoder)

#njobs = len(args.configs)
#suffix = "_{{:0{width}d}}".format(width=len(str(njobs)))

collection = {}
for config in args.configs:
    jobname = os.path.basename(config)

    collection[jobname] = {
        "script": scriptfile,
        "macros": {
            "config": config,
            "outputfile": os.path.join(args.outdir, os.path.splitext(jobname)[0]+".pickle"),
            "jobname": jobname,
            "n_rep": args.n_rep
            }
        }

with open(args.outfile + ".jobs", "w") as stream:
    json.dump(collection, stream, cls=pyssub.sbatch.SBatchScriptEncoder)
