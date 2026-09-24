#generate job files and submit to cluster for condition E

import subprocess
import sys
from datetime import datetime
from pathlib import Path


WORKSPACE_DIR = Path(__file__).parent.parent.parent
JOB_FILES_DIR = WORKSPACE_DIR / "job_files" / "condition_E"

_TEMPLATE = JOB_FILES_DIR / "run_condition_E.template"

# mainsubmission function
def submit(config: dict) -> str:
    template_file = _TEMPLATE
    if not template_file.exists():
        print(f"✗ Template not found: {template_file}", file=sys.stderr)
        sys.exit(1)

    # put configuration in template
    job_content = template_file.read_text().format(**config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_file  = JOB_FILES_DIR / f"generated_cE_{config['run_name']}_{timestamp}.job"
    job_file.write_text(job_content)

    # actual submission
    cmd = ["sbatch", str(job_file)]
    print(f"Submitting: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"✓ Submitted: job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"✗ sbatch failed:\n  stdout: {e.stdout}\n  stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)
