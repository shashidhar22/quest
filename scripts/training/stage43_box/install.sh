#!/bin/bash
# Keep the exact code that produced Stage 4.1 as the A/B baseline, then install the new code.
set -eu
export PATH=/home/ubuntu/miniforge3/bin:/usr/local/bin:$PATH
Q=/home/ubuntu/quest; OLD=/home/ubuntu/s43_old
mkdir -p $OLD
[ -f $OLD/cpt_training_lib.py ] || cp $Q/scripts/training/cpt_training_lib.py $Q/scripts/training/cpt_stage3_eval_baseline.py $OLD/
aws s3 cp s3://fh-pi-warren-h-eco/quest/tcrbench2/_transfer/s43.tgz /tmp/s43.tgz --only-show-errors
tar xzf /tmp/s43.tgz -C $Q
aws s3 cp s3://fh-pi-warren-h-eco/quest/tcrbench2/_transfer/s42box.tgz /tmp/s42box.tgz --only-show-errors
tar xzf /tmp/s42box.tgz -C /home/ubuntu          # wave1.sh + s42_check.py for Stage 4.2 wave 1
chmod +x $Q/scripts/training/autostop_when_done.sh
mkdir -p /home/ubuntu/s43_out
python -m pytest --version >/dev/null 2>&1 || pip install -q pytest 2>/dev/null || true   # best effort
cd $Q && PYTHONPATH=$Q python -m pytest tests/test_token_budget_sampler.py tests/test_cpt_resume.py -q -p no:cacheprovider 2>&1 | tail -3 | tee /home/ubuntu/s43_out/install_tests.log
python -c "import sys; sys.path.insert(0,'$Q/scripts/training'); import cpt_training_lib as L; print('new lib ok, cache:', hasattr(L,'_REGION_EVAL_CACHE'), 'resume:', hasattr(L,'resolve_resume_dir'))"
diff -q $OLD/cpt_training_lib.py $Q/scripts/training/cpt_training_lib.py && echo "!! old == new" || echo "old and new differ (expected)"
