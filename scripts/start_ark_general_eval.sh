#!/bin/bash

model_name=${1:-"sybil"}

# Launch GeneralEval web app
GE_HOME=${GE_HOME:-"/general_eval_app"}
${GE_HOME}/ge_venv/bin/streamlit run ${GE_HOME}/general_eval/app.py --browser.gatherUsageStats false &
sleep 5

# Run regular ark
ark-run ${model_name} &

wait
