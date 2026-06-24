#!/bin/bash
# Wait for v2 to finish
while pgrep -f 'outputs_alphadraft_zero_v2' > /dev/null 2>&1; do
    sleep 10
done
echo '[v3] v2 training finished, starting v3 continuation...'

# Run v3: 12 more iterations, warm-starting from v2 best checkpoint
cd ~/marshall/alphadraft_test
CUDA_VISIBLE_DEVICES=2 python3 -u fantasy_alphadraft_zero.py     --iterations 12 --episodes-per-iteration 20     --mcts-simulations 10 --shortlist-size 32     --checkpoint outputs_alphadraft_zero_v2/alphadraft_zero_model.pt     --validation-team-count 4 --validation-frequency 2     --output-dir outputs_alphadraft_zero_v3
